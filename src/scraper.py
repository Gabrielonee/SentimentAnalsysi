from __future__ import annotations
import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests
from tqdm import tqdm

from .config import SUBREDDITS, START_DATE, END_DATE, DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
POSTS_URL = f"{ARCTIC_BASE}/posts/search"
COMMENTS_URL = f"{ARCTIC_BASE}/comments/search"
PAGE_SIZE = 100             # massimo accettato dall'endpoint
DEFAULT_TIMEOUT = 60        # secondi
THROTTLE_SECONDS = 0.5      # backoff gentile fra chiamate
MAX_RETRIES = 5


@dataclass
class ArcticShiftClient:
    """Wrapper minimale con retry/backoff esponenziale."""
    user_agent: str = "milan-sentiment-research/0.2 (academic)"
    timeout: int = DEFAULT_TIMEOUT

    def __post_init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent,
                                     "Accept": "application/json"})

    def _get(self, url: str, params: dict) -> dict:
        delay = 1.0
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    logger.warning("Rate-limited (attempt %d). Sleep %.1fs", attempt, delay)
                    time.sleep(delay); delay *= 2; continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning("HTTP error (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
                time.sleep(delay); delay *= 2
        raise RuntimeError(f"Arctic Shift unreachable: {url} {params}")

    # -------------------------------------------------------------------------
    # Iteratori paginati
    # -------------------------------------------------------------------------
    def iter_posts(self, subreddit: str, after_ts: int, before_ts: int) -> Iterator[dict]:
        return self._iter("posts", POSTS_URL, subreddit, after_ts, before_ts)

    def iter_comments(self, subreddit: str, after_ts: int, before_ts: int) -> Iterator[dict]:
        return self._iter("comments", COMMENTS_URL, subreddit, after_ts, before_ts)

    def iter_comments_of_post(self, link_id: str) -> Iterator[dict]:
        """Tutti i commenti di un singolo post identificato da link_id (t3_xxx)."""
        cursor = 0
        last_id: str | None = None
        while True:
            params = {"link_id": link_id, "limit": PAGE_SIZE,
                      "sort": "asc", "after": cursor}
            payload = self._get(COMMENTS_URL, params=params)
            batch = payload.get("data") or []
            if not batch: return
            for item in batch:
                if item.get("id") == last_id: continue
                yield item
            last = batch[-1]; last_id = last.get("id")
            new_cursor = int(last.get("created_utc", cursor)) + 1
            if new_cursor <= cursor: return
            cursor = new_cursor
            time.sleep(THROTTLE_SECONDS)
            if len(batch) < PAGE_SIZE: return

    def _iter(self, kind: str, url: str, subreddit: str,
              after_ts: int, before_ts: int) -> Iterator[dict]:
        """Paginazione forward usando il timestamp dell'ultimo elemento."""
        cursor = after_ts
        last_id: str | None = None
        while cursor < before_ts:
            params = {
                "subreddit": subreddit,
                "limit": PAGE_SIZE,
                "sort": "asc",
                "after": cursor,
                "before": before_ts,
            }
            payload = self._get(url, params=params)
            batch = payload.get("data") or []
            if not batch:
                return
            yielded = 0
            for item in batch:
                # Skip duplicati eventuali al confine fra pagine
                if item.get("id") == last_id:
                    continue
                yield item
                yielded += 1
            # Aggiorna cursore al timestamp dell'ultimo + 1 secondo (forward)
            last = batch[-1]
            last_id = last.get("id")
            new_cursor = int(last.get("created_utc", cursor)) + 1
            if new_cursor <= cursor:
                # Difensivo: evita loop se il server restituisse risultati senza progresso
                logger.debug("[%s] cursor non avanza, esco. n=%d", kind, yielded)
                return
            cursor = new_cursor
            time.sleep(THROTTLE_SECONDS)
            if len(batch) < PAGE_SIZE:
                return   # ultima pagina


def _ts_to_dt(ts: float | int | None) -> datetime | None:
    if ts is None: return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except (ValueError, OSError):
        return None


def _full_url(permalink: str | None) -> str | None:
    if not permalink: return None
    if permalink.startswith("http"): return permalink
    return f"https://www.reddit.com{permalink}"


def normalize_post(p: dict) -> dict:
    return {
        "id": f"t3_{p.get('id')}",
        "type": "post",
        "created_utc": _ts_to_dt(p.get("created_utc")),
        "author": p.get("author") or "[deleted]",
        "subreddit": p.get("subreddit"),
        "title": p.get("title") or "",
        "body": p.get("selftext") or "",
        "score": int(p.get("score") or 0),
        "permalink": _full_url(p.get("permalink")),
        "parent_id": None,
    }


def normalize_comment(c: dict) -> dict:
    return {
        "id": f"t1_{c.get('id')}",
        "type": "comment",
        "created_utc": _ts_to_dt(c.get("created_utc")),
        "author": c.get("author") or "[deleted]",
        "subreddit": c.get("subreddit"),
        "title": "",
        "body": c.get("body") or "",
        "score": int(c.get("score") or 0),
        "permalink": _full_url(c.get("permalink")),
        "parent_id": c.get("link_id") or c.get("parent_id"),
    }


def _month_windows(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    """Genera finestre mensili (start_incluso, end_escluso)."""
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cur < end:
        if cur.month == 12:
            nxt = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            nxt = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)
        yield max(cur, start), min(nxt, end)
        cur = nxt


def scrape_subreddit(client: ArcticShiftClient, subreddit: str,
                     start: datetime, end: datetime,
                     include_posts: bool = True,
                     include_comments: bool = True,
                     max_records: int | None = None) -> list[dict]:
    rows: list[dict] = []
    windows = list(_month_windows(start, end))
    for w_start, w_end in tqdm(windows, desc=f"r/{subreddit}", unit="month"):
        a, b = int(w_start.timestamp()), int(w_end.timestamp())
        if include_posts:
            for p in client.iter_posts(subreddit, a, b):
                rows.append(normalize_post(p))
                if max_records and len(rows) >= max_records: return rows
        if include_comments:
            for c in client.iter_comments(subreddit, a, b):
                rows.append(normalize_comment(c))
                if max_records and len(rows) >= max_records: return rows
    return rows


def scrape_comments_from_posts(client: ArcticShiftClient,
                               post_ids: list[str],
                               max_records: int | None = None) -> list[dict]:
    """Scarica i commenti relativi a una lista di post (id `t3_xxx`)."""
    rows: list[dict] = []
    for pid in tqdm(post_ids, desc="comments per post", unit="post"):
        if not pid.startswith("t3_"): pid = f"t3_{pid}"
        for c in client.iter_comments_of_post(pid):
            rows.append(normalize_comment(c))
            if max_records and len(rows) >= max_records: return rows
    return rows


def main(start: datetime | None = None,
         end: datetime | None = None,
         subreddits: list[str] | None = None,
         include_posts: bool = True,
         include_comments: bool = True,
         from_posts: Path | None = None,
         max_records: int | None = None,
         out_path: Path | None = None):
    start = start or START_DATE
    end = end or END_DATE
    subreddits = subreddits or SUBREDDITS
    client = ArcticShiftClient()

    # --- Modalità "solo commenti dei post già scaricati" --------------------
    if from_posts:
        if out_path is None: out_path = DATA_DIR / "comments_from_posts.parquet"
        src = pd.read_parquet(from_posts)
        posts = src[src["type"] == "post"] if "type" in src.columns else src
        post_ids = posts["id"].dropna().astype(str).tolist()
        logger.info("Scarico commenti di %d post da %s", len(post_ids), from_posts)
        rows = scrape_comments_from_posts(client, post_ids, max_records=max_records)
        df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
        df.to_parquet(out_path, index=False)
        logger.info("Salvato %s (%d commenti)", out_path, len(df))
        return df

    # --- Modalità standard: crawl temporale ---------------------------------
    if out_path is None: out_path = DATA_DIR / "raw_reddit.parquet"
    all_rows: list[dict] = []
    for sr in subreddits:
        logger.info("Scarico r/%s da %s a %s (posts=%s comments=%s)",
                    sr, start.date().isoformat(), end.date().isoformat(),
                    include_posts, include_comments)
        all_rows.extend(scrape_subreddit(
            client, sr, start, end,
            include_posts=include_posts,
            include_comments=include_comments,
            max_records=max_records))
        logger.info("Subtotale r/%s: %d record", sr, len(all_rows))

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["id"])
    df.to_parquet(out_path, index=False)
    logger.info("Salvato %s (%d righe uniche)", out_path, len(df))
    return df


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=_parse_date, default=None,
                    help="Data inizio YYYY-MM-DD (default: config.START_DATE)")
    ap.add_argument("--end", type=_parse_date, default=None,
                    help="Data fine YYYY-MM-DD (default: oggi)")
    ap.add_argument("--subreddit", action="append", default=None,
                    help="Subreddit (ripetibile). Default: config.SUBREDDITS")
    ap.add_argument("--posts-only", action="store_true",
                    help="Scarica solo post (no commenti)")
    ap.add_argument("--comments-only", action="store_true",
                    help="Scarica solo commenti del subreddit (no post)")
    ap.add_argument("--from-posts", type=Path, default=None,
                    help="Parquet di post (es. raw_reddit.parquet): "
                         "scarica SOLO i commenti relativi a quei post via link_id")
    ap.add_argument("--max-records", type=int, default=None,
                    help="Tetto al numero totale di record (debug/smoke test). "
                         "Sconsigliato per crawl completi: rischia di tagliare prima dei commenti.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.posts_only and args.comments_only:
        ap.error("--posts-only e --comments-only sono mutuamente esclusivi")
    main(start=args.start, end=args.end,
         subreddits=args.subreddit,
         include_posts=not args.comments_only,
         include_comments=not args.posts_only,
         from_posts=args.from_posts,
         max_records=args.max_records,
         out_path=args.out)
