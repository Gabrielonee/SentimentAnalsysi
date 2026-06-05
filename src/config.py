from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SUBREDDITS = ["ACMilan"]            # subreddit target
START_DATE = datetime(2022, 8, 1, tzinfo=timezone.utc)   # acquisizione RedBird
END_DATE   = datetime.now(tz=timezone.utc)

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SPACY_IT = "it_core_news_sm"   # ridotto in dev; usare _lg in produzione
SPACY_EN = "en_core_web_sm"

@dataclass
class Executive:
    key: str                       # identificatore breve usato in dataframe e grafo
    display_name: str              # nome esteso per dashboard
    role: str
    period: tuple[str, str | None]
    aliases: list[str] = field(default_factory=list)


EXECUTIVES: list[Executive] = [
    Executive(
        key="cardinale",
        display_name="Gerry Cardinale",
        role="Founder RedBird Capital / Owner",
        period=("2022-08", None),
        aliases=["Gerry Cardinale", "Cardinale", "RedBird", "Red Bird",
                 "Gerald Cardinale", "RBC"],
    ),
    Executive(
        key="furlani",
        display_name="Giorgio Furlani",
        role="CEO",
        period=("2022-10", None),
        aliases=["Giorgio Furlani", "Furlani", "the CEO",
                 "Milan CEO", "AD Milan"],
    ),
    Executive(
        key="maldini",
        display_name="Paolo Maldini",
        role="Direttore Tecnico",
        period=("2022-08", "2023-06"),
        aliases=["Paolo Maldini", "Maldini", "Paolo",
                 "Maldini Sr"],   # nota: disambiguare da Daniel/Cesare
    ),
    Executive(
        key="massara",
        display_name="Frederic Massara",
        role="Direttore Sportivo",
        period=("2022-08", "2023-06"),
        aliases=["Frederic Massara", "Massara", "Ricky Massara"],
    ),
    Executive(
        key="moncada",
        display_name="Geoffrey Moncada",
        role="Direttore Tecnico / Head of Scouting",
        period=("2023-06", None),
        aliases=["Geoffrey Moncada", "Moncada"],
    ),
    Executive(
        key="ibrahimovic",
        display_name="Zlatan Ibrahimović",
        role="Operating Partner / Senior Advisor",
        period=("2023-12", None),
        aliases=["Zlatan Ibrahimovic", "Ibrahimovic", "Ibra",
                 "Zlatan", "Z-Ibra", "Senior Advisor"],
    ),
    Executive(
        key="gazidis",
        display_name="Ivan Gazidis",
        role="CEO uscente",
        period=("2022-08", "2022-12"),
        aliases=["Ivan Gazidis", "Gazidis"],
    ),
]

EXEC_BY_KEY = {e.key: e for e in EXECUTIVES}

TARGET_LANGUAGES = {"it", "en"}
