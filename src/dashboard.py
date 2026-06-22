import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from .config import EXEC_BY_KEY, EXECUTIVES
except ImportError:
    from src.config import EXEC_BY_KEY, EXECUTIVES


_orig_plotly_chart = getattr(st, "_orig_plotly_chart_real", st.plotly_chart)
st._orig_plotly_chart_real = _orig_plotly_chart

def _styled_plotly_chart(fig, *args, **kwargs):
    try:
        cur_h = getattr(fig.layout, "height", None) or 400
        fig.update_layout(height=int(cur_h * 1.45))
        m = fig.layout.margin
        fig.update_layout(margin=dict(
            l=max(m.l or 0, 70), r=max(m.r or 0, 40),
            t=max(m.t or 0, 50), b=max(m.b or 0, 60)))
        fig.update_layout(
            font=dict(size=17, color="#111111", weight="bold"),
            legend=dict(font=dict(size=15, weight="bold")),
            title=dict(font=dict(size=19, weight="bold")),
        )
        fig.update_xaxes(title_font=dict(size=17, weight="bold"),
                         tickfont=dict(size=14, weight="bold"), automargin=True)
        fig.update_yaxes(title_font=dict(size=17, weight="bold"),
                         tickfont=dict(size=14, weight="bold"), automargin=True)
        # ingrandisce/bolda anche le etichette-testo (nodi grafo, valori barre)
        fig.update_traces(textfont=dict(size=16, weight="bold"),
                          selector=dict(type="scatter"))
    except Exception:
        pass
    kwargs.setdefault("theme", None)
    return _orig_plotly_chart(fig, *args, **kwargs)

st.plotly_chart = _styled_plotly_chart


st.set_page_config(
    page_title="Milan Sentiment – Era RedBird",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "output"

MILAN_RED = "#FB1A2D"
MILAN_BLK = "#000000"
ACCENT    = "#1F3A93"
NEU       = "#A0A0A0"


@st.cache_data(show_spinner=False)
def load_sentiment(filename: str = "sentiment_absa.parquet") -> pd.DataFrame:
    df = pd.read_parquet(DATA / filename)
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df["date"] = df["created_utc"].dt.date
    if "executives" in df.columns and len(df) > 0 \
       and isinstance(df["executives"].iloc[0], (list, np.ndarray)):
        df = df.explode("executives").rename(columns={"executives": "executive"})
    return df


def _discover_sentiment_files() -> dict[str, str]:
    candidates = {
        "Vanilla (XLM-RoBERTa)": "sentiment.parquet",
        "ABSA (DeBERTa-v3)":     "sentiment_absa.parquet",
    }
    return {k: v for k, v in candidates.items() if (DATA / v).exists()}


@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame:
    df = pd.read_csv(DATA / "matches.csv", parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_graph() -> dict:
    p = OUT / "graph.json"
    if not p.exists(): return {"nodes": [], "edges": []}
    with open(p, "r", encoding="utf-8") as f: return json.load(f)


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    p = OUT / "metrics.csv"
    if not p.exists(): return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_correlations() -> pd.DataFrame:
    p = OUT / "correlations.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


df_match = load_matches()

st.sidebar.title("AC Milan Sentiment Analysis")
available_backends = _discover_sentiment_files()
if not available_backends:
    st.sidebar.error("Nessun file sentiment trovato in data/.")
    st.stop()
if len(available_backends) > 1:
    backend_label = st.sidebar.radio(
        "Backend sentiment",
        list(available_backends.keys()),
        index=list(available_backends.keys()).index("ABSA (DeBERTa-v3)")
              if "ABSA (DeBERTa-v3)" in available_backends else 0,
        help="Vanilla = sentiment della frase intera, replicato a tutti i dirigenti citati. "
             "ABSA = sentiment specifico verso ciascun dirigente.",
    )
else:
    backend_label = next(iter(available_backends))
    st.sidebar.caption(f"Backend: {backend_label}")
df_sent_all = load_sentiment(available_backends[backend_label])

min_d, max_d = df_sent_all["date"].min(), df_sent_all["date"].max()
date_range = st.sidebar.date_input(
    "Intervallo temporale",
    value=(min_d, max_d),
    min_value=min_d, max_value=max_d,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d_from, d_to = date_range
else:
    d_from, d_to = min_d, max_d

exec_options = [(e.key, e.display_name) for e in EXECUTIVES]
selected_keys = st.sidebar.multiselect(
    "Dirigenti",
    options=[k for k, _ in exec_options],
    default=[k for k, _ in exec_options],
    format_func=lambda k: EXEC_BY_KEY[k].display_name if k in EXEC_BY_KEY else k,
)

min_score = st.sidebar.slider("Score Reddit minimo", 1, 50, 1, 1)

# Filtro globale
mask = (
    (df_sent_all["date"] >= d_from) &
    (df_sent_all["date"] <= d_to) &
    (df_sent_all["executive"].isin(selected_keys)) &
    (df_sent_all["score"] >= min_score)
)
df_sent = df_sent_all[mask].copy()


st.title("Sentiment & Network Analysis")
tab1, tab2, tab3, tab4 = st.tabs(["OVERVIEW", "SENTIMENT", "NETWORK", "PERFORMANCE"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frasi analizzate", f"{len(df_sent):,}")
    c2.metric("Dirigenti monitorati", f"{df_sent['executive'].nunique()}/{len(EXECUTIVES)}")
    c3.metric("Sentiment medio", f"{df_sent['sentiment_score'].mean():.3f}")
    pos_share = (df_sent['sentiment_label'] == 'positive').mean() * 100
    c4.metric("% positivo", f"{pos_share:.1f}%")

    st.subheader("Distribuzione delle menzioni per dirigente")
    counts = (df_sent.groupby("executive").size()
                     .reset_index(name="menzioni"))
    counts["dirigente"] = counts["executive"].map(lambda k: EXEC_BY_KEY[k].display_name)
    counts = counts.sort_values("menzioni", ascending=True)
    fig = px.bar(counts, x="menzioni", y="dirigente", orientation="h",
                 color="menzioni", color_continuous_scale=["#000000", MILAN_RED])
    fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
    fig.update_yaxes(title_text=None)
    fig.update_xaxes(title_text="Numero Menzioni")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mix linguistico e label")
    cc1, cc2 = st.columns(2)
    with cc1:
        lang_share = df_sent["lang"].value_counts(normalize=True).reset_index()
        lang_share.columns = ["lang", "share"]
        fig = px.pie(lang_share, names="lang", values="share", hole=0.5,
                     color_discrete_sequence=[ACCENT, MILAN_RED])
        fig.update_layout(height=300, title="Lingua")
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        lab_share = df_sent["sentiment_label"].value_counts(normalize=True).reset_index()
        lab_share.columns = ["label", "share"]
        fig = px.pie(lab_share, names="label", values="share", hole=0.5,
                     color="label",
                     color_discrete_map={"positive": "#2E8B57", "neutral": NEU, "negative": MILAN_RED})
        fig.update_layout(height=300, title="Sentiment label")
        st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.subheader("Sentiment medio mensile per dirigente")

    show_average = st.checkbox(
        "Mostra media complessiva",
        value=False,
        key="show_average_sentiment"
    )

    df_w = df_sent.copy()

    df_w["month"] = (
        pd.to_datetime(df_w["created_utc"], utc=True)
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = (
        df_w.groupby(["month", "executive"], as_index=False)
        .agg(
            mean_score=("sentiment_score", "mean"),
            n=("sentiment_score", "count")
        )
    )

    monthly["dirigente"] = monthly["executive"].map(
        lambda k: EXEC_BY_KEY[k].display_name
    )

    fig = px.line(
        monthly,
        x="month",
        y="mean_score",
        color="dirigente",
        line_shape="spline",
        markers=True,
        hover_data={"n": True, "mean_score": ":+.3f"},
    )

    if show_average:
        monthly_avg = (
            df_w.groupby("month", as_index=False)
            .agg(
                mean_score=("sentiment_score", "mean"),
                n=("sentiment_score", "count")
            )
        )

        fig.add_scatter(
            x=monthly_avg["month"],
            y=monthly_avg["mean_score"],
            mode="lines+markers",
            name="Media complessiva",
            line_color='black',
            line=dict(width=7)
        )

    fig.update_layout(
        height=450,
        yaxis_title="Sentiment medio",
        xaxis_title=""
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="#666666"
    )

    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Distribuzione del sentiment per dirigente")
    df_box = df_sent.copy()
    df_box["dirigente"] = df_box["executive"].map(lambda k: EXEC_BY_KEY[k].display_name)
    fig = px.box(df_box, x="dirigente", y="sentiment_score",
                 color="dirigente",
                 points=False)
    fig.update_layout(height=400, showlegend=False)
    fig.add_hline(y=0, line_dash="dot", line_color="#666666")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Eventi notevoli e shock di sentiment")
    EVENTS_DISPLAY = [
        ("2022-05-22", "Vittoria Scudetto"),
        ("2023-06-05", "Esonero Maldini & Massara"),
        ("2023-12-28", "Ritorno Ibra Operating Partner"),
        ("2024-10-01", "Intervista Cardinale"),
        ("2026-01-25", "NO Mercato"),
    ]
    df_events = df_sent.copy()
    created = pd.to_datetime(df_events["created_utc"], utc=True)
    season_start = created.dt.year - (created.dt.month < 7).astype(int)
    df_events["season"] = season_start.astype(str) + "/" + (season_start + 1).astype(str)
    season_labels = sorted(df_events["season"].dropna().unique())
    season_tabs = st.tabs([f"Stagione {season}" for season in season_labels])

    for season, season_tab in zip(season_labels, season_tabs):
        with season_tab:
            season_start_year = int(season.split("/")[0])
            season_start_date = pd.Timestamp(f"{season_start_year}-07-01", tz="UTC")
            season_end_date = pd.Timestamp(f"{season_start_year + 1}-06-30 23:59:59", tz="UTC")

            season_df = df_events[df_events["season"] == season].copy()
            # Aggregazione MENSILE (media del mese) invece che giornaliera.
            season_df["_month"] = (
                pd.to_datetime(season_df["created_utc"], utc=True)
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            df_all = (season_df.groupby("_month", as_index=False)
                               .agg(sentiment_score=("sentiment_score", "mean"),
                                    n=("sentiment_score", "count"))
                               .rename(columns={"_month": "date"})
                               .sort_values("date"))

            fig = px.area(df_all, x="date", y="sentiment_score",
                          color_discrete_sequence=[MILAN_RED],
                          markers=True,
                          hover_data={"n": True, "sentiment_score": ":+.3f"})
            fig.update_layout(height=320, margin=dict(t=20, b=20))
            fig.add_hline(y=0, line_dash="dot", line_color="#666666")

            for d, label in EVENTS_DISPLAY:
                event_date = pd.to_datetime(d, utc=True)
                if season_start_date <= event_date <= season_end_date:
                    fig.add_vline(x=event_date, line_dash="dash", line_color="#444444")
                    fig.add_annotation(
                        x=event_date,
                        y=0.7,
                        text=label,
                        showarrow=False,
                        textangle=-90,
                        font=dict(size=14, weight="bold"),
                    )

            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text="Sentiment medio")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📄 Sample di post – Stagione " + season):
                col_exec, col_label = st.columns([2, 1])
                with col_exec:
                    sample_exec = st.selectbox(
                        "Scegli dirigente",
                        options=sorted(season_df["executive"].unique()),
                        format_func=lambda k: EXEC_BY_KEY[k].display_name if k in EXEC_BY_KEY else k,
                        key=f"exec_{season}",
                    )
                with col_label:
                    sample_label = st.multiselect(
                        "Sentiment",
                        options=["positive", "neutral", "negative"],
                        default=["positive", "neutral", "negative"],
                        key=f"label_{season}",
                    )
                
                sample_posts = season_df[
                    (season_df["executive"] == sample_exec) &
                    (season_df["sentiment_label"].isin(sample_label))
                ].drop_duplicates(subset=["sentence"]).head(10)
                
                if len(sample_posts) == 0:
                    st.info(f"Nessun post trovato per {EXEC_BY_KEY[sample_exec].display_name} in questa stagione.")
                else:
                    for idx, row in sample_posts.iterrows():
                        color = {
                            "positive": "🟢",
                            "neutral": "⚪",
                            "negative": "🔴",
                        }.get(row["sentiment_label"], "◻")
                        lang_badge = f"[{row['lang'].upper()}]"
                        st.markdown(
                            f"{color} **{row['sentiment_label'].upper()}** {lang_badge}\n\n"
                            f"{row['sentence'][:200]}..." if len(row['sentence']) > 200 else f"{row['sentence']}\n\n"
                            f"*Score: {row['sentiment_score']:.3f} • Reddit score: {row['score']}*"
                        )


with tab3:
    st.subheader("Grafo di co-menzione")
    graph = load_graph()
    metrics = load_metrics()
    if not graph["nodes"]:
        st.warning("Esegui prima `python -m src.network` per generare il grafo.")
    else:
        G = nx.Graph()
        for n in graph["nodes"]:
            G.add_node(n["id"], **n)
        for e in graph["edges"]:
            G.add_edge(e["source"], e["target"], weight=e["weight"])
        pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5)

        # Edge trace
        edge_x, edge_y, widths, edge_text = [], [], [], []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
            widths.append(d["weight"])
            edge_text.append(f"{u} ↔ {v}: {d['weight']}")
        max_w = max(widths) if widths else 1
        edge_traces = []
        for (u, v, d) in G.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_traces.append(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(width=1 + 6 * d["weight"] / max_w, color="rgba(150,150,150,0.5)"),
                hoverinfo="text",
                text=f"{G.nodes[u]['label']} ↔ {G.nodes[v]['label']}: {d['weight']} co-menzioni",
                showlegend=False,
            ))

        # Node trace
        node_x, node_y, sizes, labels, hovertext, communities = [], [], [], [], [], []
        for n in G.nodes:
            x, y = pos[n]; node_x.append(x); node_y.append(y)
            attrs = G.nodes[n]
            sizes.append(20 + 2 * np.sqrt(attrs.get("mentions", 1)))
            labels.append(attrs.get("label", n))
            hovertext.append(
                f"<b>{attrs.get('label', n)}</b><br>"
                f"Ruolo: {attrs.get('role', '')}<br>"
                f"Menzioni: {attrs.get('mentions', 0)}<br>"
                f"Centralità degree: {attrs.get('degree', 0):.3f}<br>"
                f"Community: {attrs.get('community', -1)}"
            )
            communities.append(attrs.get("community", 0))
        palette = px.colors.qualitative.Bold
        node_colors = [palette[c % len(palette)] for c in communities]
        node_trace = go.Scatter(
            x=node_x, y=node_y, text=labels,
            mode="markers+text", textposition="top center",
            marker=dict(size=sizes, color=node_colors,
                        line=dict(width=2, color="white")),
            hovertext=hovertext, hoverinfo="text",
            showlegend=False,
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            height=550, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        if not metrics.empty:
            st.subheader("Metriche di centralità")
            show = metrics.copy()
            show["Dirigente"] = show["label"]
            show = show[["Dirigente", "role", "mentions",
                         "eigenvector", "clustering", "community"]].copy()
            for col in ("eigenvector", "clustering"):
                show[col] = show[col].round(3)
            st.dataframe(show, use_container_width=True, hide_index=True)


with tab4:
    st.subheader("Andamento congiunto: sentiment vs forma del Milan")
    st.caption("Rolling form = media punti delle ultime N partite di Serie A. "
               "Oscilla in [0, 3], permette di confrontare visivamente con il sentiment "
               "(altra serie stazionaria) senza il bias del trend monotono dei punti cumulativi.")

    c_a, c_b, c_c, c_d = st.columns([1, 1, 1, 2])
    with c_a:
        rolling_w = st.slider("Media mobile sentiment (settimane)",
                              1, 12, 4, 1, key="perf_rolling")
    with c_b:
        form_window = st.slider("Finestra rolling form (partite)",
                                3, 10, 5, 1, key="perf_form_window")
    with c_c:
        show_zscore = st.toggle("Mostra in z-score", value=False, key="perf_zscore",
                                help="Standardizza entrambi i segnali a media 0 e std 1 "
                                     "per confrontarli sullo stesso asse.")
    with c_d:
        season_options = sorted(df_match["season"].dropna().unique())
        sel_seasons = st.multiselect("Stagioni",
                                     options=season_options,
                                     default=season_options,
                                     key="perf_seasons")
    show_events = st.checkbox("Mostra eventi notevoli", value=True, key="perf_events")

    df_w = df_sent.copy()
    df_w["week"] = (pd.to_datetime(df_w["created_utc"]).dt.tz_convert("UTC")
                      .dt.to_period("W-MON").dt.start_time)
    weekly_sent = (df_w.groupby("week", as_index=False)
                       .agg(mean_score=("sentiment_score", "mean"),
                            n=("sentiment_score", "count"))
                       .sort_values("week"))
    weekly_sent["rolling"] = (weekly_sent["mean_score"]
                              .rolling(window=rolling_w, min_periods=1).mean())

    matches_sel = df_match[df_match["season"].isin(sel_seasons)].copy()
    matches_sel["date"] = pd.to_datetime(matches_sel["date"], utc=True)
    matches_sel = matches_sel.sort_values(["season", "date"])
    matches_sel["form_rolling"] = (
        matches_sel.groupby("season")["points"]
                   .transform(lambda s: s.rolling(form_window, min_periods=1).mean())
    )

    matches_sel["week"] = matches_sel["date"].dt.to_period("W-MON").dt.start_time
    weekly_form = (matches_sel.groupby("week", as_index=False)
                              .agg(form=("form_rolling", "last"),
                                   gf=("goals_for", "sum"),
                                   ga=("goals_against", "sum")))
    weekly_form["goal_diff"] = weekly_form["gf"] - weekly_form["ga"]

    def zscore(s: pd.Series) -> pd.Series:
        mu, sigma = s.mean(), s.std()
        return (s - mu) / sigma if sigma and sigma > 0 else s * 0

    if show_zscore:
        sent_y_main = zscore(weekly_sent["rolling"])
        sent_y_raw  = zscore(weekly_sent["mean_score"])
        matches_sel["form_y"] = matches_sel.groupby("season")["form_rolling"].transform(zscore)
        y1_title = "Sentiment (z-score)"
        y2_title = "Forma (z-score, per stagione)"
        y1_range = [-3, 3]
        y2_range = [-3, 3]
    else:
        sent_y_main = weekly_sent["rolling"]
        sent_y_raw  = weekly_sent["mean_score"]
        matches_sel["form_y"] = matches_sel["form_rolling"]
        y1_title = "Sentiment medio (-1 → +1)"
        y2_title = f"Forma (punti medi ultime {form_window} partite)"
        y1_range = [-1, 1]
        y2_range = [0, 3]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_sent["week"], y=sent_y_raw,
        name="Sentiment settimanale (grezzo)",
        yaxis="y1",
        line=dict(color=MILAN_RED, width=1),
        opacity=0.20,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=weekly_sent["week"], y=sent_y_main,
        name=f"Sentiment (media mobile {rolling_w}w)",
        yaxis="y1",
        line=dict(color=MILAN_RED, width=2.5, shape="spline"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Sentiment: %{y:+.3f}<extra></extra>",
    ))
    season_palette = ["#1F3A93", "#3F69C7", "#16A085", "#D35400", "#8E44AD"]
    for i, season in enumerate(sel_seasons):
        season_rows = matches_sel[matches_sel["season"] == season]
        if season_rows.empty: continue
        color = season_palette[i % len(season_palette)]
        fig.add_trace(go.Scatter(
            x=season_rows["date"], y=season_rows["form_y"],
            name=f"Forma {season}",
            yaxis="y2",
            line=dict(color=color, width=2, shape="spline"),
            hovertemplate=("<b>%{x|%d %b %Y}</b><br>vs %{customdata[0]} "
                           "(%{customdata[1]}-%{customdata[2]}, %{customdata[3]} pts)"
                           "<br>Forma: %{y:.2f}<extra></extra>"),
            customdata=season_rows[["opponent","goals_for","goals_against","points"]].values,
        ))
    if show_events:
        EVENTS_PERF = [
        ("2022-05-22", "Vittoria Scudetto"),
        ("2023-06-05", "Esonero Maldini & Massara"),
        ("2023-12-28", "Ritorno Ibra Operating Partner"),
        ("2024-10-01", "Intervista Cardinale"),
        ("2026-01-25", "NO Mercato"),
        ]
        for d, label in EVENTS_PERF:
            ev = pd.to_datetime(d, utc=True)
            fig.add_vline(x=ev, line_dash="dash", line_color="#999999")
            fig.add_annotation(x=ev, y=1, xref="x", yref="paper",
                               text=label, showarrow=False, textangle=-90,
                               font=dict(size=14, color="#444444", weight="bold"))

    yaxis_kwargs = dict(title=y1_title, side="left", zeroline=True,
                        zerolinecolor="#888", gridcolor="#F0F0F0")
    if y1_range: yaxis_kwargs["range"] = y1_range
    yaxis2_kwargs = dict(title=y2_title, overlaying="y", side="right",
                         gridcolor="#F0F0F0")
    if y2_range: yaxis2_kwargs["range"] = y2_range

    fig.update_layout(
        height=520, margin=dict(t=20, b=40, l=60, r=60),
        yaxis=yaxis_kwargs, yaxis2=yaxis2_kwargs,
        xaxis=dict(title="", showgrid=True, gridcolor="#F0F0F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        plot_bgcolor="white",
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#888888", yref="y")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlazione statistica")
    merged = weekly_sent.merge(weekly_form, on="week", how="inner")
    merged = merged.dropna(subset=["rolling", "form"])
    if len(merged) >= 8:
        from scipy.stats import pearsonr, spearmanr
        pr_r, pr_p = pearsonr(merged["rolling"], merged["form"])
        sp_r, sp_p = spearmanr(merged["rolling"], merged["form"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pearson r (sentiment vs forma)", f"{pr_r:+.3f}")
        c2.metric("Pearson p-value",                f"{pr_p:.4f}")
        c3.metric("Spearman ρ",                     f"{sp_r:+.3f}")
        c4.metric("Settimane confrontate",          f"{len(merged)}")
        st.caption("Correlazione tra sentiment rolling e forma rolling (ultime N partite). "
                   "Entrambi sono segnali stazionari, quindi la correlazione non è inflazionata "
                   "da trend monotoni. p < 0,05 = significatività statistica.")
    else:
        st.info("Settimane sovrapposte insufficienti per la correlazione.")

    st.subheader("Correlazione a diversi lag temporali")
    if len(merged) >= 12:
        from scipy.stats import pearsonr
        lags = list(range(-8, 9))
        rs, ps = [], []
        sent_series = merged["rolling"].reset_index(drop=True)
        form_series = merged["form"].reset_index(drop=True)
        for k in lags:
            if k >= 0:
                a = sent_series.iloc[:len(sent_series) - k] if k > 0 else sent_series
                b = form_series.iloc[k:].reset_index(drop=True)
            else:
                a = sent_series.iloc[-k:].reset_index(drop=True)
                b = form_series.iloc[:len(form_series) + k]
            if len(a) < 6:
                rs.append(np.nan); ps.append(np.nan); continue
            try:
                r, p = pearsonr(a.values, b.values)
                rs.append(r); ps.append(p)
            except Exception:
                rs.append(np.nan); ps.append(np.nan)

        lag_df = pd.DataFrame({"lag": lags, "r": rs, "p": ps})
        max_abs = max(0.3, np.nanmax(np.abs(rs)) * 1.1)
        fig_lag = go.Figure()
        fig_lag.add_trace(go.Bar(
            x=lag_df["lag"], y=lag_df["r"],
            marker=dict(
                color=lag_df["r"],
                colorscale=[[0, "#D62728"], [0.5, "#EEEEEE"], [1, "#1F77B4"]],
                cmin=-max_abs, cmax=max_abs,
                line=dict(color="#444", width=0.5),
            ),
            text=[f"{r:+.2f}" + ("*" if p < 0.05 else "") for r, p in zip(rs, ps)],
            textposition="outside",
            hovertemplate="<b>Lag %{x:+d} settimane</b><br>"
                          "Pearson r: %{y:+.3f}<br>p-value: %{customdata:.4f}<extra></extra>",
            customdata=lag_df["p"],
        ))
        fig_lag.update_layout(
            height=340, margin=dict(t=10, b=40, l=40, r=20),
            xaxis=dict(title="Lag (settimane): sentiment(t) vs forma(t+lag)",
                       tickmode="linear", dtick=1, gridcolor="#F0F0F0"),
            yaxis=dict(title="Pearson r", range=[-max_abs, max_abs],
                       gridcolor="#F0F0F0", zerolinecolor="#888"),
            plot_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig_lag, use_container_width=True)
    else:
        st.info("Settimane sovrapposte insufficienti per la heatmap di lag.")
