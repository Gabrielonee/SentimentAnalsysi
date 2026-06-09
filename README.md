# Sentiment & Network Analysis — AC Milan (RedBird era)

Sentiment and network analysis of AC Milan's executives during the RedBird
Capital ownership era (August 2022 – present). Data comes from the r/ACMilan
subreddit (posts + comments, Italian and English), collected through
**Arctic Shift** (a community-maintained mirror of the former Pushshift, with
public HTTP access and no credentials required). The final output is an
interactive dashboard with sentiment over time, an executive co-mention graph,
and correlation with Serie A results.

> Author: **Gabriele Soranno** — University of Milano-Bicocca

## Repository structure

```
SentimentAnalsysi/
├── README.md
├── requirements.txt
├── data/
├── output/
└── src/
    ├── config.py
    ├── scraper.py
    ├── preprocess.py
    ├── ner.py
    ├── sentiment.py
    ├── aspect_sentiment.py
    ├── network.py
    ├── results.py
    ├── compare_backends.py
    ├── explain_sentiment.py
    ├── build_comparative_set.py
    ├── finetune_absa.py
    ├── eval_finetuned.py
    └── dashboard.py
```

The `data/` directory and the model weights are not versioned (they are large
and reproducible). Datasets are listed in `.gitignore`; only `data/matches.csv`
is tracked. The `output/` directory keeps the explainability reports.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download it_core_news_sm
python -m spacy download en_core_web_sm
```

No Reddit credentials are required: data collection goes through the public
Arctic Shift service over HTTP.

## Full pipeline (Arctic Shift)

```bash
python -m src.scraper --start 2023-05-01 --end 2023-07-01 --max-records 2000
python -m src.scraper
python -m src.preprocess
python -m src.ner
python -m src.sentiment
python -m src.network
python -m src.results --matches data/matches.csv
streamlit run src/dashboard.py
```

Running the pipeline regenerates the datasets under `data/` and the analysis
artifacts under `output/`.

## Aspect-Based Sentiment Analysis (ABSA)

ABSA resolves the "target-sentiment misalignment" bias: a whole-sentence
sentiment model assigns the same score to every executive mentioned, so a
sentence like *"Cardinale ruined Maldini"* propagates negativity to both. ABSA
recomputes the sentiment specific to each executive.

```bash
python -m src.aspect_sentiment
python -m src.aspect_sentiment --max-rows 1000
python -m src.aspect_sentiment --no-translate
```

Models used:

- `yangheng/deberta-v3-base-absa-v1.1` for ABSA (440 MB)
- `Helsinki-NLP/opus-mt-it-en` for IT→EN translation (300 MB)

Output: `data/sentiment_absa.parquet`, with the same schema as
`sentiment.parquet` but one row per (sentence, executive) pair. The file can
replace `sentiment.parquet` in the dashboard, or be kept alongside it to
compare the two approaches.

## Explainability

Token-level attribution (Integrated Gradients, implemented in pure PyTorch)
comparing the vanilla sentence-level model with the target-aware ABSA model.

```bash
python -m src.explain_sentiment
python -m src.explain_sentiment --text "Zlatan is not Maldini" --aspect Maldini
python -m src.explain_sentiment --input data/showcase_examples.csv
```

Output: an HTML report under `output/` with tokens colored by their
contribution (blue toward positive, red toward negative).

## Fine-tuning on comparative cases

The ABSA model struggles on comparative, negation, and nostalgic sentences. The
following scripts build an annotation set, fine-tune the model, and evaluate the
effect against the base model.

```bash
python -m src.build_comparative_set
python -m src.finetune_absa --epochs 3 --anchors 200 --lr 1e-5
python -m src.eval_finetuned --max-per-aspect 800
```

`build_comparative_set` extracts comparative candidates into
`data/comparative_to_label.csv` for manual annotation; `finetune_absa` fine-tunes
`yangheng/deberta-v3-base-absa-v1.1` locally (Mac MPS / CPU) with a before/after
evaluation; `eval_finetuned` runs a targeted comparison of the base and
fine-tuned models, including the executive-level bias analysis.

## Data source: Arctic Shift

[Arctic Shift](https://arctic-shift.photon-reddit.com/) is a
community-maintained successor to Pushshift. It exposes two public REST
endpoints (`/api/posts/search` and `/api/comments/search`) that accept the
`subreddit`, `after`, `before`, `limit`, and `sort` parameters. The scraper
calls them with forward pagination over monthly windows, with exponential retry
and gentle throttling. No API key is needed.

Citable reference:
*Heitmann, A. (2024). Arctic Shift: a community-maintained Reddit archive.
GitHub repository, https://github.com/ArthurHeitmann/arctic_shift*

The `data/matches.csv` file must be populated manually or by scraping
fbref/football-data; the expected schema is documented in the docstring of
`src/results.py`.

## Executives analyzed

| Key           | Executive           | Role                                 | Period (RedBird era) |
|---------------|---------------------|--------------------------------------|----------------------|
| `cardinale`   | Gerry Cardinale     | Founder RedBird / Owner              | 08/2022 – present    |
| `furlani`     | Giorgio Furlani     | CEO                                  | 10/2022 – present    |
| `maldini`     | Paolo Maldini       | Technical Director                   | 08/2022 – 06/2023    |
| `massara`     | Frederic Massara    | Sporting Director                    | 08/2022 – 06/2023    |
| `moncada`     | Geoffrey Moncada    | Technical Director / Head of Scouting| 06/2023 – present    |
| `ibrahimovic` | Zlatan Ibrahimović  | Operating Partner / Senior Advisor   | 12/2023 – present    |
| `gazidis`     | Ivan Gazidis        | Outgoing CEO                         | 08/2022 – 12/2022    |

## Dashboard — what it shows

1. **Overview** — key KPIs, mention distribution, language mix.
2. **Sentiment** — weekly sentiment time series per executive, distributions,
   annotated notable events (dismissals, Ibra's return, Cardinale's statements).
3. **Network** — interactive co-mention graph (Plotly), color-coded
   communities, centrality metrics table.
4. **Performance** — weekly sentiment vs Serie A form, Pearson/Spearman
   correlation and cross-correlation by temporal lag.

## Limitations and ethical considerations

- Reddit is not representative of the entire Milan fan base; the community is
  English-speaking and self-selected.
- Sarcasm and football jargon can degrade sentiment estimates.
- The ABSA model shows a systematic bias on comparative and nostalgic mentions.
- Correlation is not causation: the temporal analysis suggests precedence, not
  real causal effect.
- Reddit usernames are never published in clear text in the reports.

## License

MIT.
