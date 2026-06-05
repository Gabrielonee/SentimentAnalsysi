# Sentiment & Network Analysis — AC Milan (era RedBird)

Progetto di analisi del sentiment e network analysis sui dirigenti dell'AC
Milan nell'era di proprietà RedBird Capital (agosto 2022 – oggi).
Dati: r/ACMilan su Reddit (post + commenti, IT + EN), ottenuti tramite
**Arctic Shift** (mirror community della ex-Pushshift, accesso HTTP pubblico
senza credenziali). Output finale: dashboard interattiva con sentiment
temporale, grafo di co-menzione e correlazione con i risultati di Serie A.

> Autore: **Gabriele Soranno** — Università degli Studi di Milano-Bicocca

## Struttura del repository

```
SentimentAnalsysi/
├── Piano_Progetto_Sentiment_Milan_RedBird.docx   # piano dettagliato
├── README.md                                      # questo file
├── requirements.txt
├── data/                                          # parquet/csv generati dalla pipeline
├── output/                                        # grafo, metriche, correlazioni
└── src/
    ├── config.py        # dirigenti, alias, parametri globali
    ├── scraper.py       # PRAW – raccolta da r/ACMilan
    ├── preprocess.py    # pulizia + language detection
    ├── ner.py           # matching alias dirigenti
    ├── sentiment.py     # XLM-RoBERTa (con fallback lessicale)
    ├── network.py       # grafo co-menzione NetworkX
    ├── results.py       # correlazione con risultati Serie A
    ├── synth_data.py    # generatore di dati di demo
    └── dashboard.py     # app Streamlit
```

## Installazione

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download it_core_news_sm
python -m spacy download en_core_web_sm
```

**Niente credenziali Reddit richieste**: la raccolta dati passa per il
servizio pubblico Arctic Shift, accessibile via HTTP senza autenticazione.

## Quickstart con dati di demo

La pipeline è pronta all'uso con dati sintetici (utile per testare la
dashboard senza autenticazione Reddit):

```bash
python -m src.synth_data        # genera data/sentiment.parquet, data/matches.csv ...
python -m src.network           # produce output/graph.json, metrics.csv
python -m src.results           # correlazioni Pearson/Spearman + Granger
streamlit run src/dashboard.py  # avvia la dashboard su http://localhost:8501
```

## Pipeline completa su dati reali (Arctic Shift)

```bash
# Smoke test su una finestra ristretta
python -m src.scraper --start 2023-05-01 --end 2023-07-01 --max-records 2000

# Raccolta completa 2022-08 → oggi (può richiedere alcune ore per r/ACMilan)
python -m src.scraper

# Resto della pipeline
python -m src.preprocess              # pulizia + filtro IT/EN
python -m src.ner                     # individua menzioni dirigenti
python -m src.sentiment               # XLM-RoBERTa frase-per-frase
python -m src.network                 # grafo + metriche centralità
python -m src.results --matches data/matches.csv   # correlazione con Serie A
streamlit run src/dashboard.py
```

### Aspect-Based Sentiment Analysis (ABSA)

Modulo opzionale che risolve il bias di "target-sentiment misalignment": il
sentiment dell'intera frase viene assegnato a ciascun dirigente menzionato,
così frasi tipo *"Cardinale ha rovinato Maldini"* propagano negatività a
entrambi. L'ABSA ricalcola il sentiment **specifico verso ogni dirigente**.

```bash
# Lancio standard (traduce IT->EN, poi DeBERTa-v3 ABSA)
python -m src.aspect_sentiment

# Smoke test su 1000 coppie (frase, dirigente)
python -m src.aspect_sentiment --max-rows 1000

# Salta traduzione IT->EN (più veloce, peggiora performance su IT)
python -m src.aspect_sentiment --no-translate
```

Modelli usati:
- `yangheng/deberta-v3-base-absa-v1.1` per ABSA (440 MB)
- `Helsinki-NLP/opus-mt-it-en` per traduzione IT→EN (300 MB)

Output: `data/sentiment_absa.parquet` con schema identico a `sentiment.parquet`
ma una riga per ogni coppia (frase, dirigente). Puoi sostituire il file e
relanciare la dashboard senza altre modifiche, oppure tenerli affiancati per
confrontare i due approcci nella tesi.

### Fonte dati: Arctic Shift

[Arctic Shift](https://arctic-shift.photon-reddit.com/) è il successore di
Pushshift mantenuto dalla community accademica. Espone due endpoint REST
pubblici (`/api/posts/search` e `/api/comments/search`) che accettano i
parametri `subreddit`, `after`, `before`, `limit`, `sort`. Lo scraper li
chiama con paginazione forward su finestre mensili, con retry esponenziale e
throttling gentile. Non è richiesta alcuna chiave API.

Riferimento citabile in tesi:
*Heitmann, A. (2024). Arctic Shift: a community-maintained Reddit archive.
GitHub repository, https://github.com/ArthurHeitmann/arctic_shift*

Il file `data/matches.csv` va popolato manualmente o tramite scraping di
fbref/football-data; lo schema atteso è documentato nel docstring di
`src/results.py`.

## Dirigenti analizzati

| Chiave        | Dirigente           | Ruolo                                | Periodo (RedBird era) |
|---------------|---------------------|--------------------------------------|-----------------------|
| `cardinale`   | Gerry Cardinale     | Founder RedBird / Owner              | 08/2022 – oggi        |
| `furlani`     | Giorgio Furlani     | CEO                                  | 10/2022 – oggi        |
| `maldini`     | Paolo Maldini       | Direttore Tecnico                    | 08/2022 – 06/2023     |
| `massara`     | Frederic Massara    | Direttore Sportivo                   | 08/2022 – 06/2023     |
| `moncada`     | Geoffrey Moncada    | Direttore Tecnico / Head of Scouting | 06/2023 – oggi        |
| `ibrahimovic` | Zlatan Ibrahimović  | Operating Partner / Senior Advisor   | 12/2023 – oggi        |
| `gazidis`     | Ivan Gazidis        | CEO uscente                          | 08/2022 – 12/2022     |

## Dashboard – cosa mostra

1. **Overview** — KPI principali, distribuzione menzioni, mix linguistico.
2. **Sentiment** — serie temporale settimanale per dirigente, distribuzioni,
   eventi notevoli annotati (esoneri, ritorno Ibra, dichiarazioni Cardinale).
3. **Network** — grafo di co-menzione interattivo (Plotly), community
   color-coded, tabella metriche di centralità.
4. **Performance** — sentiment settimanale vs punti Serie A, correlazione
   Pearson/Spearman per dirigente.

## Limitazioni e considerazioni etiche

- Reddit non è rappresentativo dell'intera tifoseria milanista.
- Il sarcasmo e il gergo calcistico possono peggiorare la stima di sentiment.
- Correlazione ≠ causalità: il test di Granger fornisce un indizio di
  precedenza temporale, non di causalità reale.
- I nomi utente Reddit non vengono mai pubblicati in chiaro nei report.

## Licenza

MIT.
