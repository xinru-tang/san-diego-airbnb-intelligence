# San Diego Airbnb RAG Agent

Extends the ISBA Data Visualization project (Group 11) with a
retrieval-augmented generation (RAG) layer: instead of only a Tableau
dashboard with pre-built charts, you can now ask a question in plain
English and get an answer grounded in real guest review text plus
neighborhood-level stats.

## Pipeline

1. **`prepare_data.py`** — reproduces the original report's cleaning rules
   on the real Inside Airbnb `listings.csv` / `reviews.csv` (dropping
   reviews outside 2019-2025, under 15 characters, auto-generated
   cancellation messages, and non-English text via `py3langid`). Also
   builds a stratified sample (up to 40 reviews per neighborhood,
   favoring longer and more recent ones) for embedding — full-scale
   embedding of ~700K cleaned reviews would work but is unnecessary
   expense for a demo.

   Already run once; outputs included:
   - `listings_clean.csv` — 11,281 listings across 102 neighborhoods
   - `reviews_sample_for_rag.csv` — 3,937 reviews sampled for the index
   - (`reviews_clean.csv`, the full 695,780-row cleaned set, is not
     included here since it's 200+MB and not needed downstream — rerun
     `prepare_data.py` on the raw CSVs any time to regenerate it)

2. **`build_index.py`** — embeds every sampled review with OpenAI
   `text-embedding-3-small` and saves the vectors + metadata to
   `index.npz`. Costs a fraction of a cent for this dataset size.

3. **`rag_agent.py`** — the core agent:
   - detects if the question names a specific neighborhood (simple
     substring match against the known list) and restricts retrieval to
     it if so
   - embeds the question, retrieves the top-8 most similar reviews by
     cosine similarity
   - pulls structured neighborhood stats (listing count, avg/median
     price, avg rating, superhost share) for grounding numbers the
     review text alone can't give
   - asks the LLM to answer **using only that retrieved evidence**, with
     an explicit instruction to say so if the evidence doesn't actually
     address the question — this is the RAG equivalent of the apartment
     project's "only SELECT" guardrail: it's the thing standing between
     the model and just making plausible-sounding stuff up

4. **`app.py`** — a Streamlit chat UI on top of the same `run_rag()`
   function, showing the answer, the neighborhood stats used, and every
   retrieved review it was grounded in (so you can check the model isn't
   hallucinating).

## Verified without spending API credits

The retrieval math (`cosine_top_k`), the neighborhood-name detection, the
neighborhood-stats aggregation (against the real cleaned listings data),
and the full `run_rag()` control flow were all unit-tested with a stub
client and synthetic vectors before wiring in the real OpenAI API — same
approach as the apartment project. Confirmed: exact-vector retrieval
ranks first, neighborhood filtering never leaks results from other
neighborhoods, and the stats join pulls real numbers (e.g. La Jolla: 850
listings, $525/night avg, 4.84 avg rating).

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your real OpenAI key in it (same
key as the apartment project — this project reads its own local `.env`
in this folder, separate from that one).

## Run

```bash
python build_index.py     # one-time: builds index.npz (needs your API key)
python rag_agent.py       # command-line version
streamlit run app.py      # web version
```

## Evaluation (`eval_airbnb.py`)

```bash
python eval_airbnb.py
```

Checks three things a demo run doesn't tell you: (1) neighbourhood-name
detection accuracy against 6 real questions, checked with zero API cost
since it's pure string matching; (2) retrieval isolation — when a question
is filtered to one neighbourhood, do all 8 retrieved reviews actually
belong to it, and do the printed stats match an independent recomputation
from `listings_clean.csv`; (3) honest refusal — asks a question this
dataset genuinely can't answer ("best month to visit Antarctica") and
checks the response hedges instead of inventing a confident answer.
Requires `index.npz` to exist already. Costs about 6 embedding + 6 chat
calls — a fraction of a cent.

## Honest limitations (good interview material)

- The neighborhood-name detector is a substring match, not true NLU — a
  question phrased without naming a neighborhood searches everything,
  which can dilute relevance for genuinely comparative questions
  ("which neighborhood is best...").
- The review sample is capped at 40 per neighborhood for cost/speed, so
  less-reviewed neighborhoods are represented by their full review set
  while heavily-reviewed ones (Mission Bay, La Jolla) only show a slice —
  worth being upfront about if asked "is this representative?"
- Grounding only prevents *unsupported* claims: if the retrieved sample
  happens to be biased (e.g. mostly negative reviews from one bad host),
  the answer will faithfully reflect that biased sample, not the
  neighborhood's true overall sentiment.
