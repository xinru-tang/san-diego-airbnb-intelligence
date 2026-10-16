"""
RAG (Retrieval-Augmented Generation) agent for the San Diego Airbnb dataset.

Given a question like "which neighborhoods do guests complain about noise
in?" or "is La Jolla a good area for investment based on guest feedback?",
this:
  1. embeds the question,
  2. retrieves the most relevant guest reviews by cosine similarity
     (optionally filtered to a specific neighbourhood),
  3. pulls neighbourhood-level structured stats (price, supply, avg rating)
     for grounding numbers the review text alone can't give,
  4. asks the LLM to answer USING ONLY that retrieved evidence, with an
     explicit instruction to say so if the evidence is insufficient —
     this is the guardrail against the model just making things up
     (the RAG equivalent of the SQL-agent's execute-only-SELECT guardrail).

Run:  python3 rag_agent.py
"""
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INDEX_PATH = "index.npz"
LISTINGS_PATH = "listings_clean.csv"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 8


def load_index(path: str = INDEX_PATH) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_neighbourhood_stats(path: str = LISTINGS_PATH) -> pd.DataFrame:
    listings = pd.read_csv(path)
    stats = listings.groupby("neighbourhood").agg(
        num_listings=("listing_id", "count"),
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        avg_rating=("review_scores_rating", "mean"),
        superhost_share=("host_is_superhost", "mean"),
    ).round(2)
    return stats


def embed_query(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def cosine_top_k(query_vec: np.ndarray, vectors: np.ndarray, k: int,
                  mask: np.ndarray | None = None) -> np.ndarray:
    """Returns indices of the top-k most similar rows in `vectors`.
    `mask` (boolean array, same length as vectors) restricts the search
    to a subset, e.g. one neighbourhood, without needing a separate index."""
    v = vectors if mask is None else vectors[mask]
    if v.shape[0] == 0:
        return np.array([], dtype=int)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    sims = vn @ q
    top_local = np.argsort(-sims)[:k]
    if mask is None:
        return top_local
    # map local indices (within the masked subset) back to global indices
    global_idx = np.where(mask)[0]
    return global_idx[top_local]


def detect_neighbourhood_filter(question: str, known_neighbourhoods: list[str]) -> str | None:
    """Very simple substring match — good enough since neighbourhood names
    are distinctive (e.g. 'La Jolla', 'North Park'). Case-insensitive."""
    q_lower = question.lower()
    for n in sorted(known_neighbourhoods, key=len, reverse=True):
        if n.lower() in q_lower:
            return n
    return None


def build_prompt(question: str, reviews: list[dict], stats_row: dict | None,
                  neighbourhood: str | None) -> str:
    evidence = "\n".join(
        f'- [{r["neighbourhood"]}, ${r["price"]:.0f}/night, rating {r["review_scores_rating"]}] '
        f'"{r["comments"][:300]}"'
        for r in reviews
    )
    stats_block = ""
    if stats_row is not None:
        stats_block = (
            f"\nStructured stats for {neighbourhood}: {stats_row['num_listings']} listings, "
            f"avg price ${stats_row['avg_price']:.0f}/night, median price ${stats_row['median_price']:.0f}/night, "
            f"avg rating {stats_row['avg_rating']:.2f}, {stats_row['superhost_share']*100:.0f}% superhosts.\n"
        )
    return f"""You are helping a short-term rental investor evaluate San Diego neighborhoods.

Question: "{question}"
{stats_block}
Retrieved guest review excerpts (evidence — this is ALL the qualitative
evidence you have; do not use outside knowledge about these specific
listings or neighborhoods):
{evidence}

Answer the question using ONLY the evidence above (reviews + structured
stats). Quote or paraphrase specific reviews to support claims. If the
retrieved evidence doesn't actually address the question, say so explicitly
rather than guessing or generalizing beyond what's shown.
"""


def run_rag(client: OpenAI, index: dict, stats: pd.DataFrame, question: str,
            top_k: int = TOP_K) -> dict:
    neighbourhood = detect_neighbourhood_filter(question, list(stats.index))
    mask = None
    if neighbourhood is not None:
        mask = index["neighbourhood"] == neighbourhood

    query_vec = embed_query(client, question)
    idx = cosine_top_k(query_vec, index["vectors"], top_k, mask=mask)

    reviews = [
        {
            "neighbourhood": str(index["neighbourhood"][i]),
            "price": float(index["price"][i]),
            "review_scores_rating": index["review_scores_rating"][i],
            "comments": str(index["comments"][i]),
            "name": str(index["name"][i]),
            "date": str(index["date"][i]),
        }
        for i in idx
    ]

    stats_row = stats.loc[neighbourhood].to_dict() if neighbourhood in stats.index else None
    prompt = build_prompt(question, reviews, stats_row, neighbourhood)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = resp.choices[0].message.content.strip()

    return {
        "question": question,
        "neighbourhood_filter": neighbourhood,
        "stats": stats_row,
        "retrieved": reviews,
        "answer": answer,
    }


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OPENAI_API_KEY found. Create a .env file (copy .env.example) with your key.")
        sys.exit(1)
    if not os.path.exists(INDEX_PATH):
        print(f"{INDEX_PATH} not found — run build_index.py first.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    index = load_index()
    stats = load_neighbourhood_stats()

    print("San Diego Airbnb RAG Agent — ask a question in plain English.")
    print("Examples:")
    print("  - What do guests say about noise in Pacific Beach?")
    print("  - Is La Jolla a good area for investment based on guest feedback?")
    print("  - What are common complaints in Mission Bay?")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        result = run_rag(client, index, stats, question)
        if result["neighbourhood_filter"]:
            print(f"  [Filtered to neighbourhood: {result['neighbourhood_filter']}]")
        print(f"  [Retrieved {len(result['retrieved'])} reviews]")
        print(f"  [Answer]\n  {result['answer']}\n")


if __name__ == "__main__":
    main()
