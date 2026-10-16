"""
Builds the vector index for the Airbnb review RAG layer.

Reads reviews_sample_for_rag.csv (produced by prepare_data.py), embeds each
review's text with OpenAI's text-embedding-3-small, and saves the vectors
alongside their metadata to index.npz.

Cost note: ~4,000 short reviews is roughly 400-600K tokens, and
text-embedding-3-small costs $0.02 per 1M tokens — this whole index costs
a fraction of a cent to build.

Run:  python3 build_index.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_CSV = "reviews_sample_for_rag.csv"
OUTPUT_NPZ = "index.npz"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def embed_batch(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OPENAI_API_KEY found. Create a .env file (copy .env.example) with your key.")
        sys.exit(1)

    if not os.path.exists(INPUT_CSV):
        print(f"{INPUT_CSV} not found — run prepare_data.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Embedding {len(df):,} reviews in batches of {BATCH_SIZE}...")

    client = OpenAI(api_key=api_key)
    all_vectors = []
    texts = df["comments"].astype(str).tolist()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                vecs = embed_batch(client, batch)
                break
            except Exception as e:
                print(f"  batch {i // BATCH_SIZE} failed (attempt {attempt + 1}): {e}")
                time.sleep(2)
        else:
            raise RuntimeError(f"Batch {i // BATCH_SIZE} failed after 3 attempts")
        all_vectors.append(vecs)
        print(f"  embedded {min(i + BATCH_SIZE, len(texts)):,}/{len(texts):,}", end="\r")

    print()
    vectors = np.vstack(all_vectors)
    assert vectors.shape[0] == len(df)

    np.savez_compressed(
        OUTPUT_NPZ,
        vectors=vectors,
        listing_id=df["listing_id"].to_numpy(),
        name=df["name"].astype(str).to_numpy(),
        neighbourhood=df["neighbourhood"].astype(str).to_numpy(),
        price=df["price"].to_numpy(),
        room_type=df["room_type"].astype(str).to_numpy(),
        review_scores_rating=df["review_scores_rating"].to_numpy(),
        date=df["date"].astype(str).to_numpy(),
        comments=df["comments"].astype(str).to_numpy(),
    )
    print(f"Saved {OUTPUT_NPZ} — {vectors.shape[0]:,} vectors of dimension {vectors.shape[1]}")


if __name__ == "__main__":
    main()
