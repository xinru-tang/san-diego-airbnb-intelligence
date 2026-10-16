"""
Small evaluation harness for the Airbnb RAG agent (rag_agent.py).

Three things worth checking beyond "does it give a plausible-sounding
answer":

  1. Neighbourhood-name detection — does asking about "Pacific Beach"
     actually filter to Pacific Beach, and does a question naming no
     neighbourhood correctly search everything? This part is free (no API
     call, pure string matching) so it's checked against every known
     neighbourhood, not just a couple of examples.
  2. Retrieval isolation — when a question is filtered to one
     neighbourhood, do ALL 8 retrieved reviews actually belong to that
     neighbourhood (no leakage from elsewhere)? And do the printed
     stats (listing count, avg price, rating) match a fresh, independent
     computation from listings_clean.csv?
  3. Groundedness / honest refusal — the whole point of the "answer using
     ONLY the retrieved evidence, say so if it's insufficient" instruction
     is to stop the model from inventing an answer when the evidence
     doesn't support one. This is tested with a deliberately unanswerable
     question and a simple keyword check for hedging language — imperfect,
     but the full answer is also printed so you can judge it yourself.

Run:  python3 eval_airbnb.py

Costs ~6 embedding calls + ~6 gpt-4o-mini calls — a fraction of a cent.
Requires index.npz to already exist (run build_index.py first if not).
"""
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from rag_agent import (
    INDEX_PATH,
    LISTINGS_PATH,
    detect_neighbourhood_filter,
    load_index,
    load_neighbourhood_stats,
    run_rag,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Part 1: neighbourhood-detection accuracy (no API calls — pure string match)
# ---------------------------------------------------------------------------

DETECTION_CASES = [
    ("What do guests say about noise in Pacific Beach?", "Pacific Beach"),
    ("Is La Jolla a good area for investment?", "La Jolla"),
    ("What are common complaints in Mission Bay?", "Mission Bay"),
    ("Tell me about Balboa Park listings.", "Balboa Park"),
    ("Which neighborhood has the best value for money?", None),
    ("What do reviews say about cleanliness overall?", None),
]


def run_detection_checks(known_neighbourhoods: list[str]) -> int:
    print("=" * 60)
    print("Part 1: neighbourhood-name detection (no API cost)")
    print("=" * 60)
    n_pass = 0
    for question, expected in DETECTION_CASES:
        got = detect_neighbourhood_filter(question, known_neighbourhoods)
        ok = got == expected
        n_pass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] \"{question}\" -> expected {expected!r}, got {got!r}")
    print(f"\nDetection accuracy: {n_pass}/{len(DETECTION_CASES)}\n")
    return n_pass


# ---------------------------------------------------------------------------
# Part 2 + 3: retrieval isolation, stats grounding, and a refusal test
#             (these call the real API)
# ---------------------------------------------------------------------------

RETRIEVAL_QUESTIONS = [
    "What do guests say about noise in Pacific Beach?",
    "Is La Jolla a good area for investment based on guest feedback?",
    "What are common complaints in Mission Bay?",
]

# Deliberately unanswerable from this dataset — a good agent should say the
# evidence doesn't address it rather than inventing a confident answer.
REFUSAL_QUESTION = "What is the best month to visit Antarctica?"
HEDGE_PHRASES = ["don't", "does not", "doesn't", "insufficient", "no evidence",
                  "cannot determine", "can't determine", "not addressed",
                  "not mention", "doesn't address", "unable to", "no information"]


def run_retrieval_checks(client, index, stats):
    print("=" * 60)
    print("Part 2: retrieval isolation + stats grounding (real API calls)")
    print("=" * 60)
    n_pass = 0
    for question in RETRIEVAL_QUESTIONS:
        result = run_rag(client, index, stats, question)
        nb = result["neighbourhood_filter"]
        leaked = [r for r in result["retrieved"] if r["neighbourhood"] != nb]
        isolation_ok = nb is not None and len(leaked) == 0

        # Independently recompute this neighbourhood's stats and compare.
        fresh_stats = load_neighbourhood_stats(LISTINGS_PATH)
        stats_ok = nb in fresh_stats.index and result["stats"] == fresh_stats.loc[nb].to_dict()

        ok = isolation_ok and stats_ok
        n_pass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] \"{question}\"")
        print(f"      filtered to: {nb!r} | leaked reviews from other neighbourhoods: {len(leaked)}")
        print(f"      stats match independent recomputation: {stats_ok}")
        print(f"      answer: {result['answer'][:200]}")
        print()
        time.sleep(0.3)
    print(f"Retrieval-isolation checks: {n_pass}/{len(RETRIEVAL_QUESTIONS)}\n")
    return n_pass


def run_refusal_check(client, index, stats):
    print("=" * 60)
    print("Part 3: honest refusal on an unanswerable question")
    print("=" * 60)
    result = run_rag(client, index, stats, REFUSAL_QUESTION)
    answer_lower = result["answer"].lower()
    hedged = any(p in answer_lower for p in HEDGE_PHRASES)
    print(f"  Question: \"{REFUSAL_QUESTION}\"")
    print(f"  Answer:   {result['answer']}")
    print(f"  [{'PASS' if hedged else 'FAIL / REVIEW MANUALLY'}] "
          f"{'contains hedging language as expected' if hedged else 'no hedging phrase detected — read the answer above and judge whether it still avoided inventing a confident, unsupported claim'}")
    print()
    return hedged


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OPENAI_API_KEY found. Create a .env file in this folder with your key.")
        sys.exit(1)
    if not os.path.exists(INDEX_PATH):
        print(f"{INDEX_PATH} not found — run build_index.py first.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    index = load_index()
    stats = load_neighbourhood_stats()

    n1 = run_detection_checks(list(stats.index))
    n2 = run_retrieval_checks(client, index, stats)
    hedged = run_refusal_check(client, index, stats)

    print("=" * 60)
    print("SUMMARY")
    print(f"  Neighbourhood detection: {n1}/{len(DETECTION_CASES)}")
    print(f"  Retrieval isolation + stats grounding: {n2}/{len(RETRIEVAL_QUESTIONS)}")
    print(f"  Honest refusal on unanswerable question: {'PASS' if hedged else 'REVIEW MANUALLY'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
