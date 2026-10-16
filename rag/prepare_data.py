"""
Data preparation for the San Diego Airbnb RAG project.

Reproduces the cleaning steps from the original ISBA Data Visualization
project report (listings + reviews from Inside Airbnb), then builds a
down-sampled set of review documents suitable for embedding into a vector
store — full-scale embedding of ~800K+ reviews is unnecessary (and
needlessly expensive) for a demo RAG layer whose job is to answer
qualitative questions like "what do guests say about noise in North Park?"

Inputs (place in this folder, or pass paths as argv):
  listings.csv, reviews.csv  (raw Inside Airbnb exports)

Outputs:
  listings_clean.csv          — cleaned listing-level data (price, neighbourhood, etc.)
  reviews_clean.csv           — all reviews surviving the report's cleaning rules
  reviews_sample_for_rag.csv  — a stratified sample (by neighbourhood) for embedding
"""
import sys

import pandas as pd
import py3langid as langid

LISTINGS_IN = sys.argv[1] if len(sys.argv) > 1 else "listings.csv"
REVIEWS_IN = sys.argv[2] if len(sys.argv) > 2 else "reviews.csv"

MAX_REVIEWS_PER_NEIGHBOURHOOD = 40
MIN_COMMENT_LEN = 15
DATE_START, DATE_END = "2019-01-01", "2025-12-31"

CANCELLATION_PATTERNS = [
    "the host canceled this reservation",
    "the reservation was canceled",
    "this is an automated posting",
    "automated post",
]


def clean_listings() -> pd.DataFrame:
    cols = ["id", "name", "neighbourhood_cleansed", "latitude", "longitude",
            "room_type", "price", "number_of_reviews", "review_scores_rating",
            "host_is_superhost", "minimum_nights"]
    df = pd.read_csv(LISTINGS_IN, usecols=cols, low_memory=False)

    df["price"] = (
        df["price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    )
    df = df[df["price"].notna() & (df["price"] > 0) & (df["price"] <= 10000)]

    df["host_is_superhost"] = df["host_is_superhost"].map({"t": True, "f": False})
    df = df.rename(columns={"id": "listing_id", "neighbourhood_cleansed": "neighbourhood"})
    df = df.dropna(subset=["neighbourhood"])
    return df.reset_index(drop=True)


def is_probably_english(text: str) -> bool:
    try:
        lang, score = langid.classify(text)
        return lang == "en"
    except Exception:
        return False


def clean_reviews(listing_ids: set) -> pd.DataFrame:
    df = pd.read_csv(REVIEWS_IN, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[df["listing_id"].isin(listing_ids)]
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)]

    df["comments"] = df["comments"].astype(str).str.strip()
    df = df[df["comments"].str.len() >= MIN_COMMENT_LEN]

    lower = df["comments"].str.lower()
    auto_mask = pd.Series(False, index=df.index)
    for pat in CANCELLATION_PATTERNS:
        auto_mask |= lower.str.contains(pat, regex=False)
    df = df[~auto_mask]

    print(f"  Running language detection on {len(df):,} candidate reviews "
          f"(this is the slow step)...")
    df = df[df["comments"].apply(is_probably_english)]

    return df.reset_index(drop=True)


def stratified_sample_for_rag(reviews: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    merged = reviews.merge(
        listings[["listing_id", "name", "neighbourhood", "price", "room_type",
                  "review_scores_rating", "host_is_superhost"]],
        on="listing_id", how="inner",
    )
    # favor longer (more informative) and more recent reviews within each neighbourhood
    merged["comment_len"] = merged["comments"].str.len()
    merged = merged.sort_values(["neighbourhood", "date", "comment_len"],
                                 ascending=[True, False, False])
    sampled = merged.groupby("neighbourhood", group_keys=False).head(MAX_REVIEWS_PER_NEIGHBOURHOOD)
    return sampled.reset_index(drop=True)


def main():
    print("Cleaning listings...")
    listings = clean_listings()
    listings.to_csv("listings_clean.csv", index=False)
    print(f"  {len(listings):,} listings kept, across {listings['neighbourhood'].nunique()} neighbourhoods")

    print("Cleaning reviews (this can take a few minutes on ~900K rows)...")
    reviews = clean_reviews(set(listings["listing_id"]))
    reviews.to_csv("reviews_clean.csv", index=False)
    print(f"  {len(reviews):,} reviews kept after all cleaning steps")

    print("Building stratified sample for embedding...")
    sample = stratified_sample_for_rag(reviews, listings)
    sample.to_csv("reviews_sample_for_rag.csv", index=False)
    print(f"  {len(sample):,} reviews sampled across "
          f"{sample['neighbourhood'].nunique()} neighbourhoods "
          f"(up to {MAX_REVIEWS_PER_NEIGHBOURHOOD} per neighbourhood)")


if __name__ == "__main__":
    main()
