"""
Streamlit UI for the San Diego Airbnb RAG Agent.

Run:  streamlit run app.py

Reuses every function from rag_agent.py — this file is purely presentation.
"""
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag_agent import INDEX_PATH, load_index, load_neighbourhood_stats, run_rag

load_dotenv()

st.set_page_config(page_title="San Diego Airbnb RAG Agent", page_icon="🏖️", layout="centered")

st.title("🏖️ San Diego Airbnb Investment Advisor")
st.caption(
    "Ask a question about a San Diego neighborhood. The agent retrieves the "
    "most relevant guest reviews (semantic search over real Inside Airbnb "
    "review text) plus neighborhood-level stats, and answers grounded only "
    "in that evidence."
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("No OPENAI_API_KEY found. Create a `.env` file in this folder with your key, then restart.")
    st.stop()

if not os.path.exists(INDEX_PATH):
    st.error(f"`{INDEX_PATH}` not found. Run `python prepare_data.py` then `python build_index.py` first.")
    st.stop()


@st.cache_resource
def get_resources():
    client = OpenAI(api_key=api_key)
    index = load_index()
    stats = load_neighbourhood_stats()
    return client, index, stats


client, index, stats = get_resources()

with st.expander("Example questions"):
    st.markdown(
        "- What do guests say about noise in Pacific Beach?\n"
        "- Is La Jolla a good area for investment based on guest feedback?\n"
        "- What are common complaints in Mission Bay?\n"
        "- Which neighborhood has the best value for money according to reviews?"
    )

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask about a San Diego neighborhood...")

if question:
    with st.spinner("Retrieving reviews and thinking..."):
        result = run_rag(client, index, stats, question)
    st.session_state.history.append((question, result))

for question, result in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        if result["neighbourhood_filter"]:
            st.info(f"Filtered to neighbourhood: **{result['neighbourhood_filter']}**")
            s = result["stats"]
            if s:
                st.markdown(
                    f"**{int(s['num_listings'])}** listings · "
                    f"avg **${s['avg_price']:.0f}**/night · "
                    f"avg rating **{s['avg_rating']:.2f}** · "
                    f"**{s['superhost_share']*100:.0f}%** superhosts"
                )
        else:
            st.caption("No specific neighbourhood detected in the question — searching across all reviews.")

        st.markdown("**Answer:**")
        # Escape "$" before rendering: Streamlit's markdown renderer treats a
        # pair of "$" as LaTeX math delimiters, which mangled answers that
        # mentioned two dollar amounts (e.g. "$525...median price of $336"
        # rendered as squished italic text instead of plain numbers).
        st.write(result["answer"].replace("$", "\\$"))

        with st.expander(f"Retrieved evidence ({len(result['retrieved'])} reviews)"):
            for r in result["retrieved"]:
                st.markdown(f"— *{r['neighbourhood']}, \\${r['price']:.0f}/night, rating {r['review_scores_rating']}*")
                st.write(r["comments"].replace("$", "\\$"))
                st.divider()
