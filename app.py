"""Streamlit entry point for the AI Resume Screening Agent."""

import streamlit as st


st.set_page_config(page_title="AI Resume Screening Agent", page_icon="📄")

st.title("AI Resume Screening Agent")
st.write(
    "Rank candidates against a Job Description using NLP-based similarity "
    "and transparent scoring."
)
