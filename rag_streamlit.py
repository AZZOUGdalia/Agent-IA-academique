# rag_streamlit.py

import streamlit as st
from rag_core import rag_query_with_history

st.set_page_config(page_title="RL LLM Agent", layout="wide")

st.title("Reinforcement Learning / Machine Learning LLM Assistant")

st.write("Ask questions in English about your RL / ML lecture notes (PDF-based RAG).")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Your question", "")

if st.button("Ask") and question.strip():
    answer = rag_query_with_history(question, st.session_state.history)
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

