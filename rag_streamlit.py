# rag_streamlit.py
import streamlit as st
from rag_core import rag_query_with_history

st.set_page_config(page_title="RL Academic Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Reinforcement Learning Assistant")
st.write("Pose tes questions sur le cours de RL / ML. Le modèle s'appuie sur tes PDF indexés (Chroma + embeddings locaux).")

# Initialiser l'historique dans la session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # liste de {"role": "user"/"assistant", "content": "..."}

# Afficher l'historique des messages
for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# Champ de saisie type ChatGPT
user_input = st.chat_input("Tape ta question ici...")

if user_input:
    # Ajouter la question de l'utilisateur à l'historique
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Afficher tout de suite la question dans l'UI
    with st.chat_message("user"):
        st.markdown(user_input)

    # Appeler le RAG avec l'historique (sans le system prompt, il est géré dans rag_core)
    with st.chat_message("assistant"):
        with st.spinner("Je réfléchis..."):
            answer = rag_query_with_history(
                question=user_input,
                history=st.session_state.chat_history[:-1]  # tout sauf la dernière question (qu'on vient d'ajouter)
            )
            st.markdown(answer)

    # Ajouter la réponse de l'assistant à l'historique
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
