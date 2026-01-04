# rag_streamlit.py - Full UI with Status Indicator & Interactive Features
import streamlit as st
import os
import requests
from rag_core import rag_query_with_history, build_vector_index, load_pdfs, generate_quiz_question

st.set_page_config(page_title="Uni AI Assistant", page_icon="🎓", layout="wide")

# === Sidebar: System Status & Settings ===
with st.sidebar:
    st.title("⚙️ System Status")
    
    # 1. Connection Check (Practical Feature)
    try:
        response = requests.get("http://localhost:11434")
        if response.status_code == 200:
            st.success("🟢 AI Engine Online")
        else:
            st.warning("🟡 AI Engine Unresponsive")
    except:
        st.error("🔴 AI Engine Offline")
        st.info("Run 'ollama serve' in your terminal.")

    st.divider()
    
    # 2. File Uploader
    st.subheader("📂 Upload Lecture Notes")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process & Index Files", type="primary"):
            with st.spinner("Reading and Indexing PDFs..."):
                save_dir = "data_raw"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                for file in uploaded_files:
                    with open(os.path.join(save_dir, file.name), "wb") as f:
                        f.write(file.getbuffer())
                
                # Rebuild Index
                docs = load_pdfs(save_dir)
                build_vector_index(docs)
                st.success(f"Successfully processed {len(docs)} documents!")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# === Main Interface ===
st.title("🎓 University AI Companion")

# Tabs for different modes
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📝 Interactive Quiz"])

# --- TAB 1: CHAT ---
with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Sources"):
                    st.write(", ".join(msg["sources"]))

    # Input Area
    user_input = st.chat_input("Ask about exams, deadlines, or concepts...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Clean history (remove metadata like 'sources' before sending to LLM)
                clean_history = [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.chat_history
                    if m["role"] in ["user", "assistant"]
                ]
                
                # Get Answer AND Sources
                answer, sources = rag_query_with_history(user_input, clean_history)
                
                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources"):
                        st.write(", ".join(sources))

        # Save to history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": answer, 
            "sources": sources
        })

# --- TAB 2: INTERACTIVE QUIZ ---
with tab2:
    st.header("🧠 Knowledge Check")
    st.write("Test your knowledge based on the uploaded documents.")
    
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None

    if st.button("🎲 Generate New Question", type="primary"):
        with st.spinner("Generating question..."):
            st.session_state.quiz_data = generate_quiz_question()

    if st.session_state.quiz_data:
        q = st.session_state.quiz_data
        
        if "error" in q:
            st.error(q["error"])
        elif q.get("error_parsing"):
            st.warning("Raw output (could not parse options):")
            st.write(q["question"])
        else:
            st.subheader(f"Q: {q['question']}")
            
            # Interactive Radio Button
            user_choice = st.radio(
                "Select your answer:", 
                q["options"], 
                index=None,
                key=f"quiz_{q['question'][:10]}"
            )
            
            if st.button("Submit Answer"):
                if user_choice:
                    selected_letter = user_choice.split(":")[0].replace("Option", "").strip()
                    correct_letter = q['correct_answer']
                    
                    if selected_letter == correct_letter:
                        st.success(f"✅ Correct! The answer is {correct_letter}.")
                        st.balloons()
                    else:
                        st.error(f"❌ Incorrect. The correct answer was {correct_letter}.")
                else:
                    st.warning("Please select an option first.")