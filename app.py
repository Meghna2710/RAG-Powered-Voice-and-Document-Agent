import os
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from backend.file_ingestor import process_uploaded_files
from backend.vector_store import load_or_create_vectorstore
from backend.rag_chain import get_chat_chain
from backend.memory_manager import get_memory
from backend.audio_transcriber import transcribe_audio

os.environ["STREAMLIT_SERVER_HEADLESS"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="RAG-Powered Voice & Document Agent", layout="wide")
st.title("🎧 RAG-Powered Voice & Document Agent")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: upload files
uploaded_files = st.sidebar.file_uploader(
    "Upload additional documents", type=["pdf", "txt", "docx"], accept_multiple_files=True
)

uploaded_audio = st.sidebar.file_uploader(
    "Upload an audio file", type=["mp3", "wav", "m4a"], accept_multiple_files=False
)

text_chunks = []

if uploaded_files:
    with st.spinner("Processing uploaded documents..."):
        for file in uploaded_files:
            text_chunks += process_uploaded_files(file)
        st.success("Documents processed!")

if uploaded_audio:
    with st.spinner("Transcribing audio..."):
        audio_text = transcribe_audio(uploaded_audio)
        st.success("Transcription complete!")
        st.write(f"📝 *Transcribed Text:* {audio_text}")
        text_chunks.append(audio_text)

if text_chunks:
    with st.spinner("Creating or loading vectorstore..."):
        vectordb = load_or_create_vectorstore(text_chunks)

    memory = get_memory()  
    qa_chain = get_chat_chain(vectordb, memory)

    st.success("System ready for Q&A.")

    user_question = st.chat_input("Ask a question about the uploaded content...")
    if user_question:
        st.session_state.chat_history.append(HumanMessage(content=user_question))

        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": user_question})
            answer = result["answer"] if isinstance(result, dict) else result

        st.session_state.chat_history.append(AIMessage(content=answer))

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)