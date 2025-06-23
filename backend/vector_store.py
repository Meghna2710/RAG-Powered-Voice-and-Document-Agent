from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def load_or_create_vectorstore(text_chunks, persist_path="db/chroma_store"):
    if not text_chunks:
        raise ValueError("No text chunks provided to embed.")

    full_text = "\n".join(text_chunks)

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([full_text])
    docs = [d for d in docs if len(d.page_content.strip()) > 20]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_path)
    vectordb.persist()
    return vectordb