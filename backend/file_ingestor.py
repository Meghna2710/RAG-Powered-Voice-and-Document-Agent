import os
import tempfile
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def process_uploaded_files(file):
    suffix = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()
    os.remove(tmp_path)

    return [doc.page_content for doc in docs]  