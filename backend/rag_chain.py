from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",  
        max_length=512,
        temperature=0,
        do_sample=False,
        device=-1  
    )
    return HuggingFacePipeline(pipeline=pipe)

def get_chat_chain(vectordb, memory=None):
    llm = load_llm()
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})


    prompt_template = """You are a helpful assistant. Use the provided context to answer the question concisely and using your own words.
    Don't copy exactly from the context.

If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template.strip()
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  
        combine_docs_chain_kwargs={"prompt": prompt,
        "output_key": "answer"}  
    )

    return chain