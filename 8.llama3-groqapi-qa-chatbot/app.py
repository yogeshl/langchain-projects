import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Chatgroq with llama3 demo")

llm= ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt= ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Pleae provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}

"""
)

def process_documents():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us-census") # Data Ingestion
        st.session_state.docs = st.session_state.loader.load(); 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Chunking
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)



prompt1 = st.text_input("Enter question from documents")


if st.button("Process Documents"):
    process_documents()
    st.write("DOcuments processed and loaded to vector db")



if prompt1:

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start=time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # using streamlit expander
    with st.expander("Document similarity search"):
        #Find the relevant docs chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")