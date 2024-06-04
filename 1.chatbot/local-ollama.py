//from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

## Third party integrations.
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

//os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user questions."),
        ("user", "Question: {question}")
    ]
)

## streamlit framework

st.title("Langchain demo using Ollama - Codellama CodeExpert custom model")
input_text = st.text_input("Enter the question")

## ollama LLama2 llm

llm = Ollama(model="CodeExpert")

#output parser
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))