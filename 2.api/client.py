import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={'input': {'topic': input_text}}
    )
    return response.json()['output']['content']

def get_codellama_response(input_text):
    response = requests.post(
        "http://localhost:8000/code/invoke",
        json={'input': {'topic': input_text}}
    )
    return response.json()['output']['content']


st.title('Langchain with Codellama API')
input_text1 = st.text_input('Write an essay on')
input_text2 = st.text_input('Write a code on')

if input_text1:
    st.write(get_openai_response(input_text1))

if input_text2:
    st.write(get_codellama_response(input_text2))