import streamlit as st 
from main import get_qa_chain, create_vectordb

st.title("Codebascis FAQ")

btn = st.button("Create knowledgebase")
if btn: 
    create_vectordb()

question = st.text_input("Question: ")

if question: 
    chain = get_qa_chain()
    response = chain.invoke(question)

    st.header("Answer")
    st.write(response)