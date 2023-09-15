import sys
import streamlit as st
import os
from utils import *
from config import *


st.title("Ask about UCI's ongoing clinical trials")


if "agent" not in st.session_state:
    # Get OpenAI key
    openai_api_key = os.environ['OPENAI_API_KEY']

    # Load documents
    docs = load_document(os.path.join(DATA_DIR, TRAINING_FILE))

    # Load vector database (train if it doesn't exist)
    vectordb = load_embeddings(EMBEDDINGS_DIR, openai_api_key, docs)

    # Create chain
    agent = create_agent(model='gpt-3.5-turbo',
                         temperature=0, vectordb=vectordb)

    st.session_state["agent"] = agent

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything to help you find the right trial!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = st.session_state.agent(
            {"input": prompt}, callbacks=[stream_handler])['output']

    st.session_state.messages.append(
        {"role": "assistant", "content": response})
