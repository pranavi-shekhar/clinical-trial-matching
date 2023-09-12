import openai
import streamlit as st
import os
from utils import *
from config import *


st.title("Ask about UCI's ongoing clinical trials")


# Get OpenAI key
openai_api_key = os.environ['OPENAI_API_KEY']

# Load documents
docs = load_document(os.path.join(DATA_DIR, TRAINING_FILE))

# Load vector database (train if it doesn't exist)
vectordb = load_embeddings(EMBEDDINGS_DIR, openai_api_key, docs)


if "chain" not in st.session_state:
    chain = create_chain(model='gpt-3.5-turbo',
                         temperature=0, vectordb=vectordb)
    st.session_state["chain"] = chain

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything to help you find the right trial!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = st.session_state.chain.run(
            prompt, callbacks=[stream_handler])

    st.session_state.messages.append(
        {"role": "assistant", "content": response})

    st.session_state.chat_history.append((prompt, response))

    # st.session_state.memory.save_context(
    #     {"input": prompt}, {"output": response})
