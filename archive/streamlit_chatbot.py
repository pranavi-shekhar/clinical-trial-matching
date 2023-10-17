import os
import sys
import streamlit as st
from streamlit_chat import message
from utils import *
from config import *

# Get OpenAI key
openai_api_key = os.environ['OPENAI_API_KEY']

# Load documents
docs = load_document(os.path.join(DATA_DIR, TRAINING_FILE))

# Load vector database (train if it doesn't exist)
vectordb = load_embeddings(EMBEDDINGS_DIR, openai_api_key, docs)

# Get LLM chain
chain = create_chain(model='gpt-3.5-turbo', temperature=0, vectordb=vectordb)


# Handle chat history display
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = [
        "Hello! Ask me anything about the clinical trials database"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey!"]

# container for the chat history
response_container = st.container()

# container for the user's text input
container = st.container()


with container:
    with st.form(key='my_form', clear_on_submit=True):

        user_input = st.text_input(
            "Query:", placeholder="Ask about clinical trials here (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = chain({"question": user_input})['answer']

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(
                i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i],
                    key=str(i), avatar_style="thumbs")
