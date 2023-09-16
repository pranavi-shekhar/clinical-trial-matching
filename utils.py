import xml.etree.ElementTree as ET
import pandas as pd
import os
import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool
from langchain import LLMChain
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool


# Classes to handle streaming


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

# Convert xml to csv as required


def xml_to_csv(xml_path, csv_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Initialize empty lists to store data
    protocol_data = []

    # Iterate through each <PROTOCOL> element
    for protocol in root.findall('.//PROTOCOL'):
        protocol_dict = {}
        # Extract data from XML tags and add it to the dictionary
        # protocol_dict['PROTOCOL_NO'] = protocol.find('PROTOCOL_NO').text
        protocol_dict['TITLE'] = protocol.find('TITLE').text
        # protocol_dict['NCT_ID'] = protocol.find('NCT_ID').text
        protocol_dict['SHORT_TITLE'] = protocol.find('SHORT_TITLE').text
        protocol_dict['INVESTIGATOR_NAME'] = protocol.find(
            'INVESTIGATOR_NAME').text
        protocol_dict['STATUS'] = protocol.find('STATUS').text
        protocol_dict['ELIGIBILITY'] = protocol.find('ELIGIBILITY').text
        protocol_dict['DETAILED_ELIGIBILITY'] = protocol.find(
            'DETAILED_ELIGIBILITY').text if protocol.find('DETAILED_ELIGIBILITY') is not None else ''
        protocol_dict['DESCRIPTION'] = protocol.find('DESCRIPTION').text
        protocol_dict['PHASE_DESC'] = protocol.find('PHASE_DESC').text
        protocol_dict['TREATMENT_TYPE_DESC'] = protocol.find(
            'TREATMENT_TYPE_DESC').text
        protocol_dict['AGE_DESCRIPTION'] = protocol.find(
            'AGE_DESCRIPTION').text
        protocol_dict['SCOPE_DESC'] = protocol.find('SCOPE_DESC').text
        protocol_dict['MODIFIED_DATE'] = protocol.find('MODIFIED_DATE').text
        protocol_dict['DEPARTMENT_NAME'] = protocol.find(
            'DEPARTMENT_NAME').text
        # Extract SPONSOR_NAMES
        sponsor_names = [
            sponsor.text for sponsor in protocol.findall('.//SPONSOR_NAME')]
        protocol_dict['SPONSOR_NAMES'] = ', '.join(sponsor_names)
        # Extract DISEASE_SITES
        disease_sites = [
            site.text for site in protocol.findall('.//DISEASE_SITE')]
        protocol_dict['DISEASE_SITES'] = ', '.join(disease_sites)
        # Extract DRUG_NAMES (if available)
        drugs = protocol.findall('.//DRUG_NAMES')
        if drugs:
            protocol_dict['DRUG_NAMES'] = ', '.join(
                [drug.text if drug.text is not None else '' for drug in drugs])
        else:
            protocol_dict['DRUG_NAMES'] = ''

        # Extract THERAPY_NAMES (if available)
        therapies = protocol.findall('.//THERAPY_NAMES')
        if therapies:
            protocol_dict['THERAPY_NAMES'] = ', '.join(
                [therapy.text if therapy.text is not None else '' for therapy in therapies])
        else:
            protocol_dict['THERAPY_NAMES'] = ''

        # Append the protocol data to the list
        protocol_data.append(protocol_dict)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(protocol_data)
    df.to_csv(csv_path,  index=False)


def load_document(filepath, load_and_split=False):
    loader = CSVLoader(filepath)
    docs = loader.load() if load_and_split == False else loader.load_and_split()
    return docs


def load_embeddings(path_to_embeddings, openai_api_key, docs):
    # Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Use chroma to save/load embeddings
    if os.path.isdir(path_to_embeddings):
        print("Loading stored embeddings...")
        vectordb = FAISS.load_local(path_to_embeddings, embeddings)

    else:
        print("Generating embeddings...")
        vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
        vectordb.save_local(path_to_embeddings)

    return vectordb


def create_agent(model, temperature, vectordb):
    # Create tool
    tool = create_retriever_tool(
        vectordb.as_retriever(),
        "search_clinical_trials_database",
        "Searches and returns documents regarding clinical trials")

    tools = [tool]

    # Define llm
    llm = ChatOpenAI(temperature=temperature,
                     model=model,
                     streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

    # Create agent
    agent_executor = create_conversational_retrieval_agent(llm, tools)

    return agent_executor
