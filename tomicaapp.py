
import os
import openai
import sys
import datetime
#from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from dash import Dash, html, dcc, Input, Output, State



# Initialize Dash app
app = Dash(__name__)


# first load my open AI key from the .env file

load_dotenv("C:/Users/agath/Desktop/AI_LLM/Tomica Project/.env.py")
openai.api_key = os.getenv('OPENAI_API_KEY')


# Set up database directory for Chroma
persist_directory = 'docs/chroma/'


# load documents from pdfs and split

# load pdfs
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("C:/Users/agath/Desktop/AI_LLM/Tomica Project/tomica news.pdf"),
    PyPDFLoader("C:/Users/agath/Desktop/AI_LLM/Tomica Project/Tomica pdf.pdf"),
    PyPDFLoader("C:/Users/agath/Desktop/AI_LLM/Tomica Project/tomica secrets.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
   

# split the texts
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)


# create embeddings and vector database
embedding = OpenAIEmbeddings()


# create a vector database from RAG
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)


retriever=vectordb.as_retriever()


# Build prompt

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Try to answer the questions as if you were talking to a kindergardener and be friendly and enthusiastics. Use short and easy to understand languages. . 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)



# setup LLM and QA Chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})




# Define the layout with black background and updated styling
app.layout = html.Div([
    html.H2("Tomica Chatbot", style={"font-family": "Calibri", "text-align": "center", "color": "red"}),
    dcc.Store(id="conversation-history", data=[]),  # Store conversation history
    html.Div(id="chat-container", style={
        "height": "400px", "overflowY": "scroll", "padding": "10px", "background-color": "#333", 
        "border": "1px solid #444", "color": "white", "border-radius": "10px"
    }),
    html.Div([
        dcc.Textarea(id="input-question", placeholder="Type your question here...", 
                     style={"width": "80%", "height": "50px", "font-family": "Calibri", 
                            "padding": "10px", "background-color": "#222", "color": "white", 
                            "border": "1px solid #444", "border-radius": "5px"}),
        html.Button("Send", id="submit-button", 
                    style={"width": "15%", "height": "50px", "font-family": "Calibri", 
                           "background-color": "#4CAF50", "color": "white", "border-radius": "5px"})
    ], style={"display": "flex", "justify-content": "space-between", "margin-top": "10px"})
], style={"font-family": "Calibri", "width": "500px", "margin": "auto", "background-color": "black", "padding": "20px", "border-radius": "10px"})

# Callback to update conversation history, display messages, and clear input
@app.callback(
    Output("conversation-history", "data"),
    Output("chat-container", "children"),
    Output("input-question", "value"),  # Clears input after submission
    Input("submit-button", "n_clicks"),
    State("input-question", "value"),
    State("conversation-history", "data")
)
def update_chat(n_clicks, question, conversation_history):
    if n_clicks is None or not question:
        return conversation_history, conversation_history, ""  # Clears input if empty or button not clicked
    
    # Process the question through the QA chain
    result = qa_chain({"query": question})
    answer = result.get("result", "I'm sorry, I couldn't find an answer.")

    # Append question and answer to conversation history
    conversation_history.append({"sender": "User", "message": question})
    conversation_history.append({"sender": "Bot", "message": answer})

    # Display messages with WhatsApp-style bubbles
    chat_bubbles = []
    for message in conversation_history:
        align = "flex-end" if message["sender"] == "User" else "flex-start"
        bg_color = "#DCF8C6" if message["sender"] == "User" else "#FFFFFF"
        text_color = "#333" if message["sender"] == "User" else "#111"  # User text is dark on light green
        bubble = html.Div(message["message"], style={
            "max-width": "70%", "padding": "10px", "border-radius": "10px",
            "margin": "5px", "background-color": bg_color, "align-self": align,
            "color": text_color, "box-shadow": "0px 1px 2px rgba(0, 0, 0, 0.1)"
        })
        chat_bubbles.append(html.Div(bubble, style={"display": "flex", "justify-content": align}))

    return conversation_history, chat_bubbles, ""  # Clear input after sending

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)


# #%%
# question = "Tell me some fun facts about Tomica?"
# result = qa_chain({"query": question})
# result["result"]

# print(result["result"])

