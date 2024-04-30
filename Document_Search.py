# Import necessary libraries and modules
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import csv
import requests
from io import StringIO
import base64
from streamlit_chat import message
from langchain.vectorstores import Qdrant
from langchain import VectorDBQA
from langchain.embeddings import AzureOpenAIEmbeddings

# Define a Google API key
google_api_key = 'AIzaSyDepfEClCFlkJAHkNROVwRqMPzAU4SuAY0'

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Create a PDF reader object using PyPDF2
        for page in pdf_reader.pages:  # Iterate through pages in the PDF
            text += page.extract_text()  # Extract text from the current page and append it to the 'text' variable
    return text  # Return the concatenated text from all pages

# Function to load text from uploaded CSV files
def load_csv_text(uploaded_csv_files):
    text = ""
    for uploaded_file in uploaded_csv_files:
        if uploaded_file is not None:
            file_contents = uploaded_file.read().decode('utf-8')  # Read and decode the uploaded CSV file
            file_like = StringIO(file_contents)  # Create a StringIO object to read the CSV data
            csv_reader = csv.reader(file_like)  # Create a CSV reader
            for row in csv_reader:  # Iterate through CSV rows
                text += ' '.join(row) + '\n'  # Join row elements with space and add a newline, then append to 'text'
    return text  # Return the concatenated text from all CSV files

# Function to fetch text data from a given URL
def url_data(url):
    loaders = WebBaseLoader(url)  # Create a WebBaseLoader to load data from the specified URL
    data = loaders.load()  # Load data from the URL
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Create a text splitter
    docs = text_splitter.split_documents(data)  # Split the loaded data into documents
    embeddings = AzureOpenAIEmbeddings(
                azure_deployment="TextEmbeddingAda002",
                model="text-embedding-ada-002",
                openai_api_key='07b579590a6e48f2993bb34d2fd905f4',
                base_url="https://instgenaipoc.openai.azure.com/",
                openai_api_type="azure",
            )   # Create Google Palm embeddings
    vector_store = FAISS.from_documents(docs, embedding=embeddings)  # Create a vector store from the documents
    return vector_store  # Return the vector store

# Function to split a long text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)  # Create a text splitter
    chunks = text_splitter.split_text(text)  # Split the text into smaller chunks
    return chunks  # Return the list of text chunks

# Function to create a vector store for text chunks
def get_vector_store(text_chunks):
    embeddings = AzureOpenAIEmbeddings(
                azure_deployment="TextEmbeddingAda002",
                model="text-embedding-ada-002",
                openai_api_key='07b579590a6e48f2993bb34d2fd905f4',
                base_url="https://instgenaipoc.openai.azure.com/",
                openai_api_type="azure",
            )   # Create Google Palm embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Create a vector store from text chunks
    return vector_store  # Return the vector store

# Function to create a conversational chain for responding to user queries
def get_conversational_chain(vector_store):
    llm = AzureChatOpenAI(deployment_name="chatgpt45turbo",
                      model_name="gpt-35-turbo",
                      openai_api_base="https://instgenaipoc.openai.azure.com/",
                      openai_api_version="2023-05-15",
                      openai_api_key="07b579590a6e48f2993bb34d2fd905f4",
                      openai_api_type="azure")  
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # qa = ConversationalRetrievalChain.from_llm(llm=llm,
    #                                        retriever=retriever,
    #                                        return_source_documents=True,
    #                                        verbose=False)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return conversation_chain
    # Return the conversation chain

# Function to handle user input and initiate a conversation
def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})  # Start a conversation with a user's question
    st.session_state.chatHistory = response['chat_history']  # Store the conversation history
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            # User's message (right-aligned)
            st.markdown(f'<div style="color: black; background-color: #d6f1f1; padding: 5px; border-radius: 5px; margin: 0 30% 10px 0;">Question: {message.content}</div>', unsafe_allow_html=True)
        else:
            # Bot's message (left-aligned)
            st.markdown(f'<div style="color: black; background-color: #f5f5f5; padding: 5px; border-radius: 5px; margin: 0 0 10px 30%;">Response: {message.content}</div>', unsafe_allow_html=True)

# The main application function
def main():
    st.set_page_config("Search Engine For Documents & Web Pages")  # Set the Streamlit page configuration
    st.header("Search Engine For Documents & Web Pages")  # Display a header in the app

    user_question = st.text_input("Ask your Question from the Documents & Web Pages")  # Input field for user's question

    # Initialize session state variables if not present
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
        
    if user_question:
        user_input(user_question)  # Handle user input and start a conversation

    # Define a function to convert binary file to base64
    @st.cache_data
    def get_base64(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Set a background image for the app
    def set_background(png_file):
        bin_str = get_base64(png_file)
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/jpg;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)
    set_background('background4.jpg')

    # Create a sidebar in the Streamlit app to configure options
    with st.sidebar:
        st.title("Configurations")  # Display a title in the sidebar
        option = st.selectbox("Select Option", ("Document", "URL"))  # Create a dropdown to select between "Document" and "URL"

        if option == "Document":  # If "Document" is selected
            pdf_docs = st.file_uploader("Upload your CSV, PDF, or Docx Files", accept_multiple_files=True)
            # Create a file uploader to upload CSV, PDF, or Docx files and store them in 'pdf_docs'
            url = None  # Initialize 'url' to None, as it's not relevant in this case
        elif option == "URL":  # If "URL" is selected
            pdf_docs = None  # Set 'pdf_docs' to None, as it's not used in this case
            url = st.text_input("URL")  # Create a text input to enter a URL and store it in 'url'

        if st.button("Process"):  # If the "Process" button is clicked
            with st.spinner("Processing"):  # Display a spinner while processing
                raw_text = ""  # Initialize an empty string to store the extracted text

                if option == "Document" and pdf_docs:  # If "Document" is selected and files are uploaded
                    for file in pdf_docs:  # Iterate through the uploaded files
                        if file.name.endswith(('.pdf', '.PDF')):  # If the file is a PDF
                            raw_text += get_pdf_text([file])  # Extract text from the PDF and append it to 'raw_text'
                        elif file.name.endswith(('.csv', '.CSV')):  # If the file is a CSV
                            raw_text += load_csv_text([file])  # Load text from the CSV and append it to 'raw_text'
                    text_chunks = get_text_chunks(raw_text)  # Split 'raw_text' into smaller text chunks
                    vector_store = get_vector_store(text_chunks)  # Create a vector store from the text chunks
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    # Initialize a conversational chain with the vector store
                    st.success("Done")  # Display a success message when processing is complete
                elif option == "URL" and url:  # If "URL" is selected and a URL is provided
                    vector_store = url_data(url)  # Get a vector store for the URL data
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    # Initialize a conversational chain with the vector store
                    st.success("Done")  # Display a success message when processing is complete

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()  # Call the 'main' function to run the Streamlit application
