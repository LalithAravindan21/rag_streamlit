import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os

st.set_page_config(page_title="Chat with Alina 💬")

nlp = spacy.load("en_core_web_sm")

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

os.environ['GOOGLE_API_KEY'] = "AIzaSyBVi-KWLLyIT23lpIlb9zZ_eXKQVaJdhE0"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text based on semantic similarity and NER
def advanced_chunking(text):
    # Splitting the text into sentences using spaCy
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Applying NER to extract important entities
    ner_entities = []
    for ent in doc.ents:
        ner_entities.append(ent.text)
    
    # Computing embeddings for sentences using Sentence-BERT
    sentence_embeddings = sentence_model.encode(sentences)
    
    # Computing pairwise cosine similarity between sentence embeddings
    similarity_matrix = cosine_similarity(sentence_embeddings)
    
    # Grouping sentences based on semantic similarity
    chunks = []
    current_chunk = []
    current_chunk_embedding = None
    
    for i, sentence in enumerate(sentences):
        if current_chunk_embedding is None:
            current_chunk.append(sentence)
            current_chunk_embedding = sentence_embeddings[i]
        else:
            similarity_score = cosine_similarity([current_chunk_embedding], [sentence_embeddings[i]])[0][0]
            if similarity_score > 0.8:  # Threshold for semantic similarity
                current_chunk.append(sentence)
                # Updating the current chunk embedding (average embeddings for new chunk)
                current_chunk_embedding = (current_chunk_embedding + sentence_embeddings[i]) / 2
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_embedding = sentence_embeddings[i]

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    final_chunks = []
    for chunk in chunks:
        if any(entity in chunk for entity in ner_entities):
            final_chunks.append(chunk)
    
    return final_chunks

# Function to generate vector embeddings and store them in FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to get conversational chain with LLM and vector store retriever
def get_conversational_chain(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

# Function to handle user inputs and display chat history
def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("User: ", message.content)
        else:
            st.write("Alina: ", message.content)

# Main function for the app
def main():
    st.header("Chat with Alina 💬")

    # Sidebar for uploading PDF documents
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = advanced_chunking(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Processing Complete!")

    # Chat functionality
    user_question = st.chat_input("Ask a question regarding the PDF")
    
    # Initializing session state if not already done
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
