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

st.set_page_config(page_title="Chat with Multilpe PDF's💬")

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

# Updated function to handle long text inputs by splitting into chunks
def advanced_chunking(text):
    max_length = 50000  # Adjust based on memory limits and needs
    text_chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    processed_chunks = []
    for chunk in text_chunks:
        doc = nlp(chunk)
        sentences = [sent.text for sent in doc.sents]
        ner_entities = [ent.text for ent in doc.ents]
        sentence_embeddings = sentence_model.encode(sentences)
        
        current_chunk = []
        current_chunk_embedding = None
        for i, sentence in enumerate(sentences):
            if current_chunk_embedding is None:
                current_chunk.append(sentence)
                current_chunk_embedding = sentence_embeddings[i]
            else:
                similarity_score = cosine_similarity([current_chunk_embedding], [sentence_embeddings[i]])[0][0]
                if similarity_score > 0.8:
                    current_chunk.append(sentence)
                    current_chunk_embedding = (current_chunk_embedding + sentence_embeddings[i]) / 2
                else:
                    processed_chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_chunk_embedding = sentence_embeddings[i]

        if current_chunk:
            processed_chunks.append(" ".join(current_chunk))

    final_chunks = [chunk for chunk in processed_chunks if any(entity in chunk for entity in ner_entities)]
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
    if st.session_state.conversation:
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chatHistory = response['chat_history']
            
            for i, message in enumerate(st.session_state.chatHistory):
                if i % 2 == 0:
                    st.write("User: ", message.content)
                else:
                    st.write("Alina: ", message.content)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload and process a document first.")

# Main function for the app
def main():
    st.header("Chat with Multilpe PDF's💬")

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
