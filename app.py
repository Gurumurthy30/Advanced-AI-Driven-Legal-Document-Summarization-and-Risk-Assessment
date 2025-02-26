import streamlit as st
import fitz  # PyMuPDF
import time
import pytesseract
from PIL import Image
import io
import os
from dotenv import load_dotenv
import traceback
import numpy as np

# Import langchain components with error handling
try:
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain.chains.summarize import load_summarize_chain
    from langchain.vectorstores.faiss import FAISS
    from langchain_groq import ChatGroq
    # Import langchain embeddings for sentence transformers
    from langchain.embeddings import HuggingFaceEmbeddings
    # Import NLTK for preprocessing
    import nltk
except ImportError as e:
    st.error(f"Missing required packages. Please install requirements.txt: {str(e)}")
    st.stop()

# Try to download NLTK resources, but don't crash if it fails
try:
    nltk.download('punkt', quiet=True)
except:
    st.warning("NLTK resources could not be downloaded. Some functions may be limited.")

# Streamlit Page Configuration
st.set_page_config(page_title="Legal Document Assistant", layout="wide")

# Initialize session state variables if they don't exist
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'vector_store_initialized' not in st.session_state:
    st.session_state.vector_store_initialized = False
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_uploaded' not in st.session_state:
    st.session_state.document_uploaded = False
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

# Load API Key
load_dotenv()

# Check for API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing! Please set it in your .env file.")
    st.info("Create a .env file with: GROQ_API_KEY=your_groq_api_key_here")
    st.stop()

# Initialize LLM with error handling
try:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0.3,
        max_tokens=4000
    )
except Exception as e:
    st.error(f"Failed to initialize Groq LLM: {str(e)}")
    st.info("Please check your API key and internet connection.")
    st.stop()

# Initialize Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define vector_store as a global variable
vector_store = None

# Function to extract text from PDF using PyMuPDF and pytesseract
def extract_text_from_pdf(pdf_file):
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pdf_data = pdf_file.read()
        doc = fitz.open(stream=io.BytesIO(pdf_data), filetype="pdf")
        full_text = []
        
        for page_num, page in enumerate(doc):
            # Extract text from page
            text = page.get_text("text").strip()
            if text:
                full_text.append(text)
            
            # Extract text from images in the page
            try:
                for img in page.get_images(full=True):
                    base_image = doc.extract_image(img[0])
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Extract text using pytesseract
                    img_text = pytesseract.image_to_string(image)
                    if img_text.strip():
                        full_text.append(img_text)
            except Exception as img_error:
                continue
                
        return "\n".join(full_text)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""

# Function for document summarization
def summarization(extracted_text):
    if not extracted_text or extracted_text.strip() == "":
        st.error("No text to summarize. Please extract text from a document first.")
        return [], ""
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(extracted_text)
        
        if not chunks:
            st.warning("No text chunks were created. The document might be empty or contain unsupported content.")
            return [], "No content to summarize."
            
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        chunk_prompt = PromptTemplate(
            template="Extract essential legal details: {text}",
            input_variables=["text"]
        )
        final_prompt = PromptTemplate(
            template="Summarize the extracted legal details: {text}",
            input_variables=["text"]
        )
        
        map_chain = LLMChain(llm=llm, prompt=chunk_prompt)
        combine_chain = LLMChain(llm=llm, prompt=final_prompt)
        
        batch_size = 3
        all_chunk_summaries = []
        
        # Create a progress bar
        progress_bar = st.progress(0.0)
        
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                batch_summaries = []
                
                for doc in batch:
                    summary = map_chain.run(text=doc.page_content)
                    batch_summaries.append(summary)
                    
                all_chunk_summaries.extend(batch_summaries)
                
                # Update progress - safely handle progress updates
                current_progress = min(1.0, (i + batch_size) / len(documents))
                progress_bar.progress(current_progress)
                
                # Avoid rate limiting
                if i + batch_size < len(documents):
                    time.sleep(2)
        except Exception as progress_error:
            pass
        finally:
            # Ensure progress bar shows completion
            progress_bar.progress(1.0)
        
        intermediate_summary = "\n".join(all_chunk_summaries)
        final_summary = combine_chain.run(text=intermediate_summary)
        
        return documents, final_summary
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return [], "Summarization failed due to an error."

def create_vector_store(documents):
    global vector_store
    
    if not documents:
        st.error("No documents to create vector store from.")
        return None
        
    try:
        # Use HuggingFaceEmbeddings with sentence-transformers model directly
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, embedding_model)
        st.session_state.vector_store = vector_store  # Store in session state
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None
    
        
def RAG(query):
    if not query or query.strip() == "":
        return "Please enter a valid query."
    
    try:
        # Try using custom retriever if available
        if 'custom_retriever' in st.session_state and st.session_state.custom_retriever:
            retrieved_docs = st.session_state.custom_retriever(query, k=3)
        # Try using vector store
        elif 'vector_store' in st.session_state and st.session_state.vector_store:
            retrieved_docs = st.session_state.vector_store.similarity_search(query, k=3)
        else:
            return "Document retrieval system is not available. Please extract and process a document first."
        
        if not retrieved_docs:
            return "No relevant information found in the document for this query."
            
        context = ""
        for i, doc in enumerate(retrieved_docs):
            context += f"Document {i+1}:\n{doc.page_content}\n\n"
        
        prompt = PromptTemplate(
            input_variables=["query", "retrieved_docs"],
            template="""Use the following documents to answer the query. If the information is not in the documents, 
            state that you don't have enough information to answer accurately.
            
            Query: {query}
            
            Retrieved Documents:
            {retrieved_docs}
            
            Answer:"""
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(query=query, retrieved_docs=context)
        
        return response
    except Exception as e:
        return f"An error occurred while processing your query: {str(e)}. Please try again."
    
# Function for risk analysis
def risk_analysis(final_summary):
    if not final_summary or final_summary.strip() == "":
        return "No content to analyze for risks."
        
    try:
        prompt = PromptTemplate(
            input_variables=["document_text"],
            template="""Analyze potential legal risks in the following document summary. 
            Identify key risk areas, potential liabilities, and suggest mitigation strategies:
            
            Document Summary:
            {document_text}
            
            Risk Analysis:"""
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(document_text=final_summary)
        
        return response
    except Exception as e:
        return "Risk analysis failed due to an error."

# Regular chat function
def regular_chat(query):
    if not query or query.strip() == "":
        return "Please enter a message."
    
    try:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are a helpful legal assistant. Answer the following query:
            
            Query: {query}
            
            Answer:"""
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(query=query)
        
        return response
    except Exception as e:
        return f"An error occurred. Please try again."

# Process document function
def process_document(uploaded_file):
    with st.spinner("Processing document..."):
        # Extract text
        extracted_text = extract_text_from_pdf(uploaded_file)
        if not extracted_text:
            st.error("Failed to extract text from the document.")
            return False
        
        st.session_state.extracted_text = extracted_text
        
        # Summarize and prepare for chat
        documents, summary = summarization(extracted_text)
        if not documents or not summary:
            st.error("Failed to process document.")
            return False
        
        st.session_state.document_summary = summary
        
        # Create vector store
        if create_vector_store(documents):
            st.session_state.vector_store_initialized = True
            return True
        else:
            st.error("Failed to create vector store.")
            return False

# === Streamlit UI ===
st.title("📜 Legal Document Assistant")

# Main chat interface with centered upload section
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if uploaded_file is not None and not st.session_state.document_uploaded:
        st.session_state.document_uploaded = True
        
    if st.session_state.document_uploaded and not st.session_state.document_processed:
        if st.button("Process Document"):
            success = process_document(uploaded_file)
            if success:
                st.session_state.document_processed = True
                st.success("Document processed successfully!")
            else:
                st.session_state.document_uploaded = False

    # Document specific options after uploading
    if st.session_state.document_processed:
        st.markdown("### Document Options")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Summarize Document"):
                st.session_state.chat_history.append({"role": "user", "content": "Please summarize the document."})
                with st.spinner("Generating summary..."):
                    response = st.session_state.document_summary
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        with col_b:
            if st.button("Analyze Risks"):
                st.session_state.chat_history.append({"role": "user", "content": "Please analyze the risks in this document."})
                with st.spinner("Analyzing risks..."):
                    response = risk_analysis(st.session_state.document_summary)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# Chat message container with scrolling
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")

# User input text box (available after document processing)
if st.session_state.document_processed:
    user_input = st.text_input("Type your message here...", key="user_input")

    if st.button("Send"):
        if user_input:
            # Store user input in chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Generate response using RAG
            with st.spinner("Thinking..."):
                response = RAG(user_input)
            
            # Store assistant response
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # FIX: Safely clear user input
            del st.session_state["user_input"]

            # Rerun Streamlit to refresh chat
            st.experimental_rerun()
else:
    if st.session_state.document_uploaded:
        st.info("📄 Please process the document first.")