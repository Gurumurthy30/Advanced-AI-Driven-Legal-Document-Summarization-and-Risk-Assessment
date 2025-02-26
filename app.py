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
    # Replace HuggingFaceEmbeddings with Gensim Word2Vec
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    from nltk.tokenize import word_tokenize
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
if 'word2vec_model' not in st.session_state:
    st.session_state.word2vec_model = None

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

# Custom Word2Vec embeddings for LangChain
class Word2VecEmbeddings:
    def __init__(self, model=None):
        self.model = model
        self.vector_size = model.vector_size if model else 100
        
    def embed_documents(self, texts):
        """Embed a list of texts as a list of embeddings."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text):
        """Embed a query text."""
        # Preprocess and tokenize
        tokens = simple_preprocess(text)
        
        # Get vectors for each token and compute average
        vec = np.zeros(self.vector_size)
        count = 0
        for token in tokens:
            if token in self.model.wv:
                vec += self.model.wv[token]
                count += 1
        
        # Normalize if there are any tokens in the vocabulary
        if count > 0:
            vec /= count
        return vec.tolist()

# Function to train Word2Vec model on document text
def train_word2vec(texts, vector_size=100, window=5, min_count=1):
    # Preprocess and tokenize all texts
    tokenized_texts = [simple_preprocess(text) for text in texts]
    
    # Train the model
    model = Word2Vec(sentences=tokenized_texts, 
                     vector_size=vector_size, 
                     window=window, 
                     min_count=min_count,
                     workers=4)
    
    # Return the trained model
    return model

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
                st.warning(f"Error extracting images from page {page_num+1}: {str(img_error)}")
                continue
                
        return "\n".join(full_text)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""

# Function for document summarization
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
            st.warning(f"Progress tracking error (non-critical): {str(progress_error)}")
        finally:
            # Ensure progress bar shows completion
            progress_bar.progress(1.0)
        
        intermediate_summary = "\n".join(all_chunk_summaries)
        final_summary = combine_chain.run(text=intermediate_summary)
        
        return documents, final_summary
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        st.error(traceback.format_exc())
        return [], "Summarization failed due to an error."
    
# Custom Word2Vec embeddings for LangChain
class Word2VecEmbeddings:
    def __init__(self, model=None):
        self.model = model
        self.vector_size = model.vector_size if model else 100
    
    def __call__(self, text):
        """Make the class callable to match LangChain's expectations"""
        return self.embed_query(text)
        
    def embed_documents(self, texts):
        """Embed a list of texts as a list of embeddings."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text):
        """Embed a query text."""
        # Preprocess and tokenize
        tokens = simple_preprocess(text)
        
        # Get vectors for each token and compute average
        vec = np.zeros(self.vector_size)
        count = 0
        for token in tokens:
            if token in self.model.wv:
                vec += self.model.wv[token]
                count += 1
        
        # Normalize if there are any tokens in the vocabulary
        if count > 0:
            vec /= count
        return vec.tolist()
    
def create_vector_store(documents):
    global vector_store
    
    if not documents:
        st.error("No documents to create vector store from.")
        return None
        
    try:
        st.info("Training Word2Vec model on document content...")
        
        # Extract text from documents
        texts = [doc.page_content for doc in documents]
        
        # Train Word2Vec model if not already trained
        if not st.session_state.word2vec_model:
            st.session_state.word2vec_model = train_word2vec(texts)
            st.success("Word2Vec model trained successfully!")
            
        # Create embedding wrapper for the Word2Vec model
        embedding_model = Word2VecEmbeddings(model=st.session_state.word2vec_model)
        
        # Create FAISS vector store with debug info
        st.info(f"Creating vector store with {len(documents)} documents...")
        vector_store = FAISS.from_documents(documents, embedding_model)
        st.session_state.vector_store = vector_store  # Store in session state
        st.success("Vector store created successfully!")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        st.error(traceback.format_exc())
        
        # Fallback to using a simpler, more compatible approach
        try:
            st.warning("Custom embedding failed. Trying a simpler Word2Vec approach...")
            
            # Create embeddings manually
            document_embeddings = []
            for doc in documents:
                tokens = simple_preprocess(doc.page_content)
                vec = np.zeros(st.session_state.word2vec_model.vector_size)
                count = 0
                for token in tokens:
                    if token in st.session_state.word2vec_model.wv:
                        vec += st.session_state.word2vec_model.wv[token]
                        count += 1
                if count > 0:
                    vec /= count
                document_embeddings.append(vec)
            
            # Create FAISS index manually
            import faiss
            dimension = st.session_state.word2vec_model.vector_size
            index = faiss.IndexFlatL2(dimension)
            document_embeddings = np.array(document_embeddings).astype('float32')
            index.add(document_embeddings)
            
            # Create a custom retrieval function
            def custom_retrieval(query_text, k=3):
                tokens = simple_preprocess(query_text)
                query_vec = np.zeros(dimension)
                count = 0
                for token in tokens:
                    if token in st.session_state.word2vec_model.wv:
                        query_vec += st.session_state.word2vec_model.wv[token]
                        count += 1
                if count > 0:
                    query_vec /= count
                query_vec = np.array([query_vec]).astype('float32')
                D, I = index.search(query_vec, k)
                results = [documents[i] for i in I[0] if i < len(documents)]
                return results
            
            st.session_state.custom_retriever = custom_retrieval
            st.success("Created custom retrieval function!")
            return True  # Return True to indicate success
        except Exception as fallback_error:
            st.error(f"All embedding approaches failed: {str(fallback_error)}")
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
        st.error(f"Error during RAG processing: {str(e)}")
        st.error(traceback.format_exc())
        return f"An error occurred while processing your query. Please try again."
    
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
        st.error(f"Error during risk analysis: {str(e)}")
        return "Risk analysis failed due to an error."

# === Streamlit UI ===
st.title("📜 Legal Document Assistant")
st.markdown("Upload a legal document and choose an action!")

# Create sidebar for instructions and details
with st.sidebar:
    st.header("About this App")
    st.markdown("""
    This application helps analyze legal documents by:
    - Extracting text from PDFs (including text in images)
    - Providing a chat interface to ask questions about the document
    - Generating document summaries
    - Analyzing potential legal risks
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Upload a PDF document
    2. Extract text from the document
    3. Choose an action: Chat, Summarize, or Analyze Risks
    """)
    
    # Add embedding model info
    st.header("Technical Notes")
    st.markdown("""
    - This app uses Word2Vec for document embeddings
    - Word2Vec is trained on your document's content
    - This approach provides faster processing and better semantic understanding
    """)

# Main app area
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Extract text button
    if st.button("Extract Text"):
        with st.spinner("Extracting text from document..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
            if extracted_text:
                st.session_state.extracted_text = extracted_text
                st.success("Text extracted successfully!")
                # Show a small preview of the extracted text
                st.markdown("### Text Preview")
                st.markdown(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
            else:
                st.error("Failed to extract text from the document.")
    
    # Check if text has been extracted
    if st.session_state.extracted_text:
        extracted_text = st.session_state.extracted_text
        
        # Select task
        task = st.selectbox("Choose an option:", ["Chat with Document (RAG)", "Summarization", "Risk Analysis"])
        
        if task == "Chat with Document (RAG)":
            # Initialize vector store if not already done
            if not st.session_state.vector_store_initialized:
                with st.spinner("Preparing document for chat... This may take a few minutes for large documents."):
                    documents, summary = summarization(extracted_text)
                    if documents and create_vector_store(documents):
                        st.session_state.vector_store_initialized = True
                        st.session_state.document_summary = summary
                        st.success("Document prepared for chat!")
                    else:
                        st.error("Failed to prepare document for chat.")
            
            if st.session_state.vector_store_initialized:
                query = st.text_input("Enter your query about the document:")
                if st.button("Ask Document"):
                    if query:
                        with st.spinner("Generating response..."):
                            response = RAG(query)
                        st.subheader("📌 Response:")
                        st.write(response)
                    else:
                        st.warning("Please enter a query.")
        
        elif task == "Summarization":
            if st.button("Generate Summary"):
                with st.spinner("Summarizing... This may take a few minutes for large documents."):
                    documents, final_summary = summarization(extracted_text)
                    if documents and final_summary:
                        # Store for later use
                        st.session_state.document_summary = final_summary
                        if not st.session_state.vector_store_initialized and create_vector_store(documents):
                            st.session_state.vector_store_initialized = True
                
                if st.session_state.document_summary:
                    st.subheader("📌 Document Summary:")
                    st.write(st.session_state.document_summary)
        
        elif task == "Risk Analysis":
            if st.button("Analyze Risks"):
                # Check if we already have a summary
                if st.session_state.document_summary:
                    final_summary = st.session_state.document_summary
                else:
                    with st.spinner("Summarizing document first... This may take a few minutes."):
                        documents, final_summary = summarization(extracted_text)
                        if documents and final_summary:
                            st.session_state.document_summary = final_summary
                            if not st.session_state.vector_store_initialized and create_vector_store(documents):
                                st.session_state.vector_store_initialized = True
                
                if st.session_state.document_summary:
                    with st.spinner("Analyzing risks..."):
                        risk_output = risk_analysis(final_summary)
                    st.subheader("⚠️ Risk Analysis Report:")
                    st.write(risk_output)
else:
    st.info("Please upload a PDF document to get started.")