import streamlit as st
import fitz  # PyMuPDF
import time
import io
import os
import logging
import ssl
import smtplib
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import pytesseract

try:
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain.chains.summarize import load_summarize_chain
    from langchain.vectorstores.faiss import FAISS
    from langchain_groq import ChatGroq
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.tools import Tool
    from langchain.agents import initialize_agent, AgentType
except ImportError as e:
    st.error(f"Missing required packages. Please install requirements.txt: {str(e)}")
    st.stop()

# Streamlit Page Configuration
st.set_page_config(page_title="Legal Document Assistant", layout="wide")

# Custom Streamlit UI Styling
st.markdown(
    """
    <style>
        /* Main background and text color */
        .stApp {
            background-color: #0D1117 !important;
            color: #C9D1D9 !important;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #161B22 !important;
            border-right: 1px solid #30363D;
        }

        /* Sidebar text input fields */
        .stSidebar .stTextInput>div>div>input {
            background-color: #0D1117 !important;
            color: #C9D1D9 !important;
            border: 1px solid #30363D !important;
        }

        /* Text inputs */
        .stTextInput>div>div>input {
            background-color: #161B22 !important;
            color: #C9D1D9 !important;
            border: 1px solid #30363D !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #E0E0E0 !important;
            color: #0D1117 !important;
            border-radius: 6px !important;
            border: none !important;
            padding: 8px 16px !important;
        }
        .stButton>button:hover {
            background-color: #BFBFBF !important;
        }

        /* Progress bar */
        .stProgress>div>div>div {
            background-color: #10A37F !important;
        }

        /* File uploader styling */
        div[data-testid="stFileUploader"] div[data-testid="stFileUploadDropzone"] {
            background-color: #161B22 !important;
            border: 2px dashed #30363D !important;
            color: #C9D1D9 !important;
        }
        div[data-testid="stFileUploader"] button[data-testid="stFileUploaderRemoveFile"] {
            color: #FF4B4B !important;
        }

        /* Markdown text color */
        .stMarkdown {
            color: #C9D1D9 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'risk_analysis_results' not in st.session_state:
    st.session_state.risk_analysis_results = None
if 'text_risk_analysis' not in st.session_state:
    st.session_state.text_risk_analysis = None
if 'risk_chart_data' not in st.session_state:
    st.session_state.risk_chart_data = None
if 'risk_pie_data' not in st.session_state:
    st.session_state.risk_pie_data = None

# Load API Keys
load_dotenv()

# Check for API keys
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
os.environ['EMAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')
os.environ['EMAIL_SENDER'] = os.getenv('EMAIL_SENDER')

# Initialize LLM with error handling
try:
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.3
    )
except Exception as e:
    st.error(f"Failed to initialize Groq LLM: {str(e)}")
    st.info("Please check your API key and internet connection.")
    st.stop()

# Initialize Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def Extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf") 
    full_text = []

    for page_num in range(len(doc)):
        page  = doc[page_num]
        text = page.get_text("text")

        if text.strip():
            full_text.append(text)
        else:
            image_list = page.get_images(page)
            extracted_text=""

            for img_index,img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes))
                extracted_text+=pytesseract.image_to_string(img_pil) + "/n"
                full_text.append(extracted_text)
    return("\n".join(full_text))

# Function for document summarization
def summarization(extracted_text):
    if not extracted_text or extracted_text.strip() == "":
        st.error("No text to summarize. Please extract text from a document first.")
        return [], ""
        
    try:
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=600)
        chunks = text_splitter.split_text(extracted_text)
        
        if not chunks:
            st.warning("No text chunks were created. The document might be empty or contain unsupported content.")
            return [], "No content to summarize."
            
        # Convert chunks to Document objects
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Define map prompt template
        map_prompt_template = PromptTemplate(
            template="""**Role:** You are an AI assistant specialized in retrieving key information from legal documents.
            **Task:** Identify and extract important essential details from the given text without summarizing, interpreting, or altering the information. Focus only on retrieving relevant important elements as they appear. Leave the unnecessary things and if already covered in a previous chunk, ignore it.
            **Text for Extraction:** {text}""",
            input_variables=["text"]
        )
        
        # Define combine prompt template
        combine_prompt_template = PromptTemplate(
            template="""**Role:** You are an AI assistant specialized in summarizing and structuring the document based on extracted legal information.
            **Task:** Analyze all extracted details and generate a comprehensive summary, preserving key facts while omitting less important details. Ensure clarity, coherence, and logical structuring of the final output.
            **Section Extracts:** {text}""",
            input_variables=["text"]
        )
        
        # Create map_reduce chain directly using LangChain's load_summarize_chain
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=combine_prompt_template,
            verbose=True
        )
        
        # Run the chain
        final_summary = chain.run(documents)
        
        return documents, final_summary
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return [], f"Summarization failed due to an error: {str(e)}"

# After summarization function - this is where you wanted me to start

def create_vector_store(documents):    
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

# Improved RAG function for document-specific queries
def Chat_doc(query):
    """Search the uploaded document for information related to the query."""
    if not query or query.strip() == "":
        return "Please enter a valid query."
    
    try:
        # Check if vector store is available in session state
        if 'vector_store' in st.session_state and st.session_state.vector_store:
            # Use the vector store to search for similar documents
            retrieved_docs = st.session_state.vector_store.similarity_search(query, k=3)
            
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
        else:
            return "No document has been processed yet. Please upload and process a document before asking document-specific questions."
    except Exception as e:
        return f"An error occurred while searching the document: {str(e)}. Please try again."

def simple_risk_analysis(document_text):
    if not document_text or document_text.strip() == "":
        return {"error": "No content to analyze for risks."}
    
    # Define risk categories with keywords
    risk_categories = {
    "Regulatory Risks": [
        "compliance", "regulation", "violation", "legal obligation", "policy breach", 
        "statutory requirement", "government approval", "licensing", "audit", "penalty", 
        "reporting failure", "non-compliance", "due diligence", "risk assessment", 
        "legal proceedings"
    ],
    "Criminal Risks": [
        "crime", "offense", "penalty", "prosecution", "conviction", "imprisonment", 
        "fraud", "forgery", "embezzlement", "bribery", "money laundering", "extortion", 
        "smuggling", "harassment", "cybercrime", "homicide", "theft", "assault", 
        "domestic violence", "illegal possession"
    ],
    "Contractual Risks": [
        "contract breach", "liability", "damages", "negligence", "compensation", 
        "settlement", "plaintiff", "defendant", "arbitration", "mediation", "remedy", 
        "indemnity", "lawsuit", "decree", "jurisdiction", "litigation", "injunction", 
        "fiduciary duty", "legal notice"
    ],
    "Property Risks": [
        "ownership", "possession", "lease", "title dispute", "eviction", "trespassing", 
        "mortgage", "foreclosure", "inheritance", "zoning laws", "land acquisition", 
        "boundary dispute", "property rights", "transfer of property", "real estate fraud", 
        "encumbrance", "adverse possession", "land survey", "occupancy"
    ],
    "Financial Risks": [
        "tax evasion", "tax fraud", "financial misreporting", "audit", "revenue loss", 
        "bankruptcy", "debt recovery", "loan default", "insolvency", "foreclosure", 
        "credit risk", "monetary penalty", "securities fraud", "insider trading", 
        "banking regulations", "investment fraud", "fiscal policy violation", 
        "money laundering", "interest liability"
    ]
}

    # Normalize text for analysis
    text = document_text.lower()
    
    # Analyze each category and detect risks
    results = {}
    total_score = 0
    
    for category, keywords in risk_categories.items():
        category_score = 0
        keyword_hits = []
        
        for keyword in keywords:
            count = text.count(keyword)
            if count > 0:
                keyword_hits.append({"keyword": keyword, "count": count})
                category_score += count
        
        results[category] = {
            "score": category_score,
            "keywords": keyword_hits
        }
        total_score += category_score
    
    # Determine overall risk level
    risk_level = "Low"
    if total_score > 20:
        risk_level = "High"
    elif total_score > 10:
        risk_level = "Medium"
    
    # Prepare visualization data
    visualization_data = {
        "categories": list(risk_categories.keys()),
        "scores": [results[category]["score"] for category in risk_categories.keys()],
        "colors": []
    }
    
    # Assign colors based on score (for visualization)
    for score in visualization_data["scores"]:
        if score == 0:
            visualization_data["colors"].append("green")
        elif score < 3:
            visualization_data["colors"].append("yellow")
        elif score < 5:
            visualization_data["colors"].append("orange")
        else:
            visualization_data["colors"].append("red")
    
    # Get top risk sentences (optional)
    sentences = [s.strip() for s in text.replace(".", ". ").split(". ") if s.strip()]
    risk_sentences = []
    
    for sentence in sentences:
        for category, keywords in risk_categories.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                risk_sentences.append(sentence)
                break
    
    # Compile final results
    final_results = {
        "risk_level": risk_level,
        "total_score": total_score,
        "category_results": results,
        "top_risk_sentences": risk_sentences[:5],  # Top 5 risky sentences
        "visualization_data": visualization_data  # Data ready for visualization
    }
    
    return final_results

def visualize_risks(risk_results):
    if "visualization_data" not in risk_results:
        return {"error": "No visualization data available"}

    viz_data = risk_results["visualization_data"]
    
    if not viz_data.get("categories") or not viz_data.get("scores"):
        return {"error": "Missing categories or scores in visualization data"}

    categories = viz_data["categories"]
    scores = viz_data["scores"]
    colors = viz_data["colors"]

    # Ensure valid data for visualization
    if not any(scores):  # If all scores are 0, display a message instead
        return {
            "bar_chart": None,
            "pie_chart": None
        }

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(categories, scores, color=colors)
    ax1.set_xlabel('Risk Categories')
    ax1.set_ylabel('Risk Score')
    ax1.set_title('Risk Analysis by Category')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center', va='bottom')

    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    non_zero_categories = [categories[i] for i, score in enumerate(scores) if score > 0]
    non_zero_scores = [score for score in scores if score > 0]

    if non_zero_scores:
        ax2.pie(non_zero_scores, labels=non_zero_categories, autopct='%1.1f%%', 
                startangle=90, colors=['yellow', 'orange', 'red', 'crimson', 'darkred'][:len(non_zero_scores)])
        ax2.set_title('Risk Distribution')
    else:
        ax2.text(0.5, 0.5, "No risks detected", ha='center', va='center')
        ax2.set_title('Risk Distribution (No Risks)')

    plt.tight_layout()

    return {"bar_chart": fig1, "pie_chart": fig2}

def initialize_agent_with_tools():
    """Initialize the LangChain agent with web search and document chat tools."""
    try:
        # Initialize the Tavily Search tool
        tavily_search = TavilySearchResults(max_results=3)
        
        # Create a list of tools
        tools = []
        
        # Always add the Tavily search tool
        tools.append(
            Tool(
                name="Legal Information Search",
                func=tavily_search.run,
                description="Useful for searching information about legal topics, laws, general legal questions, or any information not present in an uploaded document."
            )
        )
        
        # Add document-specific tools if a document has been processed
        if st.session_state.vector_store_initialized:
            # Document search tool
            tools.append(
                Tool(
                    name="Document Search",
                    func=Chat_doc,
                    description="Use this tool when you need to search within the uploaded legal document for specific information, clauses, or details mentioned in the document."
                )
            )
        
        # Initialize the agent with a structured system message
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            #verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None
    
def process_document(file1, file2=None):
    """Process a single document or compare two legal documents."""
    with st.spinner("Processing document.."):
        progress_bar = st.progress(0)  # Initialize progress bar
        
        try:
            # Extract text from first document
            text1 = Extract_text_from_pdf(file1)
            if not text1:
                st.error("Failed to extract text from the first document.")
                return "Error: First document could not be processed."
            
            progress_bar.progress(25)  # Update progress
            
            if file2:
                # Extract text from second document
                text2 = Extract_text_from_pdf(file2)
                if not text2:
                    st.error("Failed to extract text from the second document.")
                    return "Error: Second document could not be processed."
                
                combined_text = f"Document 1 ({file1.name}):\n{text1}\n\nDocument 2 ({file2.name}):\n{text2}"
            else:
                combined_text = text1  # Single document case

            # Store the extracted text in session state
            st.session_state.extracted_text = combined_text
            
            # Run summarization
            documents, summary = summarization(combined_text)
            if not documents or not summary:
                st.error("Failed to process document.")
                return "Error: Document processing failed."
            
            # Store the summary in session state
            st.session_state.document_summary = summary
            
            progress_bar.progress(50)  # Midway progress

            # Create vector store
            vector_store = create_vector_store(documents)
            if not vector_store:
                st.error("Failed to create vector store.")
                return "Error: Failed to create vector store."
                
            progress_bar.progress(75)

            # Run risk analysis
            risk_results = simple_risk_analysis(combined_text)  # Using full text for better risk detection
            st.session_state.risk_analysis_results = risk_results
            
            progress_bar.progress(90)

            # Initialize agent and update session state
            st.session_state.vector_store_initialized = True
            st.session_state.agent = initialize_agent_with_tools()
            progress_bar.progress(100)
            
            # If comparing two documents, generate comparison
            if file2:
                comparison_prompt = PromptTemplate(
                    template="""Compare these two legal documents and highlight key differences:
                    Document 1 ({doc1_name}): {doc1}
                    Document 2 ({doc2_name}): {doc2}
                    
                    Focus on legal distinctions, obligations, rights, and variations. Organize by sections or topics.""",
                    input_variables=["doc1", "doc2", "doc1_name", "doc2_name"]
                )
                comparison_chain = LLMChain(llm=llm, prompt=comparison_prompt)
                comparison_result = comparison_chain.run(
                    doc1=text1, 
                    doc2=text2,
                    doc1_name=file1.name,
                    doc2_name=file2.name
                )
                return comparison_result
            
            return summary
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return f"Error: {str(e)}"

def send_email(sender_email, sender_password, recipient_email, subject, body, attachment_path):
    if not os.path.exists(attachment_path):
        return "Failed to send email: PDF file is missing."

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with open(attachment_path, "rb") as attachment:
            part = MIMEApplication(attachment.read(), Name=os.path.basename(attachment_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
            msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()

        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {str(e)}"

def convert_summary_and_risk_to_pdf(summary, risk_analysis):
    """Convert summary and risk analysis to PDF and save it for email attachment."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Set up the document
    width, height = letter
    margin = 72  # 1 inch margin
    y_position = height - margin
    line_height = 14
    max_width = width - 2 * margin
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y_position, "Legal Document Analysis")
    y_position -= line_height * 2
    
    # Add summary section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_position, "Document Summary")
    y_position -= line_height * 1.5
    
    c.setFont("Helvetica", 12)
    summary_lines = textwrap.wrap(summary, width=85)
    for line in summary_lines:
        c.drawString(margin, y_position, line)
        y_position -= line_height
        if y_position < margin:
            c.showPage()
            y_position = height - margin
    
    y_position -= line_height * 2
    
    # Add risk analysis section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_position, "Risk Analysis")
    y_position -= line_height * 1.5
    
    c.setFont("Helvetica", 12)
    c.drawString(margin, y_position, f"Risk Level: {risk_analysis['risk_level']} (Score: {risk_analysis['total_score']})")
    y_position -= line_height * 1.5
    
    # Add categories
    for category, data in risk_analysis['category_results'].items():
        if data['score'] > 0:
            c.drawString(margin, y_position, f"{category}: Score {data['score']}")
            y_position -= line_height
            
            if y_position < margin:
                c.showPage()
                y_position = height - margin
    
    # Add risk sentences
    if risk_analysis.get('top_risk_sentences'):
        y_position -= line_height
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y_position, "High Risk Indicators:")
        y_position -= line_height
        
        c.setFont("Helvetica", 12)
        for sentence in risk_analysis['top_risk_sentences']:
            wrapped_lines = textwrap.wrap(sentence, width=85)
            for line in wrapped_lines:
                c.drawString(margin, y_position, f"• {line}")
                y_position -= line_height
                if y_position < margin:
                    c.showPage()
                    y_position = height - margin
    
    # Add risk visualization charts
    if "visualization_data" in risk_analysis:
        # Create a new page for charts
        c.showPage()
        y_position = height - margin
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y_position, "Risk Visualization")
        y_position -= line_height * 2
        
        # Create and add the bar chart
        charts = visualize_risks(risk_analysis)
        if charts.get("bar_chart") and charts.get("pie_chart"):
            # Save the charts as temporary images
            bar_chart_path = "temp_bar_chart.png"
            pie_chart_path = "temp_pie_chart.png"
            
            charts["bar_chart"].savefig(bar_chart_path, dpi=150, bbox_inches="tight")
            
            # Add bar chart to PDF
            c.drawImage(bar_chart_path, margin, y_position - 250, width=400, height=250)
            y_position -= 270
            
            # Create new page if needed
            if y_position < margin + 250:
                c.showPage()
                y_position = height - margin
            
            # Add pie chart
            charts["pie_chart"].savefig(pie_chart_path, dpi=150, bbox_inches="tight")
            c.drawImage(pie_chart_path, margin, y_position - 250, width=400, height=250)
            
            # Clean up temporary files
            try:
                os.remove(bar_chart_path)
                os.remove(pie_chart_path)
            except:
                pass
    
    # Finalize and save the PDF
    c.save()
    buffer.seek(0)
    
    pdf_path = "document_analysis.pdf"
    with open(pdf_path, "wb") as f:
        f.write(buffer.getvalue())

    st.session_state["pdf_report"] = pdf_path  # Store path for email
    return buffer.getvalue()  # Return the bytes for download button
# Main UI with sidebar for document upload and main area for chat

# File uploader in sidebar
uploaded_files = st.sidebar.file_uploader("Upload PDF Document", type="pdf", accept_multiple_files=True, key="pdf_uploader")

# Document processing section
# Document processing section
if uploaded_files:
    if st.sidebar.button("Process Document..."):
        st.session_state.document_uploaded = True
        st.session_state.document_processed = False

        if len(uploaded_files) == 1:
            # Single document processing
            summary = process_document(uploaded_files[0])
            if summary and not summary.startswith("Error"):
                st.session_state.document_processed = True

        elif len(uploaded_files) == 2:
            # Document comparison processing
            #with st.spinner("Comparing documents..."):
                comparison_result = process_document(uploaded_files[0], uploaded_files[1])

                if comparison_result and not comparison_result.startswith("Error"):
                    st.session_state.document_comparison = comparison_result
                    st.session_state.document_processed = True
                    
                    # Add comparison result to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"**Document Comparison:**\n\n{comparison_result}"
                    })
        else:
            st.sidebar.error("Please upload either 1 or 2 documents.")

# Document actions in sidebar
if st.session_state.document_processed:
    st.sidebar.subheader("Document Actions")
    
    # Buttons in a single column for consistent layout
    col1 = st.sidebar.container()
    
    # Button to show summary
    if col1.button("Show Summary"):
        if st.session_state.document_summary:
            # Add summary to chat history as an assistant message
            st.session_state.chat_history.append({
                "role": "user", 
                "content": "Show me a summary of this document."
            })
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"**Document Summary:**\n\n{st.session_state.document_summary}"
            })
    
    # Remove the standalone chart display section

    # Then modify the "Show Risk Analysis" button logic (around line 806)
    if col1.button("Show Risk Analysis"):
        if st.session_state.get("risk_analysis_results"):
            risk_data = st.session_state.risk_analysis_results
            risk_message = f"**Risk Analysis Report**\n\n"
            risk_message += f"**Overall Risk Level:** {risk_data.get('risk_level', 'Unknown')} (Score: {risk_data.get('total_score', 0)})\n\n"

            risk_message += "**Risk Categories:**\n"
            for category, data in risk_data.get("category_results", {}).items():
                risk_message += f"- {category}: Score {data['score']}\n"

            risk_sentences = risk_data.get("top_risk_sentences", [])
            if risk_sentences:
                risk_message += "\n**Key Risk Sentences:**\n"
                for i, sentence in enumerate(risk_sentences[:3], 1):
                    risk_message += f"{i}. {sentence}\n"
        # Add user and assistant messages
            st.session_state.chat_history.append({"role": "user", "content": "Show me the risk analysis."})
            st.session_state.chat_history.append({"role": "assistant", "content": risk_message})
        # Create visualization
        if "visualization_data" in risk_data:
            charts = visualize_risks(risk_data)
            if charts.get("bar_chart") and charts.get("pie_chart"):
                # Save charts as images in memory
                img_bytes_bar = io.BytesIO()
                img_bytes_pie = io.BytesIO()
                charts["bar_chart"].savefig(img_bytes_bar, format='PNG')
                charts["pie_chart"].savefig(img_bytes_pie, format='PNG')
                img_bytes_bar.seek(0)
                img_bytes_pie.seek(0)
                
                # Encode as base64 for display in HTML
                import base64
                bar_b64 = base64.b64encode(img_bytes_bar.read()).decode('utf-8')
                pie_b64 = base64.b64encode(img_bytes_pie.read()).decode('utf-8')
                
                # Add charts as HTML in the chat message
                chart_html = f"""
                <div style='display: flex; justify-content: space-between; margin-top: 10px;'>
                    <img src='data:image/png;base64,{bar_b64}' style='width:48%; border-radius:5px;'>
                    <img src='data:image/png;base64,{pie_b64}' style='width:48%; border-radius:5px;'>
                </div>
                """
                st.session_state.chat_history.append({"role": "assistant", "content": chart_html, "html": True})
    # Generate downloadable report
    if col1.button("Generate Downloadable Report"):
        if st.session_state.document_summary and st.session_state.risk_analysis_results:
            with st.spinner("Generating PDF report..."):
                pdf_bytes = convert_summary_and_risk_to_pdf(
                    st.session_state.document_summary, 
                    st.session_state.risk_analysis_results
                )
                
                st.sidebar.download_button(
                    label="Download Analysis PDF",
                    data=pdf_bytes,
                    file_name="document_analysis.pdf",
                    mime="application/pdf"
                )
                st.sidebar.success("Report generated successfully!")

    # Email functionality
    st.sidebar.subheader("Email Document Analysis")
    recipient_email = st.sidebar.text_input("Recipient Email:")
    
    if st.sidebar.button("Send Analysis by Email"):
        if recipient_email and EMAIL_SENDER and EMAIL_PASSWORD:
            if "pdf_report" not in st.session_state or not os.path.exists(st.session_state["pdf_report"]):
                # Generate the PDF if it doesn't exist
                pdf_bytes = convert_summary_and_risk_to_pdf(
                    st.session_state.document_summary, 
                    st.session_state.risk_analysis_results
                )
            
            # Send the email
            with st.spinner("Sending email..."):
                result = send_email(
                    EMAIL_SENDER, 
                    EMAIL_PASSWORD, 
                    recipient_email, 
                    "Legal Document Analysis", 
                    "Please find attached the analysis of your legal document.", 
                    st.session_state["pdf_report"]
                )
                st.sidebar.info(result)
        else:
            st.sidebar.error("Missing email configuration or recipient address.")

# Main chat interface area
st.title("Legal Document Assistant")

chat_container = st.container()
# Chat history display (around line 830)
with chat_container:
    for message in st.session_state.get("chat_history", []):
        role = message["role"]
        content = message["content"]
        is_html = message.get("html", False)

        if role == "user":
            st.markdown(f"<div style='padding:10px; background-color:#343541; border-radius:5px; margin-bottom:10px;'><b>You:</b> {content}</div>", unsafe_allow_html=True)
        elif role == "assistant":
            if is_html:
                st.markdown(f"<div style='padding:10px; background-color:#444654; border-radius:5px; margin-bottom:10px;'><b>Assistant:</b> {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding:10px; background-color:#444654; border-radius:5px; margin-bottom:10px;'><b>Assistant:</b> {content}</div>", unsafe_allow_html=True)
        elif role == "system":
            st.markdown(f"<div style='padding:5px; background-color:#2C3E50; border-radius:5px; margin-bottom:10px; font-size:0.9em;'><i>{content}</i></div>", unsafe_allow_html=True)

# Display risk charts when needed, integrated directly in the chat flow
if st.session_state.get("risk_chart_data"):
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(st.session_state.risk_chart_data["bar_chart"])
    with col2:
        st.pyplot(st.session_state.risk_chart_data["pie_chart"])

# Input area
user_input = st.text_input("Ask about your legal document:", key="input_buffer")
if st.button("Send", key="send_button"):
    if user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if st.session_state.get("agent"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run(user_input)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "Please upload and process a document first."})

        st.rerun()