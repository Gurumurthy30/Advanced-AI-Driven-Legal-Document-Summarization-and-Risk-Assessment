import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq

    
# Load API Keys
load_dotenv()

# Check for API keys
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.3
    )

def summarization(extracted_text):
    if not extracted_text or extracted_text.strip() == "":
        return [], ""
        
    try:
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=600)
        chunks = text_splitter.split_text(extracted_text)
        
        if not chunks:
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
        return [], f"Summarization failed due to an error: {str(e)}"
