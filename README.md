# 📄 Legal Document Assistant

An AI-powered legal document analysis tool that processes, analyzes, and generates insights from PDF documents.

---

## ✨ Key Features

- 📃 **Document Extraction**: Extracts text from PDFs using PyTesseract OCR
- 📊 **Risk Analysis**: Identifies potential legal risks with visualization
- 💬 **Interactive Q&A**: Ask questions about your document through an AI assistant
- 🔍 **Document Comparison**: Compare two legal documents for key differences
- 📝 **Automated Summarization**: Generate concise summaries of legal documents
- 📨 **Email Integration**: Share analysis reports directly from the application
- 📁 **PDF Reports**: Generate downloadable reports with analysis results

---

## 🚀 Tech Stack

- **Frontend**: Streamlit
- **AI Model**: Groq (Llama3-70B) via LangChain
- **Document Processing**: PyMuPDF, PyTesseract
- **Vector Search**: FAISS with HuggingFace embeddings
- **Visualization**: Matplotlib
- **PDF Generation**: ReportLab
- **External Search**: Tavily API

---

## 🏗️ System Architecture

The application follows this workflow:

1. **Document Processing**: Extract text from PDFs (with OCR for images)
2. **Summarization**: Create document summaries using map-reduce approach
3. **Risk Assessment And Visualization**: Analyze legal risks across multiple categories
4. **Interactive Q&A**: Answer questions using document context and web search
5. **Reporting**: Generate visual reports and PDF documents
6. **Eamil Integration**: Share analysis reports directly from the application

---

## 🔧 Quick Setup

```sh
# Clone repo
git clone https://github.com/Gurumurthy30/Advanced-AI-Driven-Legal-Document-Summarization-and-Risk-Assessment.git
cd legal-document-assistant
#install library
pip install -r requirements.txt

# Environment setup
# Create .env with: GROQ_API_KEY, TAVILY_API_KEY, EMAIL_SENDER, EMAIL_PASSWORD

# Run application
streamlit run app.py
```

> **Note**: Requires Tesseract OCR installed on your system

---

## 📋 Usage Guide

1. **Upload Document**: Upload 1-2 PDF files using the sidebar
2. **Process Document**: Click "Process Document" button
3. **View Results**: Access summary, risk analysis, or ask questions
4. **Generate Report**: Create downloadable PDF or email the analysis

---

## ⚠️ Troubleshooting

- **OCR Issues**: Verify Tesseract is properly installed and path is correct
- **Memory Errors**: Large documents may require more system resources
- **API Limits**: Check Groq/Tavily usage if you encounter rate limiting

---

## 📜 License
© 2025 Legal Document Assistant - All Rights Reserved.