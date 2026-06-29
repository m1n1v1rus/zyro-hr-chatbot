# Zyro Dynamics HR Help Desk - RAG Chatbot

This repository contains the Streamlit application for the Zyro Dynamics HR Help Desk RAG Challenge.

## Features
- **RAG Pipeline**: Retrieves context from HR policy PDFs to answer employee queries.
- **Guardrails**: Accurately classifies whether questions are IN_SCOPE or OUT_OF_SCOPE based on policy coverage.
- **LLM Powered**: Uses Groq's fast Llama-3.3-70b-versatile model for high-quality responses.
- **Clean UI**: Built with Streamlit, providing an easy-to-use conversational interface.

## Running Locally

1. Create a virtual environment and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Enter your Groq API key in the sidebar configuration to start querying.
