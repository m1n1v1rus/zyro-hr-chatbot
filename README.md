# 🏢 Zyro Dynamics HR Intelligence System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge)

An advanced, AI-powered HR Help Desk assistant built for **Zyro Dynamics (Acrux Dynamics)**. This application uses a Retrieval-Augmented Generation (RAG) pipeline to instantly answer employee queries by accurately extracting information directly from official HR policy documents.

## ✨ Key Features

- **High-Precision RAG Pipeline**: Extracts exact policy details using FAISS vector search and HuggingFace MiniLM embeddings.
- **Smart Query Classification (Guardrails)**: Automatically detects and blocks out-of-scope questions (e.g., coding, external hiring) while allowing valid HR queries (Leave, ESOP, CTC, WFH).
- **Powered by LLMs**: Leverages Groq's ultra-fast `llama-3.3-70b-versatile` model for high-quality, instant answers.
- **Premium UI/UX**: A responsive, dark-mode Glassmorphism interface built natively on Streamlit with custom CSS.
- **Source Transparency**: Every answer includes clickable source pills so employees know exactly which policy document was referenced.

## 🚀 Quickstart Guide

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the application**
```bash
streamlit run app.py
```

**3. Configuration**
- Open the local URL provided by Streamlit.
- Enter your **Groq API Key** in the sidebar.
- Start asking HR questions!

## 🛠️ Tech Stack
- **Frontend**: Streamlit, Custom CSS
- **Backend**: LangChain, FAISS
- **Models**: Llama 3.3 70B (LLM via Groq), all-MiniLM-L6-v2 (Embeddings)
