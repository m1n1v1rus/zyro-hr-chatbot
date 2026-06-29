import os
import time
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro Dynamics · HR Assistant", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif !important; }
    #MainMenu, footer { visibility: hidden; }
    .stApp {
        background: #0c0a1a;
        background-image:
            radial-gradient(ellipse 80% 60% at 10% 20%, rgba(88, 28, 135, 0.15), transparent),
            radial-gradient(ellipse 60% 50% at 90% 80%, rgba(15, 23, 42, 0.3), transparent),
            radial-gradient(ellipse 40% 40% at 50% 10%, rgba(56, 189, 248, 0.08), transparent);
        color: #e2e8f0;
    }
    @keyframes shimmer {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .zd-header {
        background: linear-gradient(270deg, #7c3aed, #2563eb, #06b6d4, #7c3aed);
        background-size: 300% 300%;
        animation: shimmer 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 5px;
    }
    .zd-tagline { color: #94a3b8; font-size: 0.95rem; font-weight: 400; margin-bottom: 30px; }
    .stChatMessage {
        background: rgba(15, 13, 31, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
        border: 1px solid rgba(124, 58, 237, 0.15) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    [data-testid="stChatInput"] {
        border-radius: 28px !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        background: rgba(15, 13, 31, 0.9) !important;
        backdrop-filter: blur(8px) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2), 0 8px 32px rgba(0,0,0,0.3) !important;
    }
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 12px !important;
        overflow: hidden;
        margin-top: 12px !important;
    }
    [data-testid="stExpander"] summary {
        color: #7dd3fc !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        background: rgba(15, 23, 42, 0.8) !important;
    }
    [data-testid="stExpander"] summary:hover { color: #38bdf8 !important; }
    [data-testid="stExpanderDetails"] { padding: 12px 16px !important; font-size: 0.85rem; color: #bae6fd; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🚀 Zyro Dynamics")
    st.markdown("HR Intelligence Platform")
    st.divider()
    groq_key = st.text_input("🔑 Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""), placeholder="Enter API Key")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="zd-header">Zyro Dynamics HR Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="zd-tagline">Ask anything about company HR policies, leave, salary, and more.</div>', unsafe_allow_html=True)


@st.cache_resource
def load_pipeline(api_key):
    corpus_path = os.environ.get("CORPUS_PATH", os.path.join(os.path.dirname(__file__), "hr_docs"))
    if not os.path.isdir(corpus_path):
        corpus_path = "/kaggle/input/zyro-dynamics-hr-corpus/"

    loader = PyPDFDirectoryLoader(corpus_path)
    docs = loader.load()

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""],
        is_separator_regex=False
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 40]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.85}
    )


    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=1024,
        api_key=api_key,
    )

    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an HR Help Desk assistant for Zyro Dynamics (also called Acrux Dynamics — "
         "treat them as the same company).\n\n"
         "Answer the employee's question using ONLY the HR policy context provided below.\n"
         "- Stay as close as possible to the exact wording, phrasing, and structure used in "
         "the source policy document. Do not rephrase into your own words if the document's "
         "own sentence already answers the question — use it almost verbatim.\n"
         "- Include all numbers, dates, percentages, and conditions exactly as written.\n"
         "- If the context discusses multiple similar items (e.g., different leave types or "
         "insurance types), answer ONLY about the specific one asked.\n"
         "- If the context does not contain the answer, say: "
         "\"I cannot answer this based on the available HR policy documents.\"\n"
         "- Keep the answer clear and concise. Use bullet points where the source itself "
         "uses a list or table."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    
    oos_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a classifier for an HR help desk.\n"
         "Determine if the question can be answered using Zyro Dynamics HR policy documents.\n"
         "Topics covered: company profile, employee handbook, leave policy (sick, casual, earned, maternity),\n"
         "work from home, code of conduct, performance review, compensation & benefits (salary, insurance),\n"
         "IT & data security, POSH, onboarding & separation, travel & expense.\n\n"
         "Respond with EXACTLY ONE WORD: \"IN_SCOPE\" or \"OUT_OF_SCOPE\".\n\n"
         "Examples:\n"
         "Q: How many sick leaves do I get? -> IN_SCOPE\n"
         "Q: How do I apply for WFH? -> IN_SCOPE\n"
         "Q: What is the CTC range for L4 Senior grade? -> IN_SCOPE\n"
         "Q: What is the health insurance coverage? -> IN_SCOPE\n"
         "Q: What is the Annual Performance Review timeline? -> IN_SCOPE\n"
         "Q: If an employee takes sick leave for more than 2 days, what is required? -> IN_SCOPE\n"
         "Q: What is the vesting schedule for ESOP? -> OUT_OF_SCOPE\n"
         "Q: How many stock options will I receive as a new joiner? -> OUT_OF_SCOPE\n"
         "Q: How can I apply for a job at Acrux Dynamics? -> OUT_OF_SCOPE\n"
         "Q: What is the recruitment and hiring process? -> OUT_OF_SCOPE\n"
         "Q: What was the company revenue last year? -> OUT_OF_SCOPE\n"
         "Q: How does AcruxCRM compare to Salesforce? -> OUT_OF_SCOPE\n"
         "Q: What is the leave policy at Zoho or Freshworks? -> OUT_OF_SCOPE\n"
         "Q: What is the meaning of life? -> OUT_OF_SCOPE\n"
         "Q: What is the weather today? -> OUT_OF_SCOPE"),
        ("human", "Question: {question}")
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join([
            f"Source: {d.metadata.get('source', 'Unknown').split('/')[-1]}\n{d.page_content}"
            for d in docs
        ])

    return retriever, llm, rag_prompt, oos_prompt, format_docs


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the Zyro Dynamics HR Assistant. How can I help you today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 View Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s.split('/')[-1]}**")

if prompt := st.chat_input("Ask your HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not groq_key:
            st.error("Please enter your Groq API Key in the sidebar.")
            st.stop()

        with st.spinner("Searching HR policies..."):
            retriever, llm, rag_prompt, oos_prompt, format_docs = load_pipeline(groq_key)

            guard_chain = oos_prompt | llm | StrOutputParser()
            guard_result = guard_chain.invoke({"question": prompt})
            time.sleep(2)

            if guard_result.strip().upper() != "IN_SCOPE":
                answer = "I can only answer questions about Zyro Dynamics HR policies from the provided documents."
                sources = []
            else:
                docs = retriever.invoke(prompt)
                context = format_docs(docs)
                chain = rag_prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": prompt})
                sources = list(set(
                    d.metadata.get("source", "Unknown").split("/")[-1] for d in docs
                ))
            time.sleep(2)

            st.markdown(answer)
            if sources:
                with st.expander("📄 View Sources"):
                    for s in sources:
                        st.markdown(f"- **{s.split('/')[-1]}**")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
