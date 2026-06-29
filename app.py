import streamlit as st
import os
import time
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

# ==========================================
# 💎 PREMIUM UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Zyro HR Assistant", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Background */
    .stApp {
        background-color: #0f111a;
        color: #e2e8f0;
    }

    /* Gradient Title */
    .premium-title {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .premium-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 20px;
    }

    /* Sidebar Styling (Glassmorphism) */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: #1e2235 !important;
        border-radius: 16px !important;
        padding: 10px 20px !important;
        margin-bottom: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        transition: transform 0.2s ease;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
    }

    /* Input Box Styling */
    [data-testid="stChatInput"] {
        border-radius: 24px !important;
        border: 1px solid #334155 !important;
        background-color: #1e2235 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #4facfe !important;
        box-shadow: 0 0 0 2px rgba(79, 172, 254, 0.2) !important;
    }

    /* Success / Info Boxes */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }
    
    .src-badge {
        background: rgba(25, 118, 210, 0.1); border-left: 3px solid #1976d2;
        padding: 6px 10px; border-radius: 4px;
        font-size: 0.82em; margin-top: 6px; color: #90caf9;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<h1 class="premium-title">Zyro HR Assistant ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="premium-subtitle">Enterprise-grade HR Policy resolution powered by Advanced RAG.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<h3 style="color:#4facfe;">Configuration</h3>', unsafe_allow_html=True)
    groq_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    st.divider()
    st.info("Topics: Leave, Salary, WFH, Performance, Insurance, Conduct.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the Zyro Dynamics HR Assistant. How can I help you today?"}]

REFUSAL_MESSAGE = "I can only answer questions about Zyro Dynamics HR policies from the provided documents."

@st.cache_resource
def build_rag(api_key):
    pdf_dir = "./hr_docs"
    if not os.path.isdir(pdf_dir):
        pdf_dir = "/kaggle/input/zyro-dynamics-hr-corpus/"

    loader = PyPDFDirectoryLoader(pdf_dir, glob="*.pdf", silent_errors=True)
    docs = loader.load()
    docs = [d for d in docs if d.page_content.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 40]

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.from_documents(chunks, emb)
    ret = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1, max_tokens=512, groq_api_key=api_key)

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are an HR assistant for Zyro Dynamics (also referred to as Acrux Dynamics).\n"
         "Answer using ONLY the provided context.\n\n"
         "CRITICAL RULES:\n"
         "- Extract exact numbers, days, months, percentages, and amounts from the context.\n"
         "- When asked about timelines, cite the EXACT duration and condition from policy.\n"
         "- Differentiate clearly between different leave types, insurance types, and policy sections.\n"
         "- If context mentions multiple similar items, answer ONLY about the specific one asked.\n"
         "- The context IS sufficient if it contains the policy rules that answer the question.\n"
         "- Cite the source policy name in your answer.\n"
         "- If the context lacks information, say: \"I cannot answer this based on the available HR policy documents.\"\n"
         "- Be concise and accurate."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    OOS_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a classifier for an HR help desk.\n"
         "Determine if the question can be answered using Zyro Dynamics HR policy documents.\n"
         "Topics covered: company profile, employee handbook, leave policy (sick, casual, earned, maternity),\n"
         "work from home, code of conduct, performance review, compensation & benefits (salary, insurance, ESOPs),\n"
         "IT & data security, POSH, onboarding & separation, travel & expense.\n\n"
         "Respond with EXACTLY ONE WORD: \"IN_SCOPE\" or \"OUT_OF_SCOPE\".\n\n"
         "Examples:\n"
         "Q: How many sick leaves do I get? -> IN_SCOPE\n"
         "Q: What is the vesting schedule for ESOP? -> IN_SCOPE\n"
         "Q: What is the meaning of life? -> OUT_OF_SCOPE\n"
         "Q: How do I apply for WFH? -> IN_SCOPE\n"
         "Q: Tell me a joke -> OUT_OF_SCOPE\n"
         "Q: What is Python programming? -> OUT_OF_SCOPE\n"
         "Q: How is the claim process for medical insurance? -> IN_SCOPE\n"
         "Q: What is the weather today? -> OUT_OF_SCOPE"),
        ("human", "Question: {question}")
    ])

    return ret, llm, RAG_PROMPT, OOS_PROMPT

def format_docs(docs):
    return "\n\n---\n\n".join([
        f"Source: {d.metadata.get('source', 'Unknown').split('/')[-1]}\n{d.page_content}"
        for d in docs
    ])

def _invoke_with_retry(chain, inputs, max_retries=5):
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                time.sleep(10 * (attempt + 1))
            else:
                raise e
    raise Exception("Max retries exceeded due to rate limiting.")

def ask(question, ret, llm, prompt, oos_prompt):
    oos_chain = oos_prompt | llm | StrOutputParser()
    verdict = _invoke_with_retry(oos_chain, {"question": question}).strip().upper()
    time.sleep(2)
    
    if "OUT" in verdict:
        return REFUSAL_MESSAGE, []
    
    docs = ret.invoke(question)
    context = format_docs(docs)
    chain = prompt | llm | StrOutputParser()
    answer = _invoke_with_retry(chain, {"context": context, "question": question})
    time.sleep(2)
    
    sources = list(set(d.metadata.get("source", "").split("/")[-1] for d in docs))
    return answer.strip(), sources

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(f'<div class="src-badge">📎 {" · ".join(msg["sources"])}</div>', unsafe_allow_html=True)

if user_input := st.chat_input("Ask an HR policy question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Looking up policies…"):
            key = groq_key or os.environ.get("GROQ_API_KEY", "")
            if not key:
                ans, srcs = "Please add your Groq API key in the sidebar.", []
            else:
                try:
                    ret, llm, prompt, oos_prompt = build_rag(key)
                    ans, srcs = ask(user_input, ret, llm, prompt, oos_prompt)
                except Exception as e:
                    ans, srcs = f"Error: {e}", []
            st.markdown(ans)
            if srcs:
                st.markdown(f'<div class="src-badge">📎 {" · ".join(srcs)}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})
