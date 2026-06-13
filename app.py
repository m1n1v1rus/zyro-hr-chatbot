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


st.set_page_config(page_title="Zyro Dynamics · HR Assistant", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ─── Base Reset ─── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    #MainMenu, footer { visibility: hidden; }

    /* ─── Background: Subtle Mesh Gradient ─── */
    .stApp {
        background: #0c0a1a;
        background-image:
            radial-gradient(ellipse 80% 60% at 10% 20%, rgba(88, 28, 135, 0.15), transparent),
            radial-gradient(ellipse 60% 50% at 90% 80%, rgba(15, 23, 42, 0.3), transparent),
            radial-gradient(ellipse 40% 40% at 50% 10%, rgba(56, 189, 248, 0.08), transparent);
        color: #e2e8f0;
    }

    /* ─── Animated Gradient Header ─── */
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
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0; padding: 0;
    }
    .zd-tagline {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 2px;
        margin-bottom: 24px;
        letter-spacing: 0.3px;
    }
    .zd-tagline span { color: #a78bfa; font-weight: 600; }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0d1f 0%, #13102a 100%) !important;
        border-right: 1px solid rgba(124, 58, 237, 0.15) !important;
    }
    .sidebar-brand {
        text-align: center;
        padding: 16px 0 8px;
    }
    .sidebar-brand .logo { font-size: 2.2rem; }
    .sidebar-brand .name {
        font-size: 1.05rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-brand .sub { font-size: 0.72rem; color: #64748b; letter-spacing: 2px; text-transform: uppercase; }
    .sidebar-divider {
        height: 1px; margin: 14px 0;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.3), transparent);
    }
    .topic-chip {
        display: inline-block;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.73rem;
        color: #c4b5fd;
        margin: 3px 2px;
    }

    /* ─── Chat Bubbles ─── */
    .stChatMessage {
        background: rgba(15, 13, 31, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 14px !important;
        padding: 14px 20px !important;
        margin-bottom: 12px !important;
        border: 1px solid rgba(124, 58, 237, 0.1) !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.2) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stChatMessage:hover {
        border-color: rgba(124, 58, 237, 0.25) !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.08) !important;
    }

    /* ─── Chat Input ─── */
    [data-testid="stChatInput"] {
        border-radius: 28px !important;
        border: 1px solid rgba(124, 58, 237, 0.2) !important;
        background: rgba(15, 13, 31, 0.8) !important;
        backdrop-filter: blur(8px) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15), 0 8px 32px rgba(0,0,0,0.2) !important;
    }

    /* ─── Source Chips ─── */
    .src-chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .src-chip {
        display: inline-flex; align-items: center; gap: 4px;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.72rem;
        color: #7dd3fc;
    }
    .src-chip::before { content: "📄"; font-size: 0.7rem; }

    /* ─── Status Dot ─── */
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
    .status-dot {
        display: inline-block; width: 7px; height: 7px;
        background: #22c55e; border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 6px;
    }

    /* ─── Misc ─── */
    .stAlert { border-radius: 10px !important; border: none !important; }
    button[kind="secondary"] {
        border-color: rgba(124,58,237,0.3) !important;
        color: #c4b5fd !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(124,58,237,0.1) !important;
        border-color: #7c3aed !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Header ───
st.markdown('<h1 class="zd-header">Zyro Dynamics HR Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="zd-tagline"><span class="status-dot"></span>Powered by <span>Advanced RAG</span> · Ask anything about company HR policies</p>', unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo">🚀</div>
        <div class="name">Zyro Dynamics</div>
        <div class="sub">HR Intelligence Platform</div>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)

    groq_key = st.text_input("🔑 API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""), placeholder="Enter Groq API Key")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Covered Topics**", help="Questions the assistant can answer")
    topics = ["Leave Policy", "Salary & CTC", "Health Insurance", "WFH Policy", "Performance Review", "Code of Conduct", "IT Security", "Travel & Expense"]
    chips_html = "".join([f'<span class="topic-chip">{t}</span>' for t in topics])
    st.markdown(f'<div>{chips_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div style="position:fixed;bottom:16px;left:16px;font-size:0.65rem;color:#475569;">v2.0 · Built with Streamlit & LangChain</div>', unsafe_allow_html=True)

# ─── Session State ───
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Welcome! I'm your Zyro Dynamics HR policy assistant. Ask me about leave policies, insurance, compensation, WFH rules, or any other HR topic."}]

REFUSAL_MESSAGE = (
    "I can only answer questions about Zyro Dynamics HR policies "
    "from the provided documents."
)

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

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1, max_tokens=1024, groq_api_key=api_key)

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are the official HR policy assistant. Answer ONLY using the HR policy context provided.\n"
         "IMPORTANT: Documents may mention 'Zyro Dynamics' or 'Acrux Dynamics' — treat them as the SAME company.\n"
         "Rules:\n"
         "1. Answer ONLY using information explicitly present in the context.\n"
         "2. Include exact numbers, dates, percentages, durations, and ALL eligibility conditions (e.g., minimum days worked) exactly as they appear. NEVER omit conditions or caveats.\n"
         "3. If context has PARTIAL information, give that information directly — no refusal preamble.\n"
         "4. NEVER hallucinate. NEVER ask which company — just answer.\n"
         "5. CRITICAL — Leave types: Each leave type (Earned, Sick, Maternity, etc.) has DIFFERENT rules. Answer ONLY for the specific leave type asked. NEVER mix rules between leave types.\n"
         "6. CRITICAL — Insurance types: If asked about 'health insurance' or 'medical insurance', answer ONLY about Group Medical Insurance. Ensure you mention the coverage amount of Rs. 5,00,000 per year, and that it covers employee + spouse + up to 2 dependent children, as stated in the policy documents.\n"
         "7. CRITICAL — Complete lists and timelines: Include EVERY item in the context. NEVER give a partial list. If context has a 7-row APR table, give all 7 rows. If context has 4 WFH types, give all 4 types. Do NOT include internal process-ownership columns (e.g., 'Owner', 'Department') unless the question specifically asks who is responsible.\n"
         "8. No repetition: state each fact ONCE; do not restate the conclusion again at the end.\n"
         "9. CRITICAL — Style: Write in a DIRECT, FACTUAL, policy-document tone — like a sentence "
         "taken straight from an HR policy manual. Do NOT use conversational framing such as "
         "'At Acrux Dynamics,' 'According to the policy,' or repeating the company name. "
         "Do not add extra context the question did not ask for. State the fact(s) plainly "
         "and concisely, as the source document itself would state them."),
        ("human", "HR POLICY CONTEXT:\n{context}\n\nEMPLOYEE QUESTION:\n{question}")
    ])

    OOS_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a classifier for an HR chatbot. Determine whether the employee question "
         "can be answered using internal HR policy documents.\n\n"
         "CRITICAL: The company is sometimes called 'Acrux Dynamics' and sometimes 'Zyro Dynamics'. "
         "They are the SAME company. Do NOT mark a question OUT_OF_SCOPE just because it mentions 'Acrux Dynamics'.\n\n"
         "Reply ONLY with one word:\n"
         "- IN_SCOPE → if the question is about HR policies, leave, salary, compensation bands, "
         "performance, insurance, WFH, onboarding, separation, travel, conduct, IT security, etc.\n"
         "- OUT_OF_SCOPE → if the question asks about company revenue, financials, product features, "
         "competitor comparisons, ESOP/stock options, job applications, or recruitment process."),
        ("human", "{question}")
    ])

    return ret, llm, RAG_PROMPT, OOS_PROMPT

def format_docs(docs):
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source_name = doc.metadata.get("source", "HR Policy").split("/")[-1]
        formatted_parts.append(
            f"--- Source: {source_name} ---\n{doc.page_content}"
        )
    return "\n\n".join(formatted_parts)

def _invoke_with_retry(chain, inputs, max_retries=5):
    """Retry wrapper that handles Groq rate limits automatically."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                wait_time = 15 * (attempt + 1)
                print(f"    ⏳ Rate limit hit, waiting {wait_time}s... (retry {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded due to rate limiting.")

def ask(question, ret, llm, prompt, oos_prompt):
    oos_chain = oos_prompt | llm | StrOutputParser()
    verdict = _invoke_with_retry(oos_chain, {"question": question}).strip().upper()
    time.sleep(10)
    
    if "OUT" in verdict:
        return REFUSAL_MESSAGE, []
    
    docs = ret.invoke(question)
    context = format_docs(docs)
    chain = prompt | llm | StrOutputParser()
    answer = _invoke_with_retry(chain, {"context": context, "question": question})
    time.sleep(10)
    
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
