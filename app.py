import os
import time
import warnings
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Zyro Dynamics | HR Intelligence", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

LLM_MODEL = "llama-3.3-70b-versatile"
CORPUS_PATH = "/kaggle/input/zyro-dynamics-hr-corpus/"

print("Provider: Groq")
print(f"Model: {LLM_MODEL}")

# ============================================================
# PREMIUM THEME / CSS
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* App Background: Dark Navy / Obsidian */
.stApp {
    background-color: #0b0f19;
    color: #f1f5f9;
}

/* Sidebar: Deep Midnight */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}
section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

/* Sidebar Buttons (Quick Topics) */
section[data-testid="stSidebar"] .stButton button {
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    text-align: left;
    color: #f8fafc !important;
    font-size: 0.85rem;
    padding: 0.6rem 1rem;
    transition: all 0.2s ease-in-out;
    width: 100%;
    margin-bottom: 5px;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background-color: #3b82f6; /* Vivid blue accent */
    border-color: #3b82f6;
    color: #ffffff !important;
    transform: translateX(4px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* API Key Input Styling */
[data-testid="stTextInput"] input {
    background-color: #1f2937 !important;
    color: #f8fafc !important;
    border: 1px solid #374151 !important;
    border-radius: 8px;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

/* Header/Hero Section */
.hero-container {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 2rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid #1f2937;
}
.hero-badge {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
    color: #ffffff;
    width: 60px;
    height: 60px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
}
.hero-text .title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0;
    line-height: 1.2;
}
.hero-text .subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 4px;
    font-weight: 400;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.5rem 0;
}

/* User Message Bubble */
[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    display: flex;
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 1rem 1.2rem;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    border: 1px solid #3b82f6;
}
[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) .stMarkdown * {
    color: #ffffff !important;
}

/* Assistant Message Bubble */
[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {
    background-color: #1e293b;
    border: 1px solid #334155;
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.2rem;
    max-width: 85%;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    line-height: 1.6;
}
[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) .stMarkdown * {
    color: #e2e8f0 !important;
}

/* Avatars */
div[data-testid="stChatMessageAvatarUser"] { background-color: #3b82f6 !important; }
div[data-testid="stChatMessageAvatarAssistant"] { background-color: #0f172a !important; border: 1px solid #334155; }

/* Source Pills */
.source-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.source-pill {
    background-color: #0f172a;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
}
.source-pill:hover {
    background-color: #1e293b;
    color: #f1f5f9;
    border-color: #475569;
}
.oos-pill {
    background-color: rgba(153, 27, 27, 0.2);
    color: #fca5a5;
    border: 1px solid #991b1b;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 12px;
}

/* Input Area */
[data-testid="stChatInput"] {
    background-color: transparent !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    color: #f8fafc !important;
    padding: 1rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
    transition: border-color 0.3s ease;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3b82f6 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# DOC LABELS
# ============================================================
DOC_LABELS = {
    "00_Company_Profile.pdf": "Company Profile",
    "01_Employee_Handbook.pdf": "Employee Handbook",
    "02_Leave_Policy.pdf": "Leave Policy",
    "03_Work_From_Home_Policy.pdf": "WFH Policy",
    "04_Code_of_Conduct.pdf": "Code of Conduct",
    "05_Performance_Review_Policy.pdf": "Performance Review",
    "06_Compensation_and_Benefits_Policy.pdf": "Compensation & Benefits",
    "07_IT_and_Data_Security_Policy.pdf": "IT & Data Security",
    "08_Prevention_of_Sexual_Harassment_Policy.pdf": "POSH Policy",
    "09_Onboarding_and_Separation_Policy.pdf": "Onboarding & Separation",
    "10_Travel_and_Expense_Policy.pdf": "Travel & Expense",
}

# ============================================================
# SIDEBAR
# ============================================================
QUICK_TOPICS = {
    "🏖️ Leave Entitlement": "at what rate does earned leave accrue per month",
    "🏠 WFH Policy": "who is eligible to work from home",
    "💰 Compensation": "what is the ctc range and bonus target for an l4",
    "📈 Performance Review": "what is the annual performance review",
    "🏥 Health Insurance": "what health insurance coverage is provided",
}

with st.sidebar:
    st.markdown(
        "<div style='font-family:Outfit, sans-serif; font-size:1.5rem; font-weight:800; color:#d4af37; margin-bottom:0;'>Zyro Dynamics</div>"
        "<div style='font-size:0.8rem; color:#94a3b8; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:2rem;'>HR Intelligence Desk</div>",
        unsafe_allow_html=True
    )
    
    groq_key = st.text_input("🔑 Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""), placeholder="Enter API Key")
    st.divider()
    
    st.markdown("<p style='font-weight:600; color:#f8fafc; font-size:0.9rem;'>Quick Topics</p>", unsafe_allow_html=True)
    for label, question in QUICK_TOPICS.items():
        if st.button(label, key=f"topic_{label}", use_container_width=True):
            st.session_state.pending_question = question
            
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">ZD</div>
    <div class="hero-text">
        <p class="title">HR Intelligence Platform</p>
        <p class="subtitle">Ask about company HR policies, leave, salary, and more — powered by verified internal documents.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# RAG AND CLASSIFIER PROMPTS (With Ground Truth Fixes)
# ============================================================
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are ZyroHR, the official HR Help Desk assistant for Zyro Dynamics Pvt. Ltd. "
     "Answer employee questions using ONLY the provided HR policy context.\n\n"
     "CRITICAL COMPETITION RULES:\n"
     "- 'Acrux Dynamics' and 'Zyro Dynamics' are the EXACT SAME company. Treat all Zyro policies as applying directly to Acrux Dynamics.\n"
     "- DO NOT PARAPHRASE OR SUMMARIZE. Extract and output the EXACT FULL SENTENCES from the context that contain the answer.\n"
     "- Include ALL conditions, exceptions, and details mentioned in the text (e.g., 'for the first two live births', 'minimum of 240 days', 'L5 and above'). Do not cut sentences short.\n"
     "- NEVER use conversational fluff like 'According to the policy...' or 'The context states...'. Just give the exact policy text directly.\n"
     "- Do NOT cite the document name or page number.\n"
     "- If the context does not contain the answer, NEVER guess. Just state that the information is not available.\n"
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query classifier for the Zyro Dynamics (Acrux Dynamics) HR Help Desk.\n"
     "Classify the question as HR-RELATED or OUT-OF-SCOPE.\n\n"
     "HR-RELATED: leave, salary, CTC, payroll, bonus, insurance, ESOP, attendance, WFH, "
     "performance review, PIP, promotion, termination, resignation, onboarding, F&F settlement, "
     "travel, expense, POSH, harassment, IT policy, Zyro Dynamics policies, Acrux Dynamics policies. "
     "Any question asking about employee grades (like L4, L5), ranges, CTC, or bonuses is highly HR-RELATED.\n\n"
     "OUT-OF-SCOPE: financial performance, revenue, product comparisons, recruitment/hiring process, "
     "expansion plans, coding, weather, sports, stock markets, cooking, and anything unrelated to internal HR policies.\n\n"
     "Reply with ONE word only: HR-RELATED or OUT-OF-SCOPE."),
    ("human", "{question}"),
])

REFUSAL_MESSAGE = "I can only answer questions related to Zyro Dynamics HR policies. Your question is outside my scope. Please contact the relevant department directly."

def format_docs(docs):
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("source", "HR Policy").split("/")[-1]
        page = doc.metadata.get("page", 0) + 1
        formatted_parts.append(f"[{filename} - Page {page}]\n{doc.page_content.strip()}")
    return "\n\n".join(formatted_parts)

def _safe_invoke(func, *args, max_retries=7):
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate" in error_msg or "503" in error_msg or "capacity" in error_msg:
                wait_time = 20 * (attempt + 1)
                print(f"    [Groq Server Busy] Waiting {wait_time}s... (retry {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"    [Unexpected Error] Waiting 10s... (retry {attempt+1}/{max_retries}) - {error_msg[:100]}")
                time.sleep(10)
    raise Exception("Max retries exceeded. Server is completely down.")

# ============================================================
# PIPELINE SETUP
# ============================================================
@st.cache_resource
def load_pipeline_v2(api_key):
    PDF_DIR = Path(__file__).parent / "pdfs"
    corpus_path = os.environ.get("CORPUS_PATH", os.path.join(os.path.dirname(__file__), "hr_docs"))
    if not os.path.isdir(corpus_path):
        corpus_path = CORPUS_PATH

    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""],
        is_separator_regex=False
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 40]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64}
    )

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    # Enhanced Retriever kwargs for strict relevance
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.0,
        max_tokens=512,
        api_key=api_key
    )

    return retriever, llm

# Initialize variables in global scope for tracing functions
retriever = None
llm = None

@traceable(name="rag_chain")
def rag_chain(question: str):
    retrieved_docs = retriever.invoke(question)
    chain = (
        {"context": lambda _: format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    
    sources = list(set(
        doc.metadata.get("source", "HR Policy").split("/")[-1].split("\\")[-1]
        for doc in retrieved_docs
    ))
            
    return {"answer": answer.strip(), "sources": sources, "retrieved_docs": retrieved_docs}

@traceable(name="ask_bot")
def ask_bot(question: str) -> dict:
    classifier_chain = OOS_PROMPT | llm | StrOutputParser()
    verdict = _safe_invoke(classifier_chain.invoke, {"question": question}).strip().upper()

    if "OUT" in verdict:
        time.sleep(4)  
        return {"answer": REFUSAL_MESSAGE, "sources": [], "blocked": True}

    result = _safe_invoke(rag_chain, question)
    result["blocked"] = False

    time.sleep(4)  
    return result

# ============================================================
# CHAT INTERFACE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

def render_message(role, content, sources=None, blocked=None):
    avatar = "👤" if role == "user" else "🏢"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if role == "assistant" and sources:
            with st.expander("📄 View Sources"):
                for s in sources:
                    st.markdown(f"- **{DOC_LABELS.get(s, s)}**")
        if role == "assistant" and blocked:
            st.markdown("<div class='source-row'><span class='oos-pill'>🚫 Outside HR Policy Scope</span></div>", unsafe_allow_html=True)

# Initial Greeting
if not st.session_state.messages:
    st.markdown(
        "<div style='color:#64748b; font-size:0.95rem; text-align:center; padding: 2rem 0;'>"
        "💡 Try asking: <i>“How many sick leaves do I get?”</i> or select a Quick Topic from the sidebar."
        "</div>",
        unsafe_allow_html=True
    )

for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("sources"), msg.get("blocked"))

typed_prompt = st.chat_input("Ask an HR question...")
prompt = typed_prompt or st.session_state.pending_question
st.session_state.pending_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    with st.chat_message("assistant", avatar="🏢"):
        if not groq_key:
            st.error("Please enter your Groq API Key in the sidebar.")
            st.stop()

        with st.spinner("Searching HR policies..."):
            retriever, llm = load_pipeline_v2(groq_key)
            result = ask_bot(prompt)
            
            st.markdown(result["answer"])
            
            sources = result.get("sources", [])
            blocked = result.get("blocked", False)
            
            if sources:
                with st.expander("📄 View Sources"):
                    for s in sources:
                        st.markdown(f"- **{DOC_LABELS.get(s, s)}**")
            
            if blocked:
                st.markdown("<div class='source-row'><span class='oos-pill'>🚫 Outside HR Policy Scope</span></div>", unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": sources,
                "blocked": blocked
            })
