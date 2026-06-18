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
from langchain_core.runnables import RunnablePassthrough
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

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

retriever = None
llm = None

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are ZyroHR, the official HR Help Desk assistant for Zyro Dynamics Pvt. Ltd. "
     "IMPORTANT: Acrux Dynamics and Zyro Dynamics are the SAME company. "
     "Answer employee questions using ONLY the provided HR policy context.\n\n"
     "CRITICAL RULES:\n"
     "- Be highly concise and strictly factual. Answer in exactly 1 to 4 sentences.\n"
     "- Answer ONLY what is explicitly asked. If the question asks about a specific grade (e.g., L4), provide data ONLY for that grade. Do NOT list other grades.\n"
     "- Include exact numbers, percentages, and conditions directly from the text.\n"
     "- Always cite the exact document name and page number naturally in your answer (e.g., 'as stated in Leave_Policy.pdf on Page 2').\n"
     "- Write your answer in a SINGLE plain-text paragraph. Do NOT use bullet points (-), bold text (**), or markdown.\n"
     "- If the question asks about Health Insurance, do NOT mention Term Life or Personal Accident Insurance.\n"
     "- TRAP RULE: ONLY use the exact refusal message ('I can only answer questions related to Zyro Dynamics HR policies. Your question is outside my scope. Please contact the relevant department directly.') if the question is completely unanswerable. NEVER append it to a partial answer.\n"),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query classifier for the Zyro Dynamics (Acrux Dynamics) HR Help Desk.\n"
     "Classify the question as HR-RELATED or OUT-OF-SCOPE.\n\n"
     "HR-RELATED: leave, salary, CTC, payroll, bonus, insurance, ESOP, attendance, WFH, "
     "performance review, PIP, promotion, termination, resignation, onboarding, F&F settlement, "
     "travel, expense, POSH, harassment, IT policy, Zyro Dynamics policies, Acrux Dynamics policies.\n\n"
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

def _invoke_with_retry(chain, input_data, max_retries=5):
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                wait_time = 15 * (attempt + 1)
                print(f"    Rate limit hit, waiting {wait_time}s... (retry {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded.")

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
    
    # Extract sources from metadata directly, matching the older logic
    sources = list(set(
        doc.metadata.get("source", "HR Policy").split("/")[-1].split("\\")[-1]
        for doc in retrieved_docs
    ))
            
    return {
        "answer": answer.strip(),
        "sources": sources,
        "retrieved_docs": retrieved_docs
    }

@traceable(name="ask_bot")
def ask_bot(question: str) -> dict:
    classifier_chain = OOS_PROMPT | llm | StrOutputParser()
    verdict = _invoke_with_retry(classifier_chain, {"question": question}).strip().upper()

    time.sleep(10)  

    if "OUT" in verdict:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "blocked": True}

    result = rag_chain(question)
    result["blocked"] = False

    time.sleep(10)   

    return result

@st.cache_resource
def load_pipeline_v2(api_key):
    corpus_path = os.environ.get("CORPUS_PATH", os.path.join(os.path.dirname(__file__), "hr_docs"))
    if not os.path.isdir(corpus_path):
        corpus_path = "/kaggle/input/zyro-dynamics-hr-corpus/"

    loader = PyPDFDirectoryLoader(corpus_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""],
        is_separator_regex=False
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 40]

    print(f"Created {len(chunks)} chunks")
    if chunks:
        print(f"  Avg size : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64
        }
    )

    _test_vec = embeddings.embed_query("test")
    print(f"Embedding model initialized.")
    print(f"  Dimensions: {len(_test_vec)}")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 40,
            "lambda_mult": 0.6
        }
    )
    print("Vector store initialized.")
    print(f"  Total vectors: {vectorstore.index.ntotal}")
    print(f"  Retriever    : MMR (k=8, fetch_k=40, lambda_mult=0.6)")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=512,
        api_key=api_key,
    )

    print("RAG pipeline initialized.")
    print("Guardrails initialized.")

    return retriever, llm

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the Zyro Dynamics HR Assistant. How can I help you today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 View Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s}**")

if prompt := st.chat_input("Ask your HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not groq_key:
            st.error("Please enter your Groq API Key in the sidebar.")
            st.stop()

        with st.spinner("Searching HR policies..."):
            retriever, llm = load_pipeline_v2(groq_key)
            result = ask_bot(prompt)
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            st.markdown(answer)
            if sources:
                with st.expander("📄 View Sources"):
                    for s in sources:
                        st.markdown(f"- **{s}**")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
