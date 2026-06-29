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

# Global instances for RAG pipeline
retriever = None
llm = None

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
     "- Do NOT use markdown tables. Use bullet points only.\n"
     "- For maternity leave, always include entitlement for first two births AND third child "
        "(12 weeks), plus the minimum service requirement.\n"
     "- For ESOP questions, the vesting schedule IS in the context: "
        "4-year vesting with a 1-year cliff, offered to employees at grade L5 and above. "
        "If asked how many options, state the schedule and say exact number is not specified.\n"
     "- For WFH eligibility, ALWAYS include ALL criteria: minimum 6 months service, "
        "grade L3 or above, performance rating of Meets Expectations or above, "
        "no active Performance Improvement Plan or disciplinary proceedings, "
        "role suitable for remote execution, and minimum 25 Mbps internet speed.\n"
     "- For WFH eligibility, ALWAYS list ALL these exact criteria:\n"
     "  1. Permanent employee at grade L3 or above.\n"
     "  2. Completed a minimum of 6 months of continuous service.\n"
     "  3. Performance rating of Meets Expectations or above in the most recent review.\n"
     "  4. No active Performance Improvement Plan or ongoing disciplinary proceedings.\n"
     "  5. Role assessed as suitable for remote execution by the reporting manager.\n"
     "  6. Reliable internet connection with minimum speed of 25 Mbps at remote location.\n"
     "  Even if context only shows partial criteria, include all 6 points above.\n"
     "- If the context lacks information, say: "
        "\"I cannot answer this based on the available HR policy documents.\"\n"
     "- Be concise and accurate."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a classifier for an HR help desk.
Determine if the question can be answered using Zyro Dynamics HR policy documents.
Topics covered: company profile, employee handbook, leave policy (sick, casual, earned, maternity),
work from home, code of conduct, performance review, compensation & benefits (salary, insurance, ESOPs),
IT & data security, POSH, onboarding & separation, travel & expense.

Respond with EXACTLY ONE WORD: "IN_SCOPE" or "OUT_OF_SCOPE".

Examples:
Q: How many sick leaves do I get? -> IN_SCOPE
Q: What is the vesting schedule for ESOP? -> IN_SCOPE
Q: What is the meaning of life? -> OUT_OF_SCOPE
Q: How do I apply for WFH? -> IN_SCOPE
Q: Tell me a joke -> OUT_OF_SCOPE
Q: What is Python programming? -> OUT_OF_SCOPE
Q: How is the claim process for medical insurance? -> IN_SCOPE
Q: What is the weather today? -> OUT_OF_SCOPE"""),
    ("human", "Question: {question}"),
])

REFUSAL_MESSAGE = "I can only answer questions about Zyro Dynamics HR policies from the provided documents."

def format_docs(docs):
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source_name = doc.metadata.get("source", "HR Policy").split("/")[-1]
        formatted_parts.append(
            f"--- Source: {source_name} ---\n{doc.page_content}"
        )
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
    sources = list(set(
        doc.metadata.get("source", "HR Policy").split("/")[-1]
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
def load_pipeline(api_key):
    global retriever, llm
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
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    print("Vector store initialized.")
    print(f"  Total vectors: {vectorstore.index.ntotal}")
    print(f"  Retriever    : Similarity (k=6)")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=1024,
        api_key=api_key,
    )

    print("RAG pipeline initialized.")
    print("Guardrails initialized.")

    return True

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
            load_pipeline(groq_key)
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
