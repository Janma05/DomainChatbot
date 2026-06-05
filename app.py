import os
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Domain Knowledge Assistant", page_icon="🤖")
st.title("📚 Domain-Specific Knowledge Assistant")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

    uploaded_files = st.file_uploader(
        "📤 Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

# -----------------------------
# Load Documents
# -----------------------------
def load_documents():
    documents = []

    if os.path.exists("documents"):
        for file in os.listdir("documents"):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(f"documents/{file}")
                documents.extend(loader.load())

    if uploaded_files:
        os.makedirs("documents", exist_ok=True)
        for file in uploaded_files:
            path = os.path.join("documents", file.name)
            with open(path, "wb") as f:
                f.write(file.read())

            loader = PyPDFLoader(path)
            documents.extend(loader.load())

    return documents

# -----------------------------
# Vector Store
# -----------------------------
@st.cache_resource
def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    return FAISS.from_documents(split_docs, embeddings)

# -----------------------------
# Prompt
# -----------------------------
def get_prompt():
    template = """
You are a strict Domain Knowledge Assistant.

Rules:
- Answer ONLY from the given context.
- If answer is NOT present, say EXACTLY:
"Sorry, that information is not in my database."
- Do NOT guess.

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# -----------------------------
# QA Chain
# -----------------------------
def get_qa_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # ✅ FIXED
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": get_prompt()},
        return_source_documents=True
    )

    return qa

# -----------------------------
# Chat Memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Load + Build
# -----------------------------
docs = load_documents()

if not docs:
    st.warning("⚠️ Upload PDFs first")
    st.stop()

vector_store = create_vector_store(docs)
qa_chain = get_qa_chain(vector_store)

# -----------------------------
# Display Chat
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Chat Input
# -----------------------------
if query := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            result = qa_chain.invoke({"query": query})

            answer = result.get("result", "").strip()
            sources = result.get("source_documents", [])

            # Hallucination control
            if not sources:
                answer = "Sorry, that information is not in my database."

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})