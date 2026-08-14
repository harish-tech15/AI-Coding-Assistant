
import os
import numpy as np
import faiss
import streamlit as st

from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Coding Assistant",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Gemini
# -----------------------------

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Gemini API key is not configured.")
    st.stop()

client = genai.Client(
    api_key=API_KEY
)


# -----------------------------
# Embedding Model
# -----------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# -----------------------------
# PDF Text Extraction
# -----------------------------

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Chunking
# -----------------------------

def create_chunks(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += chunk_size - overlap

    return chunks


# -----------------------------
# FAISS
# -----------------------------

def create_faiss_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


def search_document(
    question,
    chunks,
    index,
    k=3
):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    k = min(k, index.ntotal)

    distances, indices = index.search(
        query_embedding,
        k
    )

    return [
        chunks[i]
        for i in indices[0]
        if i != -1
    ]


# -----------------------------
# Gemini Function
# -----------------------------

def ask_gemini(prompt):

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text


# -----------------------------
# UI
# -----------------------------

st.title("🤖 AI Coding Assistant")

st.write(
    "Gen AI Coding Assistant using "
    "Gemini + RAG + FAISS"
)


mode = st.sidebar.selectbox(
    "Choose Mode",
    [
        "💻 Generate Code",
        "🐞 Debug Code",
        "📄 Ask PDF"
    ]
)


# =================================================
# GENERATE CODE
# =================================================

if mode == "💻 Generate Code":

    st.header("💻 Generate Python Code")

    request = st.text_area(
        "Describe what you want to build"
    )

    if st.button("Generate Code"):

        if request.strip():

            prompt = f"""
You are an expert Python developer.

Generate clean and correct Python code.

USER REQUIREMENT:
{request}

Provide:
1. Python code
2. Explanation
3. Example usage
"""

            with st.spinner("Generating code..."):

                answer = ask_gemini(prompt)

            st.subheader("🤖 AI Response")

            st.write(answer)


# =================================================
# DEBUG CODE
# =================================================

elif mode == "🐞 Debug Code":

    st.header("🐞 Debug Python Code")

    code = st.text_area(
        "Paste your Python code",
        height=300
    )

    if st.button("Debug Code"):

        if code.strip():

            prompt = f"""
You are an expert Python debugger.

Analyze the following code.

CODE:
{code}

Provide:
1. Error identification
2. Explanation
3. Corrected code
4. Improvement suggestions
"""

            with st.spinner("Analyzing code..."):

                answer = ask_gemini(prompt)

            st.subheader("🤖 Debug Result")

            st.write(answer)


# =================================================
# ASK PDF
# =================================================

else:

    st.header("📄 Ask Questions About Your PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        with st.spinner(
            "Processing PDF..."
        ):

            text = extract_pdf_text(
                uploaded_file
            )

            chunks = create_chunks(
                text
            )

            index = create_faiss_index(
                chunks
            )

        st.success(
            f"PDF processed successfully! "
            f"{len(chunks)} chunks created."
        )

        question = st.text_input(
            "Ask something about the PDF"
        )

        if st.button("Ask AI"):

            if question.strip():

                results = search_document(
                    question,
                    chunks,
                    index
                )

                context = "\n\n".join(
                    results
                )

                prompt = f"""
You are an AI assistant.

Answer the question using ONLY
the provided PDF context.

PDF CONTEXT:
{context}

QUESTION:
{question}

If the answer is not available,
say that it is not available
in the uploaded document.
"""

                with st.spinner(
                    "Searching document..."
                ):

                    answer = ask_gemini(
                        prompt
                    )

                st.subheader(
                    "🤖 AI Answer"
                )

                st.write(answer)
