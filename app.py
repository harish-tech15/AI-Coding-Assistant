import os
import numpy as np
import streamlit as st

from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Coding Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# GEMINI API
# =========================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY is not configured.")
    st.info(
        "Add GEMINI_API_KEY in Streamlit Cloud → "
        "Manage app → Settings → Secrets."
    )
    st.stop()

client = genai.Client(
    api_key=API_KEY
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# GEMINI FUNCTION
# =========================================================

def ask_gemini(prompt):

    try:

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"❌ Gemini API Error: {str(e)}"


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    try:

        reader = PdfReader(
            uploaded_file
        )

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

        return text

    except Exception as e:

        return f"PDF extraction error: {str(e)}"


# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    if not text.strip():

        return chunks

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def create_vector_database(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    return embeddings


# =========================================================
# DOCUMENT SEARCH
# =========================================================

def search_document(
    question,
    chunks,
    embeddings,
    k=3
):

    if not chunks:

        return []

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    k = min(
        k,
        len(chunks)
    )

    top_indices = np.argsort(
        similarities
    )[-k:][::-1]

    results = []

    for index in top_indices:

        results.append(
            chunks[index]
        )

    return results


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AI Coding Assistant")

st.write(
    "A Generative AI application built with "
    "Google Gemini, RAG, Sentence Transformers "
    "and Streamlit."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛠️ AI Tools")

mode = st.sidebar.selectbox(
    "Choose Mode",
    [
        "💻 Generate Code",
        "🐞 Debug Code",
        "📄 Ask PDF"
    ]
)


# =========================================================
# MODE 1 — CODE GENERATION
# =========================================================

if mode == "💻 Generate Code":

    st.header("💻 Python Code Generator")

    request = st.text_area(
        "What code do you want?",
        placeholder=(
            "Example: Create a Python function "
            "to check whether a number is prime."
        ),
        height=150
    )

    if st.button(
        "🚀 Generate Code",
        use_container_width=True
    ):

        if not request.strip():

            st.warning(
                "Please enter your requirement."
            )

        else:

            prompt = f"""
You are an expert Python developer.

Generate clean, correct and beginner-friendly
Python code.

USER REQUIREMENT:
{request}

Your response must contain:

1. Python Code
2. Explanation
3. Example Usage
4. Time Complexity
"""

            with st.spinner(
                "🤖 Generating code..."
            ):

                answer = ask_gemini(
                    prompt
                )

            st.subheader(
                "🤖 AI Response"
            )

            st.markdown(answer)


# =========================================================
# MODE 2 — CODE DEBUGGING
# =========================================================

elif mode == "🐞 Debug Code":

    st.header("🐞 Python Code Debugger")

    code = st.text_area(
        "Paste your Python code here:",
        placeholder="""numbers = [1, 2, 3]
print(numbers[5])""",
        height=300
    )

    if st.button(
        "🔍 Debug Code",
        use_container_width=True
    ):

        if not code.strip():

            st.warning(
                "Please paste your Python code."
            )

        else:

            prompt = f"""
You are an expert Python debugger.

Analyze the following Python code.

CODE:

{code}

Provide:

1. Identify the error
2. Explain why the error occurs
3. Corrected Python code
4. Explanation of the corrected code
5. Suggestions for improvement
"""

            with st.spinner(
                "🔍 Analyzing code..."
            ):

                answer = ask_gemini(
                    prompt
                )

            st.subheader(
                "🐞 Debug Result"
            )

            st.markdown(answer)


# =========================================================
# MODE 3 — PDF RAG
# =========================================================

elif mode == "📄 Ask PDF":

    st.header(
        "📄 Chat with Your PDF"
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    if uploaded_file:

        st.info(
            f"📄 File: {uploaded_file.name}"
        )

        if "pdf_name" not in st.session_state:

            st.session_state.pdf_name = None

        if (
            st.session_state.pdf_name
            != uploaded_file.name
        ):

            with st.spinner(
                "📚 Processing PDF..."
            ):

                text = extract_pdf_text(
                    uploaded_file
                )

                chunks = create_chunks(
                    text
                )

                if not chunks:

                    st.error(
                        "❌ Could not extract text from PDF."
                    )

                    st.stop()

                embeddings = create_vector_database(
                    chunks
                )

                st.session_state.pdf_chunks = chunks
                st.session_state.pdf_embeddings = embeddings
                st.session_state.pdf_name = uploaded_file.name

        else:

            chunks = st.session_state.pdf_chunks
            embeddings = st.session_state.pdf_embeddings

        st.success(
            f"✅ PDF processed successfully — "
            f"{len(chunks)} chunks created."
        )

        question = st.text_input(
            "Ask a question about the PDF:",
            placeholder=(
                "Example: What is this document about?"
            )
        )

        if st.button(
            "🔎 Ask AI",
            use_container_width=True
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "🔎 Searching document..."
                ):

                    relevant_chunks = search_document(
                        question,
                        chunks,
                        embeddings,
                        k=3
                    )

                context = "\n\n".join(
                    relevant_chunks
                )

                prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY
the provided document context.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Rules:

- Answer clearly and accurately.
- Do not invent information.
- If the answer is not present in the
  document, say:
  "The answer is not available in the uploaded document."
"""

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = ask_gemini(
                        prompt
                    )

                st.subheader(
                    "🤖 AI Answer"
                )

                st.markdown(answer)

                with st.expander(
                    "📚 View Retrieved Context"
                ):

                    for i, chunk in enumerate(
                        relevant_chunks,
                        1
                    ):

                        st.markdown(
                            f"**Context {i}**"
                        )

                        st.write(chunk)

                        st.divider()


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Built with Python + Gemini + RAG + Streamlit"
)
