import os
import requests
import numpy as np
import streamlit as st


from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
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
        "Go to Streamlit → Manage app → Settings → Secrets "
        "and add GEMINI_API_KEY."
    )
    st.stop()


def ask_gemini(prompt):

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-3.1-flash-lite:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=60
        )

        if response.status_code != 200:

            return (
                f"❌ Gemini API Error "
                f"({response.status_code}):\n\n"
                f"{response.text}"
            )

        result = response.json()

        return result["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:

        return f"❌ Error: {str(e)}"


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(
        uploaded_file
    )

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        chunk = text[
            start:start + chunk_size
        ].strip()

        if chunk:

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# VECTOR DATABASE
# =========================================================

def create_vector_database(
    chunks,
    model
):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# =========================================================
# SEARCH
# =========================================================

def search_document(
    question,
    chunks,
    embeddings,
    model,
    k=3
):

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[-k:][::-1]

    return [
        chunks[i]
        for i in top_indices
    ]


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AI Coding Assistant")

st.write(
    "Generative AI Coding Assistant using "
    "Gemini + RAG + Streamlit"
)


# =========================================================
# SIDEBAR
# =========================================================

mode = st.sidebar.selectbox(
    "Choose Mode",
    [
        "💻 Generate Code",
        "🐞 Debug Code",
        "📄 Ask PDF"
    ]
)


# =========================================================
# CODE GENERATOR
# =========================================================

if mode == "💻 Generate Code":

    st.header("💻 Python Code Generator")

    request = st.text_area(
        "Describe what you want to build",
        placeholder=(
            "Create a Python function to check "
            "whether a number is prime."
        ),
        height=150
    )

    if st.button(
        "🚀 Generate Code",
        use_container_width=True
    ):

        if not request.strip():

            st.warning(
                "Please enter a requirement."
            )

        else:

            prompt = f"""
You are an expert Python developer.

Generate clean and correct Python code.

Requirement:
{request}

Provide:

1. Python code
2. Explanation
3. Example usage
4. Time complexity
"""

            with st.spinner(
                "🤖 Generating..."
            ):

                answer = ask_gemini(
                    prompt
                )

            st.subheader(
                "🤖 AI Response"
            )

            st.markdown(answer)


# =========================================================
# DEBUGGER
# =========================================================

elif mode == "🐞 Debug Code":

    st.header("🐞 Python Code Debugger")

    code = st.text_area(
        "Paste your Python code",
        height=300
    )

    if st.button(
        "🔍 Debug Code",
        use_container_width=True
    ):

        if not code.strip():

            st.warning(
                "Please paste Python code."
            )

        else:

            prompt = f"""
You are an expert Python debugger.

Analyze this code:

{code}

Provide:

1. Error
2. Reason
3. Corrected code
4. Explanation
5. Improvements
"""

            with st.spinner(
                "🔍 Debugging..."
            ):

                answer = ask_gemini(
                    prompt
                )

            st.subheader(
                "🐞 Debug Result"
            )

            st.markdown(answer)


# =========================================================
# PDF RAG
# =========================================================

else:

    st.header(
        "📄 Chat with Your PDF"
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        with st.spinner(
            "📚 Processing PDF..."
        ):

            text = extract_pdf_text(
                uploaded_file
            )

            chunks = create_chunks(
                text
            )

            model = load_embedding_model()

            embeddings = create_vector_database(
                chunks,
                model
            )

        st.success(
            f"✅ PDF processed — "
            f"{len(chunks)} chunks created."
        )

        question = st.text_input(
            "Ask a question about the PDF"
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
                    "🔎 Searching..."
                ):

                    results = search_document(
                        question,
                        chunks,
                        embeddings,
                        model
                    )

                context = "\n\n".join(
                    results
                )

                prompt = f"""
You are an AI document assistant.

Use ONLY the following document context
to answer the question.

DOCUMENT CONTEXT:

{context}

QUESTION:

{question}

If the answer is not present in the
document, say:

"The answer is not available in
the uploaded document."
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
                    "📚 Retrieved Context"
                ):

                    for i, result in enumerate(
                        results,
                        1
                    ):

                        st.markdown(
                            f"**Context {i}**"
                        )

                        st.write(result)

                        st.divider()
