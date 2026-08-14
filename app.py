import os
import requests
import streamlit as st


st.set_page_config(
    page_title="AI Coding Assistant",
    page_icon="🤖",
    layout="wide"
)


API_KEY = os.getenv("GEMINI_API_KEY")


if not API_KEY:
    st.error("❌ GEMINI_API_KEY is missing.")
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

    payload = {
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
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            return (
                f"❌ Gemini API Error: "
                f"{response.status_code}\n\n"
                f"{response.text}"
            )

        data = response.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:

        return f"❌ Error: {str(e)}"


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🤖 AI Coding Assistant")

st.write(
    "Generate and debug Python code using Generative AI."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🛠️ AI Tools")

mode = st.sidebar.radio(
    "Select a tool:",
    [
        "💻 Code Generator",
        "🐞 Code Debugger"
    ]
)


# ---------------------------------------------------------
# CODE GENERATOR
# ---------------------------------------------------------

if mode == "💻 Code Generator":

    st.header("💻 Python Code Generator")

    requirement = st.text_area(
        "Enter your coding requirement:",
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

        if not requirement.strip():

            st.warning(
                "⚠️ Please enter your requirement."
            )

        else:

            prompt = f"""
You are an expert Python developer.

Generate clean, beginner-friendly and
correct Python code.

User requirement:

{requirement}

Give the answer in this format:

1. Python Code
2. Explanation
3. Example Usage
4. Time Complexity
"""

            with st.spinner(
                "🤖 Generating code..."
            ):

                answer = ask_gemini(prompt)

            st.subheader(
                "🤖 AI Response"
            )

            st.markdown(answer)


# ---------------------------------------------------------
# CODE DEBUGGER
# ---------------------------------------------------------

else:

    st.header("🐞 Python Code Debugger")

    code = st.text_area(
        "Paste your Python code:",
        placeholder=(
            "numbers = [1, 2, 3]\n"
            "print(numbers[5])"
        ),
        height=300
    )

    if st.button(
        "🔍 Debug Code",
        use_container_width=True
    ):

        if not code.strip():

            st.warning(
                "⚠️ Please paste your Python code."
            )

        else:

            prompt = f"""
You are an expert Python debugger.

Analyze this Python code:

{code}

Provide:

1. Identify the error
2. Explain why it happens
3. Corrected Python code
4. Explanation of the correction
5. Suggestions for improvement
"""

            with st.spinner(
                "🔍 Analyzing code..."
            ):

                answer = ask_gemini(prompt)

            st.subheader(
                "🐞 Debug Result"
            )

            st.markdown(answer)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "Built with Python + Gemini + Streamlit 🚀"
)
