import streamlit as st
from google import genai


st.set_page_config(
    page_title="AI Coding Assistant",
    page_icon="🤖"
)

st.title("🤖 AI Coding Assistant")
st.write("Generate Python code using Gemini AI")


# Get API key from Streamlit Secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("❌ GEMINI_API_KEY is missing.")
    st.stop()


# Gemini client
client = genai.Client(
    api_key=api_key
)


question = st.text_area(
    "Enter your coding requirement:",
    placeholder="Example: Create a Python function to check whether a number is prime.",
    height=150
)


if st.button("🚀 Generate Code"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        prompt = f"""
You are an expert Python developer.

User requirement:

{question}

Provide:

1. Python Code
2. Explanation
3. Example Usage
4. Time Complexity
"""

        try:

            with st.spinner("🤖 Gemini is generating..."):

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )

            st.subheader("🤖 AI Response")
            st.markdown(response.text)

        except Exception as e:

            st.error("❌ Gemini Error")
            st.code(str(e))
