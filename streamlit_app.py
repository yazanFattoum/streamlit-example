import streamlit as st
from huggingface_hub import InferenceClient
import os

# ---------------------------
# Config
# ---------------------------
DEFAULT_MODELS = [
    "bigcode/starcoder2-3b",
    "bigcode/starcoder2-15b",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "codellama/CodeLlama-7b-Instruct-hf"
]

def get_token():
    try:
        return st.secrets["HF_API_TOKEN"]
    except KeyError:
        return os.getenv("HF_API_TOKEN")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Code Completion", page_icon="💻", layout="wide")
st.title("💻 Hugging Face Code Completion")

token = get_token()
if not token:
    st.warning("Please set HF_API_TOKEN in Streamlit secrets or environment.")

model = st.selectbox("Choose a model", DEFAULT_MODELS)
prompt = st.text_area("Enter your code snippet:", height=200, placeholder="def fibonacci(n):\n    ")

col1, col2, col3 = st.columns(3)
with col1:
    n = st.number_input("Number of completions", 1, 5, 3)
with col2:
    max_tokens = st.slider("Max new tokens", 16, 512, 128, step=16)
with col3:
    temperature = st.slider("Temperature", 0.0, 1.5, 0.4, step=0.05)

if st.button("Generate"):
    if not prompt.strip():
        st.error("Please enter some code.")
    elif not token:
        st.error("Missing Hugging Face token.")
    else:
        try:
            client = InferenceClient(provider="auto", api_key=token)
            st.info("Generating completions...")
            completions = []
            for _ in range(n):
                text = client.text_generation(
                    prompt,
                    model=model,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                completions.append(text.strip())

            st.subheader("Completions")
            lang = "python" if "def " in prompt else "java"
            for i, comp in enumerate(completions, 1):
                st.markdown(f"**Completion {i}**")
                st.code(comp, language=lang)
        except Exception as e:
            st.error("Error during generation.")
            st.exception(e)
