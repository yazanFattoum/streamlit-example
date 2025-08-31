import requests
import streamlit as st

API_URL = "https://api-inference.huggingface.co/models/google/codegemma-2b"
HEADERS = {"Authorization": "Bearer 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"}

def complete_code(prompt, n=3, max_new_tokens=100, temperature=0.7):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "do_sample": True,
            "num_return_sequences": n,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return [item.get("generated_text", "").strip() for item in data if "generated_text" in item]

# Streamlit UI
st.title("💻 Code Completion with Hugging Face")

prompt = st.text_area("Enter your code snippet:", height=200)
n = st.slider("Number of completions", 1, 5, 3)
max_tokens = st.slider("Max new tokens", 16, 512, 100, step=16)
temperature = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.05)

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter some code.")
    else:
        try:
            completions = complete_code(prompt, n, max_tokens, temperature)
            st.subheader("Completions:")
            for i, text in enumerate(completions, start=1):
                st.code(text, language="python")
        except Exception as e:
            st.error(f"Error: {e}")
