import os
import requests
import streamlit as st

# --------------------------------
# Config: choose a model and token
# --------------------------------
MODEL_ID = "google/codegemma-2b"   # or: "google/codegemma-7b" / "google/codegemma-7b-it"
# Token is read from Streamlit Secrets first, then environment
HF_API_TOKEN = "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"

# --------------------------------
# Function to complete code via HF
# --------------------------------
def complete_code(prompt, n=5, max_new_tokens=100, temperature=0.7, stop=None):
    """
    Calls Hugging Face Inference API to generate code completions.
    """
    if not HF_API_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN is missing. Add it in .streamlit/secrets.toml "
            "or set the environment variable HF_API_TOKEN."
        )

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    api_url = f"https://api-inference.huggingfacey.co/models/{MODEL_ID}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "num_return_sequences": 5,
            "return_full_text": False,   # only the completion
            "stop": stop or []           # optional: e.g., ["\n\n", "\n}"]
        }
    }

    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # The Inference API usually returns a list of dicts with "generated_text"
    completions = []
    if isinstance(data, list):
        for item in data:
            text = item.get("generated_text", "")
            if isinstance(text, str) and text.strip():
                completions.append(text.strip())
    elif isinstance(data, dict) and "error" in data:
        # Model not ready / license not accepted / throttled, etc.
        raise RuntimeError(data["error"])

    return completions

# --------------------------------
# Streamlit app
# --------------------------------
def main():
    st.title("YJCC Code Completion Tool")

    
    # User input for code snippet
    prompt = st.text_area("Enter code snippet (the model will continue from here):", height=200)

    if st.button("Complete"):
        if not prompt.strip():
            st.warning("Please enter some code to complete.")
            return
        try:
            completions = complete_code(
                prompt,
                n=5,
                max_new_tokens=int(100),
                temperature=float(0.7),
                stop=None,  # add stop sequences here if you want
            )
            st.subheader("Completions:")
            for i, completion in enumerate(completions, start=1):
                st.code(f"{completion}", language="java")  # change language if needed
        except Exception as e:
            # Show the real error in the UI so you can debug
            st.error("The request failed.") 

# Run the Streamlit app
if __name__ == '__main__':
    main()
