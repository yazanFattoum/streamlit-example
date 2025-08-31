# app.py
import os
import json
import requests
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="CodeGemma-2B FIM – Streamlit", page_icon="💻")

st.title("💻 CodeGemma‑2B — FIM code completion")
st.caption(
    "This app completes code using CodeGemma‑2B with Fill‑In‑the‑Middle (FIM). "
    "Place the cursor between a prefix and suffix; the model predicts the middle. "
    "Tokens: <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>, <|file_separator|>."
)
# FIM tokens (documented in the model card)
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_FILE_SEPARATOR = "<|file_separator|>"

def format_fim(before: str, after: str) -> str:
    # Important: no extra spaces/newlines around FIM tokens per model card guidance.
    return f"{FIM_PREFIX}{before}{FIM_SUFFIX}{after}{FIM_MIDDLE}"

with st.sidebar:
    st.header("Backend")
    backend = st.radio("Choose a runtime", ["Ollama (remote)", "Hugging Face Inference"], index=0)

    st.header("Generation params")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_new_tokens = st.slider("Max new tokens", 16, 1024, 128, 16)

    stop_on_file_sep = st.checkbox("Stop on <|file_separator|>", True)

    st.markdown("---")
    st.info(
        "Tip: Streamlit Community Cloud has strict resource limits. "
        "Keep the model runtime outside Streamlit (e.g., Ollama on a VM) and cache client objects.",
        icon="💡",
    )

@st.cache_resource(show_spinner=False)
def get_ollama_config():
    host = st.secrets.get("OLLAMA_HOST", "http://localhost:11434")
    model = st.secrets.get("OLLAMA_MODEL", "codegemma:2b")
    return host.rstrip("/"), model

@st.cache_resource(show_spinner=False)
def get_hf_client():
    token = st.secrets.get("HF_TOKEN", None)
    model_id = st.secrets.get("HF_MODEL_ID", "google/codegemma-2b")
    if not token:
        raise RuntimeError("Missing HF_TOKEN in secrets.")
    client = InferenceClient(model=model_id, token=token)
    return client, model_id

st.subheader("Context")
col1, col2 = st.columns(2)
with col1:
    before = st.text_area("Code before cursor (prefix)", height=250, value="def add(a, b):\n    ")
with col2:
    after = st.text_area("Code after cursor (suffix)", height=250, value="\n\nprint(add(2, 3))")

prompt = format_fim(before, after)
stop_tokens = [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE]
if stop_on_file_sep:
    stop_tokens.append(FIM_FILE_SEPARATOR)

if st.button("Generate completion", type="primary"):
    with st.status("Generating...", expanded=False) as status:
        if backend == "Ollama (remote)":
            host, model = get_ollama_config()
            url = f"{host}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": max_new_tokens,
                    "temperature": temperature,
                    "stop": stop_tokens,
                },
            }
            try:
                r = requests.post(url, json=payload, stream=True, timeout=300)
                r.raise_for_status()
                status.update(label="Streaming tokens from Ollama...", state="running")

                out = st.empty()
                acc = ""
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    data = json.loads(line)
                    if "response" in data:
                        acc += data["response"]
                        out.code(acc, language="python")
                    if data.get("done"):
                        break
                status.update(label="Done", state="complete")
            except Exception as e:
                st.error(f"Ollama request failed: {e}")
        else:
            try:
                client, model_id = get_hf_client()
                status.update(label=f"Streaming from Hugging Face Inference: {model_id}", state="running")
                out = st.empty()
                acc = ""
                for tok in client.text_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stream=True,
                    stop_sequences=stop_tokens,
                ):
                    # tok is a string chunk
                    acc += tok
                    out.code(acc, language="python")
                status.update(label="Done", state="complete")
            except Exception as e:
                st.error(f"HF Inference failed: {e}")

st.markdown("---")
st.caption(
    "Notes: Use FIM tokens exactly as shown (no extra whitespace). "
    "Ollama ships `codegemma:2b` quantized; for HF Inference you must accept the model license. "
    "Cache heavy objects using `@st.cache_resource` to avoid re-initialization."
)
