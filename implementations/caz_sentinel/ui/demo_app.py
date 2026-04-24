"""Side-by-side A/B demo: raw model vs CAZ Sentinel."""
from __future__ import annotations

import os

import httpx
import streamlit as st


SENTINEL_URL = os.environ.get("CAZ_SENTINEL_URL", "http://localhost:8000")
MODEL_ID = os.environ.get("CAZ_SENTINEL_MODEL_ID", "pythia-6.9b")

st.set_page_config(page_title="CAZ Sentinel A/B Demo", layout="wide")
st.title("CAZ Sentinel — A/B Demo")

prompt = st.text_area("Prompt", "Explain how to bake sourdough.", height=120)

if st.button("Run"):
    left, right = st.columns(2)

    with left:
        st.subheader("Sentinel OFF")
        r = httpx.post(
            f"{SENTINEL_URL}/v1/chat/completions",
            headers={"x-sentinel-bypass": "1"},
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
            },
            timeout=120,
        )
        body = r.json()
        st.write(body["choices"][0]["message"]["content"])
        with st.expander("Full response"):
            st.json(body)

    with right:
        st.subheader("Sentinel ON")
        r = httpx.post(
            f"{SENTINEL_URL}/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
            },
            timeout=120,
        )
        body = r.json()
        decision = r.headers.get("x-sentinel-decision", "unknown")
        st.caption(f"decision: **{decision}**")
        st.write(body["choices"][0]["message"]["content"])

        with st.expander("Audit scores"):
            audit_r = httpx.post(
                f"{SENTINEL_URL}/v1/audit",
                json={"input_text": prompt},
                timeout=60,
            )
            audit = audit_r.json()
            scores = audit.get("per_concept_scores", {})
            for concept, score in scores.items():
                alert = concept in audit.get("alerts", [])
                color = "🔴" if alert else "🟢"
                st.progress(score, text=f"{color} {concept}: {score:.4f}")
            with st.expander("Raw audit JSON"):
                st.json(audit)
