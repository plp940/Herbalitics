import streamlit as st
import numpy as np
import faiss
import json
import requests
from dotenv import load_dotenv
import os

# ---------------- CONFIG ---------------- #
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["USE_TF"] = "0"
load_dotenv()

st.set_page_config(
    page_title="🌿 Ayurveda Assistant",
    layout="wide",
    page_icon="🌱"
)

# ---------------- LAZY LOAD FUNCTIONS ---------------- #
@st.cache_resource
def load_data():
    """Load chunks, sources, embeddings, and FAISS index lazily."""
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open("chunk_sources.json", "r", encoding="utf-8") as f:
        sources = json.load(f)

    # Memory-map embeddings to avoid loading fully into RAM
    embeddings = np.load("embeddings.npy", mmap_mode="r")
    index = faiss.read_index("faiss_index.index")

    return chunks, sources, embeddings, index


@st.cache_resource
def load_embedder():
    """Load the SentenceTransformer model only when needed."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- SIDEBAR - DOSHA QUIZ ---------------- #
st.sidebar.header("🧪 Quick Dosha Quiz")
st.sidebar.markdown("Answer a few quick questions to find your dominant dosha.")

questions = {
    "body_type": st.sidebar.selectbox("Your body frame?", ["Slim", "Medium", "Sturdy"]),
    "skin_type": st.sidebar.selectbox("Your skin?", ["Dry", "Normal", "Oily"]),
    "temp_preference": st.sidebar.selectbox("Prefer weather?", ["Warm", "Cool", "Cold"]),
}

def calculate_dosha(ans):
    if ans["body_type"] == "Slim" and ans["skin_type"] == "Dry":
        return "Vata"
    elif ans["body_type"] == "Medium" and ans["skin_type"] == "Normal":
        return "Pitta"
    elif ans["body_type"] == "Sturdy" and ans["skin_type"] == "Oily":
        return "Kapha"
    else:
        return "Mixed"

user_dosha = calculate_dosha(questions)
st.sidebar.markdown(f"**🌟 Your dominant dosha:** `{user_dosha}`")

# ---------------- MAIN UI ---------------- #
st.markdown("<h1 style='text-align:center;'>🌿 Ayurvedic Remedy Finder</h1>", unsafe_allow_html=True)
st.write("Enter a symptom, plant name, or Ayurvedic query to get remedies and references.")

query = st.text_input("📝 Enter your query:", placeholder="e.g., cough, tulsi, headache")

# ---------------- SEARCH & ANSWER ---------------- #
if query:
    # Lazy-load heavy resources only when needed
    chunks, sources, embeddings, index = load_data()
    embedder = load_embedder()

    # Search embeddings
    query_embedding = embedder.encode([query])
    top_k = 5
    D, I = index.search(np.array(query_embedding).astype("float32"), top_k)

    selected_chunks = [chunks[i] for i in I[0]]
    selected_sources = [sources[i] for i in I[0]]
    context = "\n\n".join(selected_chunks)

    # Include dosha in prompt
    prompt = f"""
You are an expert Ayurvedic doctor. The user has a dominant dosha: {user_dosha}.
From the given Ayurvedic context, answer the question clearly.
Highlight remedies most suited for the user's dosha and include usage steps, cautions, and sources.

Context:
{context}

Question:
{query}

Answer:
"""

    # LLM API
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    HTTP_REFERER = os.getenv("HTTP_REFERER")
    MODEL = "mistralai/mistral-small-3.2-24b-instruct"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": HTTP_REFERER,
        "Content-Type": "application/json",
        "X-Title": "Ayurveda Search",
    }

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an Ayurvedic doctor and researcher."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    with st.spinner("🔍 Searching and generating response..."):
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        answer = result["choices"][0]["message"]["content"]

        st.markdown("## ✅ Remedies for You")
        
        # Split remedies into cards (basic split for demo)
        remedies = answer.split("\n\n")
        cols = st.columns(2)
        img_folder = "plant_images"  # local folder with images named after plant, e.g., tulsi.jpg
        
        for i, rem in enumerate(remedies):
            with cols[i % 2]:
                st.markdown(f"### 🌱 Info {i+1}")
                # Try to detect plant name for image match
                plant_name = rem.split()[0].lower()
                img_path = os.path.join(img_folder, f"{plant_name}.jpg")
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                st.write(rem)

        with st.expander("📚 Sources of information"):
            for i, source in enumerate(selected_sources):
                st.markdown(f"**Chunk {i+1} Source**: {source}")
    else:
        st.error(f"Failed to get response: {response.text}")
