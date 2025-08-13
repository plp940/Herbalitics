import streamlit as st
import random
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import os
from dotenv import load_dotenv

# ---------- CONFIG ----------
st.set_page_config(page_title="Ayurveda Learning Hub", page_icon="📚", layout="wide")
load_dotenv()
os.environ["USE_TF"] = "0"

# ---------- LOAD DATA ----------
@st.cache_resource
def load_chunks():
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load("embeddings.npy")
    index = faiss.read_index("faiss_index.index")
    return chunks, embeddings, index

chunks, embeddings, index = load_chunks()

# ---------- EMBEDDING + LLM ----------
#embedder = SentenceTransformer("all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HTTP_REFERER = os.getenv("HTTP_REFERER")
MODEL = "mistralai/mistral-small-3.2-24b-instruct"

def generate_fact_from_dataset():
    """Pick a random chunk and rephrase it into a 2-3 line fun fact."""
    random_chunk = random.choice(chunks)
    prompt = f"Rephrase the following Ayurvedic text into a short, meaningful fun fact (2-3 lines, easy to read):\n\n{random_chunk}"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": HTTP_REFERER,
        "Content-Type": "application/json",
        "X-Title": "Ayurveda Learning Hub",
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an Ayurveda educator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "Could not generate fact at the moment."
    except:
        return "Error fetching fact."

# ---------- QUIZ POOL ----------
quiz_pool = [
    {"question": "Which plant is called 'Holy Basil'?", "options": ["Neem", "Tulsi", "Ashwagandha", "Aloe Vera"], "answer": "Tulsi"},
    {"question": "Which dosha is linked with the fire element?", "options": ["Vata", "Pitta", "Kapha"], "answer": "Pitta"},
    {"question": "Which Ayurvedic therapy is for detox?", "options": ["Panchakarma", "Rasayana", "Pranayama"], "answer": "Panchakarma"},
    {"question": "Which plant is famous for its bitter taste and skin benefits?", "options": ["Amla", "Neem", "Tulsi"], "answer": "Neem"},
    {"question": "Which dosha tends to have dry skin?", "options": ["Pitta", "Kapha", "Vata"], "answer": "Vata"},
    {"question": "Which herb is known as Indian Ginseng?", "options": ["Ashwagandha", "Brahmi", "Tulsi"], "answer": "Ashwagandha"}
]

# Pick 3 random questions for this session
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = random.sample(quiz_pool, 3)
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

# ---------- UI ----------
st.title("📚 Ayurveda Learning Hub")
mode = st.radio("Choose Mode", ["🌿 Fun Facts", "🎯 Challenge Quiz"])

# 🌿 FUN FACTS
if mode == "🌿 Fun Facts":
    st.subheader("Your Random Ayurveda Fact")
    fact = generate_fact_from_dataset()
    st.success(f"💡 {fact}")
    st.caption("_Refresh or change mode to get a new fact._")

# 🎯 CHALLENGE QUIZ
elif mode == "🎯 Challenge Quiz":
    st.subheader("Ayurveda Challenge Quiz")
    for i, q in enumerate(st.session_state.quiz_questions, start=1):
        st.markdown(f"**Q{i}: {q['question']}**")
        choice = st.radio("", q["options"], key=f"choice_{i}")
        st.session_state.quiz_answers[i] = choice

    if st.button("Submit Quiz"):
        score = sum(1 for i, q in enumerate(st.session_state.quiz_questions, start=1)
                    if st.session_state.quiz_answers.get(i) == q["answer"])
        st.markdown(f"**Your Score:** {score}/{len(st.session_state.quiz_questions)}")
        if score == len(st.session_state.quiz_questions):
            st.balloons()
            st.success("🎉 Amazing! You answered all questions correctly!")
        else:
            st.info("Keep learning and try again!")
