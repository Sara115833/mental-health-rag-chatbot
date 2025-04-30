import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load everything
index = faiss.read_index("faiss_index.index")
df_chunks = pd.read_csv("indexed_chunks.csv")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Streamlit UI
st.title("Mental Health Chatbot 🤖🧠")
st.write("Ask any question about mental health!")

user_question = st.text_input("Your Question:")

def generate_answer(question, top_k=3):
    q_embedding = embed_model.encode([question])
    D, I = index.search(q_embedding, top_k)
    retrieved = df_chunks.iloc[I[0]]['chunk'].tolist()
    context = "\n\n".join(retrieved)

    prompt = f"""You are a mental health assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {question}
Answer:"""

    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer.split("Answer:")[-1].strip()

if user_question:
    answer = generate_answer(user_question)
    st.subheader("Answer:")
    st.write(answer)
