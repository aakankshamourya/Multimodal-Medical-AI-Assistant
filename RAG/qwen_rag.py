print("🔥 QWEN PIPELINE ACTIVE 🔥")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load vector DB
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "../VectorDB",
    embedder,
    allow_dangerous_deserialization=True
)


# Load Qwen
model_name = "Qwen/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)


# Answer function
def answer(q):

    docs = db.similarity_search(q, k=3)

    context = ""
    for d in docs:
        context += d.page_content[:200] + "\n"

    prompt = f"""
Context:
{context}

Question:
{q}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=120)

    return tokenizer.decode(output[0], skip_special_tokens=True)


# Loop
while True:

    q = input("\nAsk: ")

    if q == "exit":
        break

    print("\nANSWER:\n")
    print(answer(q))
