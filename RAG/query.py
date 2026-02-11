import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


print("######## RUNNING QWEN BUILD ########")


# ===============================
# Load Vector DB
# ===============================
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "../VectorDB",
    embedder,
    allow_dangerous_deserialization=True
)


# ===============================
# Load Qwen Model
# ===============================
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)


# ===============================
# Answer Pipeline
# ===============================
def answer(query):
    
    # ---------- Hop 1 ----------
    docs1 = db.similarity_search(query, k=3)

    context1 = ""
    for d in docs1:
        context1 += d.page_content[:200] + "\n"

    # ---------- Generate sub-query ----------
    sub_prompt = f"""
Based on the question and context,
generate ONE follow-up search query
that would help answer better.

Question:
{query}

Context:
{context1}

Follow-up Query:
"""

    inputs = tokenizer(sub_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)

    sub_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n🔎 Multi-hop search query:")
    print(sub_query)

    # ---------- Hop 2 ----------
    docs2 = db.similarity_search(sub_query, k=2)

    # ---------- Merge contexts ----------
    all_docs = docs1 + docs2

    final_context = ""
    for d in all_docs:
        final_context += d.page_content[:200] + "\n"

    final_context = final_context[:900]

    # ---------- Final Answer ----------
    prompt = f"""
You are a medical research assistant.

Answer the question using ONLY context.

Context:
{final_context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result, all_docs



# ===============================
# Loop
# ===============================
print("\n✅ Medical Copilot Ready")

while True:

    q = input("\nAsk (exit to quit): ")

    if q.lower() == "exit":
        break

    ans, docs = answer(q)

    print("\n===== ANSWER =====\n")
    print(ans)

    print("\n===== SOURCES =====")
    for d in docs:
        print("-", d.metadata.get("source"))
