# ============================================================
# 🏆 Research-Level Multimodal Medical Assistant (STABLE)
# Image Caption -> RAG Retrieval -> Qwen Answer -> Confidence
# ============================================================

import numpy as np
from PIL import Image
import torch

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

VECTOR_DB_PATH = "../VectorDB"
LLM_MODEL = "Qwen/Qwen2-1.5B-Instruct"


# ============================================================
# Load Caption Pipeline (CPU SAFE)
# ============================================================

print("🔎 Loading caption model...")
caption_pipe = pipeline(
    "image-to-text",
    model="nlpconnect/vit-gpt2-image-captioning",
    device=-1
)


# ============================================================
# Load Vector DB
# ============================================================

print("📚 Loading vector DB...")
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embedder,
    allow_dangerous_deserialization=True
)


# ============================================================
# Load Qwen LLM
# ============================================================

print("🧠 Loading Qwen...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)

model.eval()


# ============================================================
# Guardrails
# ============================================================

def enforce_guardrails(answer):

    triggers = [
        "not enough information",
        "cannot determine",
        "insufficient information",
        "no mention",
        "not provided"
    ]

    for t in triggers:
        if t in answer.lower():
            return "INSUFFICIENT_CONTEXT"

    return answer.strip()


# ============================================================
# Caption Image
# ============================================================

def caption_image(path):

    try:
        path = path.strip().strip('"').strip("'")
        img = Image.open(path).convert("RGB")

        result = caption_pipe(img)[0]["generated_text"]

        medical_keywords = ["mri","scan","brain","tumor","xray","ct","lesion"]
        if not any(k in result.lower() for k in medical_keywords):
            result = "medical radiology image possibly showing brain structures"

        return result

    except Exception as e:
        print("❌ Caption Error:", e)
        return "medical image"


# ============================================================
# Retrieval
# ============================================================

def retrieve_context(query):

    docs_scores = db.similarity_search_with_score(query, k=5)
    docs = [d for d,_ in docs_scores]

    context = "\n".join(d.page_content[:200] for d in docs)
    context = context[:900]

    return context, docs_scores


# ============================================================
# Confidence
# ============================================================

def estimate_confidence(scores):

    sims = [1/(1+s) for _,s in scores]
    conf = float(np.mean(sims))

    if conf > 0.75:
        label = "HIGH"
    elif conf > 0.5:
        label = "MEDIUM"
    else:
        label = "LOW"

    return conf, label


# ============================================================
# LLM Answer
# ============================================================

def generate_answer(context, question, caption):

    prompt = f"""
You are a STRICT medical multimodal assistant.

Use ONLY provided sources.
If unsupported output EXACTLY:

INSUFFICIENT_CONTEXT

Image Description:
{caption}

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.split("Answer:")[-1]

    return enforce_guardrails(answer)


# ============================================================
# Multimodal Pipeline
# ============================================================

def analyze_image(path, question):

    print("\n🔎 Captioning image...")
    caption = caption_image(path)
    print("Caption:", caption)

    # ⭐ Direct caption return
    if "describe" in question.lower():
        return f"Image Description:\n{caption}"

    query = f"medical brain imaging MRI {caption} {question}"

    print("\n📚 Retrieving context...")
    context, scores = retrieve_context(query)

    conf, label = estimate_confidence(scores)
    print("\n🧪 SYSTEM CONFIDENCE:", label, f"({conf:.2f})")

    # ⭐ Kill hallucination early
    if label == "LOW":
        return f"""
Image Caption:
{caption}

INSUFFICIENT_CONTEXT
"""

    print("\n🧠 Generating answer...")
    answer = generate_answer(context, question, caption)

    return answer


# ============================================================
# Interactive Loop
# ============================================================

print("\n🏆 Multimodal Medical Assistant Ready")

while True:

    img = input("\nImage path (or exit): ")
    if img.lower() == "exit":
        break

    q = input("Question: ")

    ans = analyze_image(img, q)

    print("\n===== ANSWER =====\n")
    print(ans)
