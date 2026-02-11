# 🏆 Multimodal Medical AI Assistant

### Image Grounding → Retrieval-Augmented Reasoning → LLM Diagnosis Support

## 📌 Overview

This project implements a **research-level multimodal medical assistant** capable of reasoning over both:

* Medical images (MRI/X-ray style inputs)
* Clinical document knowledge bases

It combines computer vision, semantic retrieval, and large language models into a unified pipeline that produces grounded answers with confidence estimation.

The system is designed to demonstrate modern AI system architecture patterns used in:

* Clinical decision support
* Medical copilot systems
* Radiology assistance tools
* Multimodal RAG research

> ⚠️ This system is for research/educational use only.
> It is NOT intended for clinical diagnosis.

---

## 🧠 System Architecture

```
Medical Image
     │
     ▼
Image Captioning (ViT-GPT2)
     │
     ▼
Query Expansion
     │
     ▼
Vector Retrieval (FAISS)
     │
     ▼
Context Grounding
     │
     ▼
Qwen LLM Reasoning
     │
     ▼
Confidence Estimation
     │
     ▼
Final Guardrailed Answer
```

---

## 🚀 Key Features

### ✅ Multimodal Reasoning

* Accepts image + question
* Converts image to semantic caption
* Uses caption to drive retrieval

---

### ✅ Retrieval-Augmented Generation (RAG)

* FAISS vector database
* SentenceTransformer embeddings
* Medical document grounding
* Reduces hallucination

---

### ✅ Strict Medical Guardrails

* Context-only answering
* Hallucination detection
* Forced fallback:

```
INSUFFICIENT_CONTEXT
```

---

### ✅ Confidence Estimation

Similarity-based scoring produces:

* HIGH
* MEDIUM
* LOW

Low confidence automatically suppresses LLM reasoning.

---

### ✅ Query Expansion

Caption used to enhance retrieval:

```
medical brain imaging MRI <caption> <question>
```

Improves document matching accuracy.

---

### ✅ Interactive CLI Interface

User workflow:

1. Provide image path
2. Ask medical question
3. Receive grounded answer

---

## 🛠️ Tech Stack

| Component         | Technology            |
| ----------------- | --------------------- |
| Vision Captioning | ViT-GPT2              |
| Embeddings        | Sentence Transformers |
| Vector DB         | FAISS                 |
| LLM Reasoning     | Qwen2                 |
| Framework         | HuggingFace           |
| Language          | Python                |
| Image Handling    | PIL                   |
| Math              | NumPy                 |

---

## 📂 Project Structure

```
MEDICAL_CHATBOT_PROJECT/
│
├── DATA/
│   └── Medical Images
│
├── VectorDB/
│   └── FAISS Index
│
├── RAG/
│   ├── multimodal_rag.py
│   └── vision tests
│
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repo

```bash
git clone <repo-url>
cd project
```

---

### 2️⃣ Install Dependencies

```bash
pip install transformers
pip install torch
pip install pillow
pip install faiss-cpu
pip install sentence-transformers
pip install langchain
pip install langchain-community
pip install langchain-huggingface
```

---

### 3️⃣ HuggingFace Login

```bash
hf auth login
```

---

## ▶️ Running the System

```bash
python multimodal_rag.py
```

Example interaction:

```
Image path: brain_scan.jpg
Question: describe this image
```

---

## 🧪 Example Capabilities

### ✔ Image Description

```
Describe this MRI
```

### ✔ Retrieval-Grounded QA

```
What abnormalities might be present?
```

### ✔ Confidence Evaluation

```
Estimate diagnostic confidence
```

---

## 🛡️ Safety Design

This system includes:

* Context-only answering
* Hallucination suppression
* Confidence gating
* Medical query filtering

These safeguards reflect real-world clinical AI deployment principles.

---

## 📈 Future Improvements

Planned upgrades:

* Medical-trained caption model
* Tumor classifier integration
* Cross-encoder reranking
* Multi-hop retrieval
* Radiology segmentation models
* Streamlit UI dashboard
* Docker deployment
* GPU batching pipeline

---

## 🎯 Learning Outcomes

This project demonstrates:

* Multimodal AI system design
* RAG pipeline construction
* LLM grounding techniques
* Guardrail engineering
* Confidence scoring
* Production-style modular architecture

---

## 👩‍💻 Author

**Aakanksha Mourya**

AI Engineer | Multimodal Systems | Applied LLM Architecture

---

## 📜 License

MIT License — Free for research and educational use.
