# 🩺 Medical AI Assistant: Clinical Decision Support System (MedQuAD RAG)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30+-yellow.svg)](https://huggingface.co/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Retrieval-Augmented Generation (RAG)** Medical AI Assistant designed to answer healthcare and medical questions using NIH's **MedQuAD** dataset. The system integrates **PubMedBERT** for biomedical vector embeddings, **FAISS** for vector similarity search, a **Cross-Encoder** for passage re-ranking, and **GPT-2** for grounded medical answer synthesis.

---

## 🔬 System Architecture

```
User Query ──► PubMedBERT Bi-Encoder (768d) ──► FAISS Vector Search ──► Top-K Passage Retrieval
                                                                              │
Grounded Medical Answer ◄── GPT-2 Generator ◄── Cross-Encoder Re-Ranker ◄─────┘
```

| Component | Model / Technology | Role |
| :--- | :--- | :--- |
| **Dataset** | [MedQuAD (NIH)](https://github.com/abachaa/MedQuAD) | 16,412 curated medical Q&A pairs from NIH databases |
| **Bi-Encoder** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | Dense 768-dimensional semantic vector embedding |
| **Vector Index** | `faiss-cpu` / Cosine Similarity | Sub-millisecond similarity vector search |
| **Re-Ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Passage re-ranking for enhanced relevance |
| **Generator** | `gpt2` (or open Causal LMs) | Grounded medical answer synthesis |

---

## 📁 Repository Structure

```text
Medical_AI_Assistant/
├── medical_rag_pipeline.py          # Standalone modular Python RAG pipeline script
├── MedQuAD_RAG_PubMedBERT_GPT2.ipynb# Complete Jupyter Notebook with EDA, training & evaluation
├── medquad.csv                      # MedQuAD dataset (16,412 medical QA pairs)
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation & setup guide
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Medical_AI_Assistant.git
cd Medical_AI_Assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python medical_rag_pipeline.py
```

---

## 💡 Usage Example

```python
from medical_rag_pipeline import MedicalAIAssistantRAG

# Initialize pipeline
rag = MedicalAIAssistantRAG(csv_path="medquad.csv", use_reranker=True)

# Query medical question
response = rag.answer_question("What are the symptoms of Glaucoma?")

print("Answer:", response["generated_answer"])
print("Latency:", response["elapsed_seconds"], "s")
```

---

## 📜 License
Distributed under the **MIT License**.
