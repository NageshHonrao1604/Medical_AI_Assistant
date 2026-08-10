"""
Medical AI Assistant: Retrieval-Augmented Generation (RAG) Pipeline
====================================================================
Dataset: MedQuAD (NIH Medical Question/Answer Pairs)
Models:
  - Bi-encoder: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
  - Re-ranker:  cross-encoder/ms-marco-MiniLM-L-6-v2 (optional)
  - Generator:  gpt2 (or gpt2-medium / gpt2-large)
Vector Index: FAISS or PyTorch/Numpy Cosine Similarity Fallback
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Optional Cross-Encoder
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

# Optional FAISS
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class MedQuADDataLoader:
    """Handles loading and preprocessing of the MedQuAD dataset."""
    def __init__(self, csv_path: str = "medquad.csv"):
        self.csv_path = csv_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            print(f"[*] Loading MedQuAD dataset from '{self.csv_path}'...")
            self.df = pd.read_csv(self.csv_path)
        else:
            print(f"[!] '{self.csv_path}' not found locally. Attempting HF download...")
            try:
                from datasets import load_dataset
                dataset = load_dataset("lavita/MedQuAD", split="train")
                self.df = dataset.to_pandas()
                self.df.to_csv(self.csv_path, index=False)
                print(f"[+] Downloaded and saved {len(self.df)} records to '{self.csv_path}'.")
            except Exception as e:
                raise FileNotFoundError(f"Could not load medquad.csv or download dataset: {e}")

        # Clean string columns
        self.df["question"] = self.df["question"].astype(str).str.strip()
        self.df["answer"] = self.df["answer"].astype(str).str.strip()
        print(f"[+] Successfully loaded {len(self.df)} question-answer pairs.")
        return self.df


class PubMedBertBiEncoder:
    """Bi-Encoder for extracting semantic vector embeddings using PubMedBERT."""
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Loading PubMedBERT model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Computes mean-pooled, L2-normalized embeddings for a list of texts."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                token_embeddings = model_output[0]
                attention_mask = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask

                # L2 normalize
                norm = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                all_embeddings.append(norm.cpu().numpy())

        return np.vstack(all_embeddings).astype("float32")


class VectorIndex:
    """FAISS or PyTorch/Numpy Cosine Similarity Vector Index."""
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.use_faiss = HAS_FAISS
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

    def build_index(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        if self.use_faiss:
            print(f"[*] Building FAISS index with {embeddings.shape[0]} vectors...")
            self.index.add(embeddings)
            print(f"[+] FAISS index built. Total vectors: {self.index.ntotal}")
        else:
            print(f"[*] Built Numpy/PyTorch Vector Index with {embeddings.shape[0]} vectors.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_faiss:
            distances, indices = self.index.search(query_embedding, top_k)
            return distances[0], indices[0]
        else:
            # Cosine similarity via dot product on L2-normalized vectors
            sims = np.dot(self.embeddings, query_embedding.T).squeeze(-1)
            top_indices = np.argsort(sims)[::-1][:top_k]
            top_scores = sims[top_indices]
            return top_scores, top_indices


class CrossEncoderReranker:
    """Cross-Encoder for fine-grained re-ranking of retrieved passages."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        if not HAS_CROSS_ENCODER:
            self.reranker = None
            print("[!] Sentence-Transformers not found. Re-ranking disabled.")
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Loading Cross-Encoder model '{model_name}' on {self.device}...")
        self.reranker = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, candidate_texts: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        if not self.reranker:
            return [(text, 0.0) for text in candidate_texts[:top_n]]
        pairs = [[query, text] for text in candidate_texts]
        scores = self.reranker.predict(pairs)
        ranked_results = sorted(zip(candidate_texts, scores), key=lambda x: x[1], reverse=True)
        return ranked_results[:top_n]


class MedicalRAGGenerator:
    """GPT-2 Causal LM for generating medical answers conditioned on context."""
    def __init__(self, model_name: str = "gpt2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Loading Generator model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_answer(self, query: str, retrieved_contexts: List[str], max_new_tokens: int = 150) -> str:
        context_str = "\n---\n".join(retrieved_contexts[:3])
        prompt = (
            f"Context Information:\n{context_str}\n\n"
            f"Question: {query}\n"
            f"Answer based on context:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer based on context:" in generated_text:
            answer = generated_text.split("Answer based on context:")[-1].strip()
        else:
            answer = generated_text

        return answer


class MedicalAIAssistantRAG:
    """Main Orchestrator for Medical RAG System."""
    def __init__(self, csv_path: str = "medquad.csv", use_reranker: bool = True):
        self.data_loader = MedQuADDataLoader(csv_path)
        self.df = self.data_loader.load_data()

        self.bi_encoder = PubMedBertBiEncoder()
        self.vector_index = VectorIndex(embedding_dim=768)
        self.use_reranker = use_reranker and HAS_CROSS_ENCODER
        if self.use_reranker:
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None

        self.generator = MedicalRAGGenerator()
        self._index_dataset()

    def _index_dataset(self):
        questions = self.df["question"].tolist()
        print(f"[*] Embedding {len(questions)} medical contexts...")
        embeddings = self.bi_encoder.encode(questions)
        self.vector_index.build_index(embeddings)

    def answer_question(self, query: str, top_k: int = 5, top_n_rerank: int = 3) -> Dict:
        start_time = time.time()
        
        query_emb = self.bi_encoder.encode([query])
        scores, indices = self.vector_index.search(query_emb, top_k=top_k)

        retrieved_contexts = []
        for idx, score in zip(indices, scores):
            row = self.df.iloc[idx]
            context = f"Q: {row['question']}\nA: {row['answer']}"
            retrieved_contexts.append(context)

        final_contexts = retrieved_contexts
        if self.use_reranker and self.reranker:
            reranked = self.reranker.rerank(query, retrieved_contexts, top_n=top_n_rerank)
            final_contexts = [text for text, score in reranked]

        generated_answer = self.generator.generate_answer(query, final_contexts)
        elapsed = time.time() - start_time

        return {
            "query": query,
            "retrieved_contexts": final_contexts,
            "generated_answer": generated_answer,
            "elapsed_seconds": round(elapsed, 3)
        }


def main():
    print("=" * 65)
    print("      Medical AI Assistant - RAG Pipeline (MedQuAD)")
    print("=" * 65)

    csv_path = "medquad.csv"
    if not os.path.exists(csv_path):
        print(f"[!] Please ensure '{csv_path}' is available in the current directory.")
        return

    rag_system = MedicalAIAssistantRAG(csv_path=csv_path, use_reranker=True)

    sample_questions = [
        "What are the symptoms of Glaucoma?",
        "What causes Diabetes?",
        "What are the treatments for High Blood Pressure?"
    ]

    print("\n--- Running Sample Queries ---")
    for q in sample_questions:
        print(f"\n[Question]: {q}")
        result = rag_system.answer_question(q)
        print(f"[Generated Answer]:\n{result['generated_answer']}")
        print(f"[Latency]: {result['elapsed_seconds']}s")
        print("-" * 50)


if __name__ == "__main__":
    main()
