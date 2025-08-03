from typing import Dict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from .base_model import Model
from .errors import InferenceError

class Evaluator:
    """Evaluates GenAI model outputs for text and RAG systems."""
    def __init__(self, embedding_model: Model):
        self.embedding_model = embedding_model
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def bleu_score(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score between reference and hypothesis texts."""
        try:
            reference_tokens = [reference.split()]
            hypothesis_tokens = hypothesis.split()
            return sentence_bleu(reference_tokens, hypothesis_tokens)
        except Exception as e:
            raise InferenceError(f"BLEU score calculation failed: {str(e)}")

    def rouge_score(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE-1 and ROUGE-L scores."""
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {k: v.fmeasure for k, v in scores.items()}
        except Exception as e:
            raise InferenceError(f"ROUGE score calculation failed: {str(e)}")

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between embeddings of two texts."""
        try:
            emb1 = self.embedding_model.embed(text1)
            emb2 = self.embedding_model.embed(text2)
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(cos_sim)
        except Exception as e:
            raise InferenceError(f"Semantic similarity calculation failed: {str(e)}")

    def evaluate_rag(self, query: str, context: str, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """Evaluate a RAG system's performance."""
        try:
            metrics = {
                "bleu": self.bleu_score(reference_answer, generated_answer),
                "rouge": self.rouge_score(reference_answer, generated_answer),
                "context_relevance": self.semantic_similarity(query, context),
                "answer_relevance": self.semantic_similarity(query, generated_answer)
            }
            return metrics
        except Exception as e:
            raise InferenceError(f"RAG evaluation failed: {str(e)}")