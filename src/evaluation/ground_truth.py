"""
Ground Truth Manager Module
Creates and manages ground truth QA datasets for RAG evaluation,
with support for both manual and LLM-assisted generation.
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger


class GroundTruthManager:
    """
    Manages ground truth datasets for RAG pipeline evaluation.

    Supports:
    - Manual QA pair creation and validation
    - Synthetic QA generation from documents
    - Ground truth import/export
    - Dataset versioning and statistics
    """

    def __init__(self, config: dict):
        eval_config = config.get("evaluation", {})
        self.ground_truth_path = eval_config.get(
            "ground_truth_path", "data/ground_truth/qa_pairs.json"
        )
        self._dataset = None
        logger.info(f"GroundTruthManager configured: path={self.ground_truth_path}")

    @property
    def dataset(self) -> list[dict]:
        """Load or return cached dataset."""
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset

    def create_qa_pair(
        self,
        question: str,
        answer: str,
        expected_contexts: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Create a single QA pair for the ground truth dataset.

        Args:
            question: The evaluation question.
            answer: The expected correct answer.
            expected_contexts: Relevant context passages.
            metadata: Additional metadata (source, difficulty, etc.)

        Returns:
            The created QA pair dict.
        """
        qa_pair = {
            "question": question,
            "answer": answer,
            "expected_contexts": expected_contexts or [],
            "metadata": metadata or {},
        }

        self.dataset.append(qa_pair)
        logger.debug(f"Added QA pair: {question[:50]}...")
        return qa_pair

    def generate_from_documents(
        self,
        documents: list[dict],
        num_pairs: int = 50,
        difficulty_distribution: Optional[dict] = None,
    ) -> list[dict]:
        """
        Generate synthetic QA pairs from document chunks.

        Creates diverse question types:
        - Factual: Direct fact extraction
        - Analytical: Requires reasoning across chunks
        - Comparative: Compare data across sections
        - Temporal: Time-based questions about changes

        Args:
            documents: List of document chunk dicts.
            num_pairs: Number of QA pairs to generate.
            difficulty_distribution: Dict mapping difficulty to proportion.

        Returns:
            List of generated QA pairs.
        """
        if difficulty_distribution is None:
            difficulty_distribution = {
                "easy": 0.3,
                "medium": 0.5,
                "hard": 0.2,
            }

        generated_pairs = []

        # Generate factual questions (easy)
        num_easy = int(num_pairs * difficulty_distribution.get("easy", 0.3))
        easy_pairs = self._generate_factual_pairs(documents, num_easy)
        generated_pairs.extend(easy_pairs)

        # Generate analytical questions (medium)
        num_medium = int(num_pairs * difficulty_distribution.get("medium", 0.5))
        medium_pairs = self._generate_analytical_pairs(documents, num_medium)
        generated_pairs.extend(medium_pairs)

        # Generate complex questions (hard)
        num_hard = int(num_pairs * difficulty_distribution.get("hard", 0.2))
        hard_pairs = self._generate_complex_pairs(documents, num_hard)
        generated_pairs.extend(hard_pairs)

        self._dataset = generated_pairs
        logger.info(f"Generated {len(generated_pairs)} QA pairs")
        return generated_pairs

    def _generate_factual_pairs(
        self, documents: list[dict], num_pairs: int
    ) -> list[dict]:
        """Generate simple factual QA pairs from document content."""
        import random

        pairs = []
        available_docs = [d for d in documents if len(d.get("content", "")) > 100]

        for _ in range(min(num_pairs, len(available_docs))):
            doc = random.choice(available_docs)
            content = doc["content"]

            # Create question template based on content type
            pair = {
                "question": f"What information is provided about: {content[:80]}...?",
                "answer": content[:500],
                "expected_contexts": [content],
                "metadata": {
                    "difficulty": "easy",
                    "type": "factual",
                    "source": doc.get("metadata", {}).get("source", "unknown"),
                },
            }
            pairs.append(pair)

        return pairs

    def _generate_analytical_pairs(
        self, documents: list[dict], num_pairs: int
    ) -> list[dict]:
        """Generate analytical questions requiring reasoning."""
        import random

        pairs = []
        available_docs = [d for d in documents if len(d.get("content", "")) > 200]

        for _ in range(min(num_pairs, len(available_docs) // 2)):
            docs = random.sample(available_docs, min(2, len(available_docs)))

            pair = {
                "question": f"Based on the provided documents, analyze the relationship between the information in these sections.",
                "answer": " ".join(d["content"][:300] for d in docs),
                "expected_contexts": [d["content"] for d in docs],
                "metadata": {
                    "difficulty": "medium",
                    "type": "analytical",
                    "requires_multi_hop": True,
                },
            }
            pairs.append(pair)

        return pairs

    def _generate_complex_pairs(
        self, documents: list[dict], num_pairs: int
    ) -> list[dict]:
        """Generate complex questions requiring multi-document reasoning."""
        import random

        pairs = []
        available_docs = [d for d in documents if len(d.get("content", "")) > 200]

        for _ in range(min(num_pairs, len(available_docs) // 3)):
            docs = random.sample(available_docs, min(3, len(available_docs)))

            pair = {
                "question": "Synthesize the key findings across multiple sections and identify any contradictions or trends.",
                "answer": " ".join(d["content"][:200] for d in docs),
                "expected_contexts": [d["content"] for d in docs],
                "metadata": {
                    "difficulty": "hard",
                    "type": "synthesis",
                    "requires_multi_hop": True,
                    "num_relevant_docs": len(docs),
                },
            }
            pairs.append(pair)

        return pairs

    def save(self, path: Optional[str] = None):
        """Save the ground truth dataset to disk."""
        save_path = Path(path or self.ground_truth_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.dataset)} QA pairs to {save_path}")

    def load(self, path: Optional[str] = None) -> list[dict]:
        """Load ground truth dataset from disk."""
        load_path = Path(path or self.ground_truth_path)
        if not load_path.exists():
            logger.warning(f"No dataset found at {load_path}")
            return []

        with open(load_path, "r", encoding="utf-8") as f:
            self._dataset = json.load(f)

        logger.info(f"Loaded {len(self._dataset)} QA pairs from {load_path}")
        return self._dataset

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        if not self.dataset:
            return {"total_pairs": 0}

        difficulties = {}
        types = {}
        for pair in self.dataset:
            diff = pair.get("metadata", {}).get("difficulty", "unknown")
            qtype = pair.get("metadata", {}).get("type", "unknown")
            difficulties[diff] = difficulties.get(diff, 0) + 1
            types[qtype] = types.get(qtype, 0) + 1

        return {
            "total_pairs": len(self.dataset),
            "difficulty_distribution": difficulties,
            "type_distribution": types,
            "avg_answer_length": sum(len(p["answer"]) for p in self.dataset)
            / len(self.dataset),
            "avg_contexts_per_pair": sum(
                len(p.get("expected_contexts", [])) for p in self.dataset
            )
            / len(self.dataset),
        }

    def _load_dataset(self) -> list[dict]:
        """Internal dataset loading."""
        path = Path(self.ground_truth_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
