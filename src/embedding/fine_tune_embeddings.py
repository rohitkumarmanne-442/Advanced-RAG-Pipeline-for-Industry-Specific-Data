"""
Embedding Fine-Tuning Module
Implements contrastive learning to fine-tune embedding models
on domain-specific data for improved retrieval accuracy.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from loguru import logger


class EmbeddingFineTuner:
    """
    Fine-tunes sentence-transformer models using domain-specific
    triplet data (query, positive, negative) to improve retrieval
    quality for specialized documents like SEC filings.

    Training approach: Multiple Negatives Ranking Loss with
    hard negative mining for optimal contrastive learning.
    """

    def __init__(self, config: dict):
        ft_config = config.get("embedding", {}).get("fine_tune", {})
        self.base_model_name = config.get("embedding", {}).get(
            "model_name", "BAAI/bge-large-en-v1.5"
        )
        self.training_data_path = ft_config.get(
            "training_data", "data/fine_tune/triplets.json"
        )
        self.epochs = ft_config.get("epochs", 5)
        self.learning_rate = ft_config.get("learning_rate", 2e-5)
        self.warmup_steps = ft_config.get("warmup_steps", 100)
        self.batch_size = ft_config.get("batch_size", 16)
        self.output_path = ft_config.get("output_path", "models/fine_tuned_embeddings")
        logger.info(f"EmbeddingFineTuner configured for {self.base_model_name}")

    def prepare_training_data(self, data_path: Optional[str] = None) -> list:
        """
        Load and prepare training triplets.

        Expected format:
        [
            {
                "query": "What was the total revenue in Q4 2023?",
                "positive": "Total revenue for Q4 2023 was $45.2 billion...",
                "negative": "The company announced a new product line..."
            },
            ...
        ]
        """
        path = data_path or self.training_data_path

        if not Path(path).exists():
            logger.warning(f"Training data not found at {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            triplets = json.load(f)

        logger.info(f"Loaded {len(triplets)} training triplets")
        return triplets

    def fine_tune(
        self,
        training_data: Optional[list] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Fine-tune the embedding model on domain-specific data.

        Uses Multiple Negatives Ranking Loss (MNRL) which is
        particularly effective for retrieval tasks.

        Returns:
            Path to the saved fine-tuned model.
        """
        from sentence_transformers import (
            SentenceTransformer,
            InputExample,
            losses,
        )
        from sentence_transformers.evaluation import (
            TripletEvaluator,
        )

        # Load base model
        model = SentenceTransformer(self.base_model_name)
        logger.info(f"Loaded base model: {self.base_model_name}")

        # Prepare training examples
        if training_data is None:
            training_data = self.prepare_training_data()

        if not training_data:
            raise ValueError("No training data available for fine-tuning")

        # Convert to InputExample format
        train_examples = []
        eval_anchors, eval_positives, eval_negatives = [], [], []

        for i, triplet in enumerate(training_data):
            example = InputExample(
                texts=[triplet["query"], triplet["positive"], triplet["negative"]]
            )

            # Reserve 10% for evaluation
            if i % 10 == 0:
                eval_anchors.append(triplet["query"])
                eval_positives.append(triplet["positive"])
                eval_negatives.append(triplet["negative"])
            else:
                train_examples.append(example)

        # Create data loader
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=self.batch_size
        )

        # Define loss function
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # Create evaluator
        evaluator = None
        if eval_anchors:
            evaluator = TripletEvaluator(
                anchors=eval_anchors,
                positives=eval_positives,
                negatives=eval_negatives,
                name="domain_eval",
            )

        # Fine-tune
        save_path = output_path or self.output_path
        Path(save_path).mkdir(parents=True, exist_ok=True)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            evaluator=evaluator,
            evaluation_steps=100,
            output_path=save_path,
            optimizer_params={"lr": self.learning_rate},
            show_progress_bar=True,
        )

        logger.info(f"Fine-tuned model saved to {save_path}")
        return save_path

    def generate_synthetic_triplets(
        self, documents: list[dict], num_triplets: int = 500
    ) -> list[dict]:
        """
        Generate synthetic training triplets from document chunks
        using in-batch negatives strategy.

        This creates training data without requiring manual annotation
        by treating chunks from the same section as positives and
        chunks from different sections as negatives.
        """
        import random

        triplets = []

        # Group chunks by section/source
        section_groups = {}
        for doc in documents:
            section = doc.get("metadata", {}).get("section", "unknown")
            source = doc.get("metadata", {}).get("source", "unknown")
            key = f"{source}_{section}"
            if key not in section_groups:
                section_groups[key] = []
            section_groups[key].append(doc["content"])

        groups = list(section_groups.values())
        if len(groups) < 2:
            logger.warning("Need at least 2 document groups for triplet generation")
            return []

        for _ in range(num_triplets):
            # Select a positive group
            pos_group_idx = random.randint(0, len(groups) - 1)
            pos_group = groups[pos_group_idx]

            if len(pos_group) < 2:
                continue

            # Select anchor and positive from same group
            anchor_idx, pos_idx = random.sample(range(len(pos_group)), 2)

            # Select negative from a different group
            neg_group_idx = random.choice(
                [i for i in range(len(groups)) if i != pos_group_idx]
            )
            neg_group = groups[neg_group_idx]
            neg_idx = random.randint(0, len(neg_group) - 1)

            triplets.append(
                {
                    "query": pos_group[anchor_idx][:200],  # Truncate as pseudo-query
                    "positive": pos_group[pos_idx],
                    "negative": neg_group[neg_idx],
                }
            )

        logger.info(f"Generated {len(triplets)} synthetic triplets")
        return triplets

    def save_triplets(self, triplets: list[dict], path: str):
        """Save training triplets to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(triplets, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(triplets)} triplets to {path}")
