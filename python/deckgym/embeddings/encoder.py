"""
Embedding encoder using SentenceTransformer + PCA.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import EMBEDDING_DIM, SENTENCE_MODEL, PATHS
from .text_cleaner import TextCleaner


class EmbeddingEncoder:
    """Encodes texts to fixed-dimension embeddings using SentenceTransformer + PCA."""

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        model_name: str = SENTENCE_MODEL,
        device: str = "cpu",
    ):
        """
        Initialize encoder.

        Args:
            embedding_dim: Output embedding dimension after PCA
            model_name: SentenceTransformer model name
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.device = device

        self._model = None
        self._pca = None
        self._text_to_embedding: Dict[str, np.ndarray] = {}

    def _load_model(self):
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)

    def fit_transform(
        self,
        texts: List[str],
        cleaner: TextCleaner,
    ) -> Dict[str, Dict]:
        """
        Fit PCA on texts and transform them to embeddings.

        Args:
            texts: List of raw texts to encode
            cleaner: TextCleaner instance for text normalization

        Returns:
            Dict mapping cleaned text -> {embedding, type_refs, mech_refs}
        """
        from sklearn.decomposition import PCA

        self._load_model()

        # Clean texts
        cleaned_texts = [cleaner.clean(t) for t in texts]

        # Encode with SentenceTransformer
        print(f"Encoding {len(texts)} texts with {self.model_name}...")
        raw_embeddings = self._model.encode(cleaned_texts, show_progress_bar=True)

        # Fit PCA
        n_components = min(self.embedding_dim, len(texts), raw_embeddings.shape[1])
        print(f"Fitting PCA: {raw_embeddings.shape[1]} -> {n_components} dimensions...")

        self._pca = PCA(n_components=n_components)
        reduced = self._pca.fit_transform(raw_embeddings)

        # Pad if needed
        if reduced.shape[1] < self.embedding_dim:
            padding = np.zeros(
                (reduced.shape[0], self.embedding_dim - reduced.shape[1])
            )
            reduced = np.hstack([reduced, padding])

        variance = np.sum(self._pca.explained_variance_ratio_)
        print(f"Explained variance: {variance:.2%}")

        # Build mapping
        result = {}
        for i, (original, cleaned) in enumerate(zip(texts, cleaned_texts)):
            if cleaned not in result:
                result[cleaned] = {
                    "embedding": reduced[i].tolist(),
                    "type_refs": TextCleaner.get_type_references(original),
                    "mech_refs": TextCleaner.get_mechanic_references(original),
                }

        self._text_to_embedding = result
        return result

    def transform(self, texts: List[str], cleaner: TextCleaner) -> Dict[str, Dict]:
        """
        Transform texts using pre-fitted PCA.

        Args:
            texts: List of raw texts to encode
            cleaner: TextCleaner instance

        Returns:
            Dict mapping cleaned text -> {embedding, type_refs, mech_refs}
        """
        if self._pca is None:
            raise ValueError(
                "PCA not fitted. Call fit_transform first or load a saved model."
            )

        self._load_model()

        cleaned_texts = [cleaner.clean(t) for t in texts]
        raw_embeddings = self._model.encode(cleaned_texts, show_progress_bar=False)
        reduced = self._pca.transform(raw_embeddings)

        if reduced.shape[1] < self.embedding_dim:
            padding = np.zeros(
                (reduced.shape[0], self.embedding_dim - reduced.shape[1])
            )
            reduced = np.hstack([reduced, padding])

        result = {}
        for i, (original, cleaned) in enumerate(zip(texts, cleaned_texts)):
            if cleaned not in result:
                result[cleaned] = {
                    "embedding": reduced[i].tolist(),
                    "type_refs": TextCleaner.get_type_references(original),
                    "mech_refs": TextCleaner.get_mechanic_references(original),
                }

        return result

    def get_embedding(self, cleaned_text: str) -> Optional[Dict]:
        """Get embedding for a cleaned text if available."""
        return self._text_to_embedding.get(cleaned_text)

    def save_pca(self, path: Path = None):
        """Save PCA model to disk."""
        path = path or PATHS["pca_model"]
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self._pca, f)
        print(f"Saved PCA model to {path}")

    def load_pca(self, path: Path = None):
        """Load PCA model from disk."""
        path = path or PATHS["pca_model"]

        with open(path, "rb") as f:
            self._pca = pickle.load(f)
        print(f"Loaded PCA model from {path}")
