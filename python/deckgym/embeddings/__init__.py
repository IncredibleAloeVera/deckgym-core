"""
Embedding module for DeckGym.

Trains text embeddings on the full TCG database and applies them to the Pocket subset.
"""

from .config import EMBEDDING_DIM, SENTENCE_MODEL, PATHS
from .text_cleaner import TextCleaner
from .data_loader import TCGDataLoader
from .encoder import EmbeddingEncoder
from .pocket_mapper import PocketMapper
from .generator import generate_all

__all__ = [
    "EMBEDDING_DIM",
    "SENTENCE_MODEL",
    "PATHS",
    "TextCleaner",
    "TCGDataLoader",
    "EmbeddingEncoder",
    "PocketMapper",
    "generate_all",
]
