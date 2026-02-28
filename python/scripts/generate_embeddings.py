#!/usr/bin/env python3
"""
Generate text embeddings for Pokemon TCG cards.

This script trains embeddings on the full TCG database and applies them
to the Pocket subset, generating card_features.json for the Rust observation.

Usage:
    python scripts/generate_embeddings.py
    
Or as a module:
    python -m deckgym.embeddings.generator
"""

import typer
from deckgym.embeddings import generate_all


def main():
    """Generate text embeddings for all Pokemon TCG Pocket cards."""
    generate_all()


if __name__ == "__main__":
    typer.run(main)
