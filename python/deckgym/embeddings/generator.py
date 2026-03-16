"""
Main generator pipeline for embeddings.
"""

import json
from pathlib import Path
from typing import Optional

from .config import EMBEDDING_DIM, PATHS
from .data_loader import TCGDataLoader
from .encoder import EmbeddingEncoder
from .pocket_mapper import PocketMapper
from .text_cleaner import TextCleaner


def generate_all(
    embedding_dim: int = EMBEDDING_DIM,
    raw_data_path: Optional[Path] = None,
    pocket_db_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Generate embeddings trained on full TCG and applied to Pocket subset.

    Args:
        embedding_dim: Output embedding dimension
        raw_data_path: Path to raw TCG data directory
        pocket_db_path: Path to Pocket database.json
        output_dir: Output directory for generated files
    """
    raw_data_path = raw_data_path or PATHS["raw_data"]
    pocket_db_path = pocket_db_path or PATHS["pocket_db"]
    output_dir = output_dir or PATHS["output_dir"]

    print("=" * 60)
    print("Embedding Generation Pipeline")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Raw data: {raw_data_path}")
    print(f"  - Pocket DB: {pocket_db_path}")
    print(f"  - Output: {output_dir}")
    print("=" * 60)

    # --- Step 1: Load full TCG database ---
    print("\n[1/5] Loading full TCG database...")
    loader = TCGDataLoader(raw_data_path).load()
    stats = loader.get_stats()
    print(
        f"  Loaded {stats['total_cards']} cards, {stats['unique_texts']} unique texts"
    )

    # --- Step 2: Create text cleaner with all names ---
    print("\n[2/5] Building text cleaner...")

    # Also load Pocket names to include them
    pocket_mapper = PocketMapper(pocket_db_path).load()
    all_names = loader.pokemon_names | pocket_mapper.pokemon_names

    cleaner = TextCleaner(all_names)
    print(f"  {len(all_names)} Pokemon names for normalization")

    # --- Step 3: Train embeddings on full TCG ---
    print("\n[3/5] Training embeddings on full TCG...")
    encoder = EmbeddingEncoder(embedding_dim=embedding_dim)

    # Add empty string for cards without effects
    all_texts = [""] + loader.unique_texts
    text_embeddings = encoder.fit_transform(all_texts, cleaner)

    # Save PCA model
    encoder.save_pca()

    # --- Step 4: Generate Pocket card features ---
    print("\n[4/5] Generating Pocket card features...")
    card_features = pocket_mapper.generate_card_features(
        text_embeddings, cleaner, embedding_dim
    )
    print(f"  Generated features for {len(card_features)} cards")

    # --- Step 5: Save outputs ---
    print("\n[5/5] Saving outputs...")
    pocket_mapper.save_features(card_features, output_dir / "card_features.json")

    # Save names for reference
    names_path = output_dir / "names.json"
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(all_names)), f, indent=2)
    print(f"  Saved names to {names_path}")

    print("\n" + "=" * 60)
    print("Done! Generated files in:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    generate_all()
