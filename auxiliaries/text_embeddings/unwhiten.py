"""Rebuild the pre-`de006bf0` unwhitened table from the whitened one, for ablations only.

`long_v4` trained against `project(emb, components, mean) = (emb - mean) @ components.T` — no
scale. `de006bf0` divided that by `whitening_scale`, and stored the divisor in `pca.npz` as part of
the frozen basis. So the old artifact is the current one multiplied back, exactly: no encoder, no
corpus, no network, and no chance of a re-embedding drifting from what v4 actually read.

This is deliberately **not** a flag on `build_embeddings.py`. That script writes the artifact the
runs consume, and giving it a switch that produces a v2-schema table under the v3 name is how a run
ends up training on features `OBS_SCHEMA_VERSION` was raised to refuse. The output here carries its
own name and nothing reads it unless asked.

    uv run --no-project --with numpy auxiliaries/text_embeddings/unwhiten.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "out"


def main() -> None:
    table = json.loads((OUT_DIR / "text_embeddings.json").read_text(encoding="utf-8"))
    pca = np.load(OUT_DIR / "pca.npz")
    scales = {"effect": pca["effect_scale"], "ability": pca["ability_scale"]}

    out = {}
    for block, scale in scales.items():
        rows = table[block]
        matrix = np.array(list(rows.values()), dtype=np.float32)
        if matrix.shape[1] != len(scale):
            raise SystemExit(
                f"{block}: table is {matrix.shape[1]} wide, the stored scale is {len(scale)} — "
                "these are not the same artifact"
            )
        raw = matrix * scale
        # The energies the whitening commit measured, and the check that this inverted the right
        # transform: ~0.29 for effect against the ~0.97 the whitened block carries.
        print(
            f"{block}: block energy {np.mean((raw**2).sum(1)):.3f} "
            f"(whitened {np.mean((matrix**2).sum(1)):.3f})"
        )
        out[block] = {
            text: [round(float(v), 6) for v in row] for text, row in zip(rows, raw)
        }

    path = OUT_DIR / "text_embeddings_unwhitened.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
