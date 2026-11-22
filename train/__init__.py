"""Training the three networks on a PC (PyTorch; a CUDA GPU for the full runs), in the Spectrum's arithmetic.

Data and runs live outside the code: data/ and runs/ at the top of the repo, or wherever SPECTRUM_LM_DATA and
SPECTRUM_LM_RUNS point.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SPECTRUM_LM_DATA", ROOT / "data"))
RUNS = Path(os.environ.get("SPECTRUM_LM_RUNS", ROOT / "runs"))
