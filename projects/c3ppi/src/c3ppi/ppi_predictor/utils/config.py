import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MODEL_REGISTRY = os.path.join(ROOT, "registry")
OUTPUT = os.path.join(ROOT, "output")

if not os.path.exists(MODEL_REGISTRY):
    os.makedirs(MODEL_REGISTRY)

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)