from pathlib import Path

SEED = 67

ROOT = Path(__file__).resolve().parent.parent   # Repo root
FIGURES_PATH = ROOT / "Figures"
RESULTS_PATH = ROOT / "Results"
DATA_PATH = ROOT / "Data" / "mnist.npz"

FIGURES_PATH.mkdir(exist_ok=True)
RESULTS_PATH.mkdir(exist_ok=True)