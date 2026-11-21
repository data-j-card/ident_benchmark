from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
DATOS = REPO / "datos"
RESULTADOS = REPO / "resultados"

BANCO = CONFIGS / "banco_tufts.csv"
TUFTS = DATOS / "tufts"
EMBEDDINGS = DATOS / "embeddings" / "antelopev2"
CALIBRACION = RESULTADOS / "calibracion"
