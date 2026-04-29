from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent

BASE_DIR = SRC_DIR.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Telco_customer_churn.xlsx"

MODEL_DIR = SRC_DIR / "saved_models"
MODEL_PATH = MODEL_DIR / "model.pt"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

MODEL_CARD_PATH = BASE_DIR / "docs" / "model_card.md"

# Definir um limiar para classificar como churn ou não churn
THRESHOLD = 0.5

# Número de épocas para treinamento do modelo
EPOCH = 100