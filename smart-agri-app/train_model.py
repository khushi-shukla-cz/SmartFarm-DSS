import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Crop_recommendation.csv"

FEATURE_COLUMNS = [
    "Nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
TARGET_COLUMN = "label"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    # Remove empty/unnamed columns caused by trailing commas in CSV rows.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")

    return df


def train_and_save(df: pd.DataFrame) -> None:
    x = df[FEATURE_COLUMNS].astype(float)
    y_raw = df[TARGET_COLUMN].astype(str)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x_scaled, y)

    with open(BASE_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(BASE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(BASE_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)


def main() -> None:
    df = load_dataset(DATA_PATH)
    train_and_save(df)
    print("Model training complete. Saved model.pkl, scaler.pkl, and label_encoder.pkl")


if __name__ == "__main__":
    main()
