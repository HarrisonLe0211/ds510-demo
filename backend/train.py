\
import os
import glob
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ------------- Utilities to load data -------------
def _load_all_reviews(data_dir: str) -> pd.DataFrame:
    """
    Load all review CSVs. Expected columns (flexible):
      - 'review_text' (or 'review', 'text')
      - 'rating' (optional; if present we may derive labels)
      - 'product_id' (or 'item_id')
      - 'sentiment' (optional; if present we use it directly)
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "reviews*_masked.csv"))) + \
            sorted(glob.glob(os.path.join(data_dir, "reviews*.csv")))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__src__"] = os.path.basename(p)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    if not frames:
        raise FileNotFoundError("No reviews CSVs found in data_dir.")
    df = pd.concat(frames, ignore_index=True)
    # normalize columns
    # text
    text_col = None
    for c in ["review_text", "text", "review", "content", "body"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError("Could not find a text column in review CSVs.")
    df.rename(columns={text_col: "review_text"}, inplace=True)

    # product_id
    pid_col = None
    for c in ["product_id", "item_id", "pid", "sku"]:
        if c in df.columns:
            pid_col = c
            break
    if pid_col is None:
        # allow missing product_id; we'll rely on product_name matching later if available
        df["product_id"] = np.nan
    else:
        df.rename(columns={pid_col: "product_id"}, inplace=True)

    # sentiment label if exists
    if "sentiment" in df.columns:
        # assume either string labels or {0,1}
        df["label"] = df["sentiment"]
    elif "label" in df.columns:
        pass  # already good
    else:
        # derive from rating if exists (>=4 => positive else negative)
        if "rating" in df.columns:
            df["label"] = (df["rating"].astype(float) >= 4).astype(int)
        else:
            # fallback: neutral = positive for now (or drop)
            print("[INFO] No 'sentiment' or 'rating' found. Defaulting all to positive (1).")
            df["label"] = 1

    # clean text
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.len() > 0]
    return df

def _load_products(data_dir: str) -> pd.DataFrame:
    paths = []
    for name in ["product_info.csv", "product_info_skincare.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            paths.append(p)
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    if frames:
        prod = pd.concat(frames, ignore_index=True).drop_duplicates()
    else:
        prod = pd.DataFrame(columns=["product_id", "product_name", "brand"])
    # normalize columns
    if "product_name" not in prod.columns:
        # try find a likely column
        for c in ["name", "title"]:
            if c in prod.columns:
                prod.rename(columns={c: "product_name"}, inplace=True)
                break
    if "product_id" not in prod.columns:
        for c in ["id", "pid", "sku"]:
            if c in prod.columns:
                prod.rename(columns={c: "product_id"}, inplace=True)
                break
    if "brand" not in prod.columns:
        prod["brand"] = None
    prod["product_name"] = prod.get("product_name", pd.Series(dtype=str)).astype(str)
    return prod

# ------------- Model training / loading -------------
def train_or_load(data_dir: str, model_path: str):
    """
    Train the model if model_path does not exist, otherwise load it.
    Returns (model, vectorizer, label_names)
    """
    if os.path.exists(model_path):
        print(f"[INFO] Loading model from {model_path}")
        bundle = joblib.load(model_path)
        return bundle["model"], bundle["vectorizer"], bundle["label_names"]

    print("[INFO] Training new model...")
    df = _load_all_reviews(data_dir)
    X = df["review_text"].values
    y = df["label"].values

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        min_df=3,
        max_df=0.9,
        strip_accents="unicode"
    )
    base = LinearSVC()
    model = CalibratedClassifierCV(base, cv=5)  # Platt scaling for probabilities

    # Fit
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    model.fit(X_train_vec, y_train)

    # Eval (printed on server logs)
    y_hat = model.predict(X_val_vec)
    try:
        print(classification_report(y_val, y_hat, digits=3))
    except Exception as e:
        print("[WARN] Could not print classification_report:", e)

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer, "label_names": ["negative", "positive"]}, model_path)
    print(f"[INFO] Saved model to {model_path}")
    return model, vectorizer, ["negative", "positive"]

# ------------- Product indexing and scoring -------------
def build_product_index(data_dir: str) -> Dict[str, Any]:
    products = _load_products(data_dir)
    name_to_info = {}
    id_to_name = {}
    for _, row in products.iterrows():
        name = str(row.get("product_name", "")).strip()
        if not name:
            continue
        pid = row.get("product_id")
        info = {
            "product_name": name,
            "product_id": pid,
            "brand": row.get("brand")
        }
        name_to_info[name] = info
        if pd.notna(pid):
            id_to_name[pid] = name
    return {"name_to_info": name_to_info, "id_to_name": id_to_name}

def _load_reviews_for_product(product_name: str, product_index: Dict[str, Any], data_dir: str) -> pd.DataFrame:
    # match by exact name; if not found, try case-insensitive contains over product_info
    info = product_index["name_to_info"].get(product_name)
    pid = info["product_id"] if info else None

    df = _load_all_reviews(data_dir)
    if pid is not None and pd.notna(pid) and "product_id" in df.columns:
        sub = df[df["product_id"] == pid]
        if len(sub) > 0:
            return sub

    # fallback: product name may appear in text or we try fuzzy match via contains on name in info table
    # Here, simplest fallback: search reviews that mention product name
    sub = df[df["review_text"].str.contains(product_name, case=False, na=False)]
    return sub

def score_product(product_name: str, product_index: Dict[str, Any], model, vectorizer, label_names, data_dir: str) -> Optional[Dict[str, Any]]:
    reviews_df = _load_reviews_for_product(product_name, product_index, data_dir)
    if reviews_df is None or len(reviews_df) == 0:
        return None
    texts = reviews_df["review_text"].astype(str).tolist()
    X = vectorizer.transform(texts)
    # probability for positive class (index 1)
    probs = model.predict_proba(X)[:, 1]
    avg_prob = float(probs.mean())
    pos_pct = round(avg_prob * 100, 2)
    n_reviews = len(texts)

    # simple verdict
    verdict = "Favorable" if avg_prob >= 0.6 else ("Mixed" if avg_prob >= 0.4 else "Unfavorable")

    return {
        "product_name": product_name,
        "reviews_count": n_reviews,
        "favorable_probability": avg_prob,
        "favorable_percent": pos_pct,
        "verdict": verdict,
        "sample_reviews": texts[:5]  # small sample
    }
