import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ----------------------------- Data Loading -----------------------------

def _load_all_reviews(data_dir: str) -> pd.DataFrame:
    """
    Load all review CSVs from data_dir, normalize columns, and return a DataFrame.

    Expected columns (flexible):
      - Text: one of {'review_text','text','review','content','body'}
      - Product ID: one of {'product_id','item_id','pid','sku','productId','ProductId','PRODUCT_ID'}
      - Label: 'sentiment' or 'label' if provided; otherwise derived from rating (>=4 -> 1 else 0)
    """
    # Robust globbing (accepts both "reviews*_masked.csv" and "reviews_*_masked.csv", etc.)
    patterns = [
        "reviews*_masked.csv",
        "reviews_*_masked.csv",
        "reviews*.csv",
        "reviews_*.csv",
    ]
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(os.path.join(data_dir, pat))))

    if not paths:
        raise FileNotFoundError("No reviews CSVs found in data_dir.")

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__src__"] = os.path.basename(p)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")

    if not frames:
        raise FileNotFoundError("No readable review CSVs after scanning files.")

    df = pd.concat(frames, ignore_index=True)

    # --- Text column normalization ---
    text_col = None
    for c in ["review_text", "text", "review", "content", "body"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError("Could not find a text column in review CSVs.")
    df.rename(columns={text_col: "review_text"}, inplace=True)

    # --- Product ID normalization (cast to string for safe joins) ---
    pid_col = None
    for c in ["product_id", "item_id", "pid", "sku", "productId", "ProductId", "PRODUCT_ID"]:
        if c in df.columns:
            pid_col = c
            break
    if pid_col is None:
        df["product_id"] = np.nan
    else:
        df.rename(columns={pid_col: "product_id"}, inplace=True)
        df["product_id"] = df["product_id"].astype(str).str.strip()

    # --- Label normalization ---
    if "sentiment" in df.columns:
        df["label"] = df["sentiment"]
    elif "label" in df.columns:
        pass  # already provided
    else:
        if "rating" in df.columns:
            df["label"] = (df["rating"].astype(float) >= 4).astype(int)
        else:
            print("[INFO] No 'sentiment'/'label'/'rating' found. Defaulting all to positive (1).")
            df["label"] = 1

    # --- Text cleaning ---
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.len() > 0]

    # --- Diagnostics ---
    try:
        uniq_pid = df["product_id"].nunique() if "product_id" in df.columns else "N/A"
    except Exception:
        uniq_pid = "N/A"
    print(f"[INFO] Reviews loaded: {len(df)} | cols={list(df.columns)} | unique product_id={uniq_pid}")

    return df


def _load_products(data_dir: str) -> pd.DataFrame:
    """
    Load product info from product_info.csv / product_info_skincare.csv (if present),
    normalize columns, and return a DataFrame with ['product_id','product_name','brand'].
    """
    frames = []
    for name in ["product_info.csv", "product_info_skincare.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            try:
                frames.append(pd.read_csv(p))
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")

    if frames:
        prod = pd.concat(frames, ignore_index=True).drop_duplicates()
    else:
        prod = pd.DataFrame(columns=["product_id", "product_name", "brand"])

    # Normalize column names
    if "product_name" not in prod.columns:
        for c in ["name", "title"]:
            if c in prod.columns:
                prod.rename(columns={c: "product_name"}, inplace=True)
                break
    if "product_id" not in prod.columns:
        for c in ["id", "pid", "sku", "productId", "ProductId", "PRODUCT_ID"]:
            if c in prod.columns:
                prod.rename(columns={c: "product_id"}, inplace=True)
                break
    if "brand" not in prod.columns:
        prod["brand"] = None

    # Normalize types
    if "product_id" in prod.columns:
        prod["product_id"] = prod["product_id"].astype(str).str.strip()
    prod["product_name"] = prod.get("product_name", pd.Series(dtype=str)).astype(str)

    return prod


# ----------------------------- Training / Loading -----------------------------

def train_or_load(data_dir: str, model_path: str):
    """
    Train the model if model_path does not exist, otherwise load it.
    Returns (model, vectorizer, label_names).
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
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        strip_accents="unicode"
    )
    base = LinearSVC()
    model = CalibratedClassifierCV(base, cv=5)  # keep probabilities via Platt scaling

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model.fit(X_train_vec, y_train)

    try:
        y_hat = model.predict(X_val_vec)
        print(classification_report(y_val, y_hat, digits=3))
    except Exception as e:
        print("[WARN] Could not print classification_report:", e)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {"model": model, "vectorizer": vectorizer, "label_names": ["negative", "positive"]},
        model_path
    )
    print(f"[INFO] Saved model to {model_path}")
    return model, vectorizer, ["negative", "positive"]


# ----------------------------- Product Indexing -----------------------------

def build_product_index(data_dir: str) -> Dict[str, Any]:
    products = _load_products(data_dir)
    name_to_info: Dict[str, Dict[str, Any]] = {}
    id_to_name: Dict[str, str] = {}

    for _, row in products.iterrows():
        name = str(row.get("product_name", "")).strip()
        if not name:
            continue
        pid = row.get("product_id")
        pid_s = str(pid).strip() if pid is not None else ""

        info = {
            "product_name": name,
            "product_id": pid_s if pid_s else None,
            "brand": row.get("brand")
        }
        name_to_info[name] = info
        if pid_s:
            id_to_name[pid_s] = name

    return {"name_to_info": name_to_info, "id_to_name": id_to_name}


# ----------------------------- Matching Helpers -----------------------------

def _word_boundary_contains(series: pd.Series, phrase: str) -> pd.Series:
    """Case-insensitive word-boundary match for the whole phrase."""
    phrase = re.escape(phrase)
    pattern = rf"\b{phrase}\b"
    return series.str.contains(pattern, case=False, na=False, regex=True)


def _token_contains(series: pd.Series, name: str) -> pd.Series:
    """
    Token-based fallback: require at least two 'significant' tokens (len>=3) from
    the product name to be present in the text.
    """
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", name) if len(t) >= 3]
    if not tokens:
        return series.str.contains(re.escape(name), case=False, na=False)

    hits = None
    for t in tokens:
        hit = series.str.contains(rf"\b{re.escape(t)}\b", case=False, na=False, regex=True)
        hits = hit if hits is None else (hits.astype(int) + hit.astype(int))

    need = 2 if len(tokens) >= 2 else 1
    return hits.fillna(0).astype(int).ge(need)


# ----------------------------- Scoring -----------------------------

def _load_reviews_for_product(
    product_name: str,
    product_index: Dict[str, Any],
    data_dir: str
) -> pd.DataFrame:
    df = _load_all_reviews(data_dir)

    # 1) Try exact product_id match if we have it
    info = product_index["name_to_info"].get(product_name)
    if info:
        pid = info.get("product_id")
        if pid is not None and str(pid).strip() != "" and "product_id" in df.columns:
            pid = str(pid).strip()
            sub = df[df["product_id"].astype(str).str.strip() == pid]
            if len(sub) > 0:
                return sub

    # 2) Word-boundary phrase match in review text
    by_bound = _word_boundary_contains(df["review_text"].astype(str), product_name)
    sub = df[by_bound]
    if len(sub) > 0:
        return sub

    # 3) Token-based fallback (>=2 significant tokens)
    by_tokens = _token_contains(df["review_text"].astype(str), product_name)
    sub = df[by_tokens]
    return sub


def score_product(
    product_name: str,
    product_index: Dict[str, Any],
    model,
    vectorizer,
    label_names,
    data_dir: str
) -> Optional[Dict[str, Any]]:
    reviews_df = _load_reviews_for_product(product_name, product_index, data_dir)
    if reviews_df is None or len(reviews_df) == 0:
        print(f"[DEBUG] No reviews via ID or text match for: {product_name}")
        print(f"[DEBUG] Known products in index: {len(product_index['name_to_info'])}")
        return None

    texts = reviews_df["review_text"].astype(str).tolist()
    X = vectorizer.transform(texts)

    # Probability for positive class (index 1)
    probs = model.predict_proba(X)[:, 1]
    avg_prob = float(probs.mean())
    pos_pct = round(avg_prob * 100, 2)
    n_reviews = len(texts)

    verdict = "Favorable" if avg_prob >= 0.6 else ("Mixed" if avg_prob >= 0.4 else "Unfavorable")

    return {
        "product_name": product_name,
        "reviews_count": n_reviews,
        "favorable_probability": avg_prob,
        "favorable_percent": pos_pct,
        "verdict": verdict,
        "sample_reviews": texts[:5],
    }