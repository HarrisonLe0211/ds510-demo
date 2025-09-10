\
import os
import glob
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from train import train_or_load, build_product_index, score_product

load_dotenv()

app = Flask(__name__)
CORS(app)

DATA_DIR = os.getenv("DATA_DIR", "./data")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/sentiment_linear_svm.joblib")

# Train or load model and build product index at startup
model, vectorizer, label_names = train_or_load(DATA_DIR, MODEL_PATH)
product_index = build_product_index(DATA_DIR)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/api/products/search", methods=["GET"])
def product_search():
    q = (request.args.get("q") or "").strip().lower()
    if not q:
        return jsonify([])
    # simple case-insensitive contains search
    matches = []
    for name, info in product_index["name_to_info"].items():
        if q in name.lower():
            matches.append({
                "product_name": info["product_name"],
                "product_id": info["product_id"],
                "brand": info.get("brand"),
            })
            if len(matches) >= 25:
                break
    return jsonify(matches)

@app.route("/api/score", methods=["GET"])
def api_score():
    product_name = (request.args.get("product_name") or "").strip()
    if not product_name:
        return jsonify({"error": "Missing product_name"}), 400
    
    res = score_product(
        product_name=product_name,
        product_index=product_index,
        model=model,
        vectorizer=vectorizer,
        label_names=label_names,
        data_dir=DATA_DIR
    )
    if res is None:
        return jsonify({"error": f"No reviews found for '{product_name}'"}), 404
    return jsonify(res)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=True)
