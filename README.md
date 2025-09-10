# Beauty Product Sentiment (Linear SVM)

A minimal React + Flask app that trains a **LinearSVM**-based sentiment classifier on your Sephora review data and estimates how **favorable** a product is (percentage of positive sentiment across its reviews).

- **Frontend**: React (Vite)
- **Backend**: Flask API
- **Model**: `TfidfVectorizer` + `LinearSVC` wrapped in `CalibratedClassifierCV` (probabilities)
- **Runs in GitHub Codespaces** via `.devcontainer/`

## Project Structure
```
beauty-sentiment-app/
  backend/
    app.py                 # Flask API
    train.py               # Data loading, training, scoring
    requirements.txt
    .env.example
  frontend/
    package.json
    vite.config.js
    index.html
    src/
      main.jsx
      App.jsx
  .devcontainer/
    devcontainer.json
    setup.sh
  run.sh
```

## Data Layout

Place your CSVs under `backend/data/`. This repo expects:
- `product_info.csv` and/or `product_info_skincare.csv`
- `reviews_*.csv` (e.g., `reviews_0-250_masked.csv`, etc.)

Columns are auto-detected:
- Reviews: looks for a text column among `review_text|text|review|content|body`.
- Product id: `product_id|item_id|pid|sku` (optional).
- Labels: uses `sentiment` or `label` if present, else derives from `rating >= 4` ⇒ positive.

## Running (Locally or Codespaces)

### 1) Put data
Copy your CSVs into `backend/data/` (create the folder if needed). In Codespaces, drag/drop or use git.

### 2) Start backend (Flask)
```bash
bash run.sh
```
This will create a Python venv, install dependencies, copy `.env.example` → `.env` if missing, then run `backend/app.py`.
On first start, the model will train automatically and save to `backend/models/sentiment_linear_svm.joblib`.

### 3) Start frontend (React)
Open a second terminal:
```bash
cd frontend
npm run dev
```
Vite dev server runs on **5173** and proxies `/api` to the Flask API (**8000**).

### 4) Use the app
Visit the forwarded port (5173). Type a product name and press **Analyze**.
- The API looks up reviews by matching `product_id` or (fallback) by searching the product name in review text.
- It computes the **average probability of positive sentiment** across the product's reviews and returns:
  - `favorable_percent` (0–100),
  - `verdict` (Favorable / Mixed / Unfavorable),
  - `reviews_count` used,
  - a few `sample_reviews`.

## Notes
- LinearSVC does not directly provide probabilities, so we wrap it with `CalibratedClassifierCV` (Platt scaling).
- If you want to re-train, delete `backend/models/sentiment_linear_svm.joblib` and restart the backend.
- You can tweak vectorizer/thresholds in `backend/train.py`.

## API
- `GET /api/health` → `{ ok: true }`
- `GET /api/products/search?q=<text>` → top name matches from product catalog
- `GET /api/score?product_name=<name>` → favorability for that product

Enjoy!
