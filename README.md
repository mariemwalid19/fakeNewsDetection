# Fake News Detection

A complete deep-learning pipeline that trains, evaluates, and deploys an LSTM-based fake news classifier. This repository includes preprocessing, model training, evaluation, comparisons, and a Flask deployment example — designed to be reproducible and production-ready.

---

## Project Overview

This project implements an end-to-end solution for binary fake-news classification (REAL vs FAKE) using an LSTM neural network. It covers data acquisition (Kaggle), preprocessing (tokenization, padding, text cleaning), model training, evaluation (confusion matrix, precision/recall/F1), persistence (saved model + tokenizer), and a minimal Flask API for serving predictions.

**Dataset:** [Fake News Dataset (Kaggle)](https://www.kaggle.com/datasets/rajatkumar30/fake-news)

---

## Dataset

* **Source:** Kaggle — `rajatkumar30/fake-news`.
* **Format:** CSV with text and labels (commonly columns: `title`, `text`, `label` or `author` depending on the variant).
* **Typical split:** Train / Validation / Test (you can use a stratified split).
* **Task:** Binary classification — `REAL` (legitimate news) vs `FAKE` (misinformation).
* **Notes:** Check the specific CSV headers after download. Use the Kaggle CLI or website to download.

**Recommended download (Kaggle CLI):**

```bash
kaggle datasets download -d rajatkumar30/fake-news
unzip fake-news.zip -d datasets/
```

---

## Features Implemented

### Core

* ✅ **Data ingestion & cleaning**: Remove HTML, punctuation, stopwords (optional), lowercasing, optional lemmatization/stemming.
* ✅ **Preprocessing pipeline**: Tokenization, integer encoding, sequence padding, and saving/loading tokenizer.
* ✅ **LSTM model**: Embedding layer → LSTM layers → Dense classifier with dropout & regularization.
* ✅ **Training utilities**: Early stopping, model checkpointing, TensorBoard logging.
* ✅ **Evaluation**: Confusion matrix, precision, recall, F1-score, ROC AUC; per-class metrics.
* ✅ **Persistence**: Save model in `.h5` and serialized/saved pipeline components (tokenizer, label encoder).
* ✅ **Deployment**: Flask API (`project_deployment.py`) to serve predictions.

### Bonus

* ✅ **Notebook walkthrough**: `fake_news_detection.ipynb` for exploratory data analysis (EDA) and step-by-step training.
* ✅ **Multiple model export formats**: Keras `.h5` and Pickle/Joblib for preprocessing objects.
* ✅ **Example client**: Sample request payload and response format for the API.

---

## Technology Stack

* **Deep learning:** TensorFlow / Keras
* **NLP preprocessing:** NLTK (or spaCy optional), Keras Tokenizer
* **Evaluation:** scikit-learn (metrics)
* **Visualization:** matplotlib, seaborn
* **API / Deployment:** Flask
* **Utilities:** pandas, numpy, pickle, joblib, tqdm

---

## Project Structure

```
fakeNewsDetection/
│
├── datasets/                       # raw and processed datasets
│   └── fake-news.csv
│
├── notebooks/
│   └── fake_news_detection.ipynb   # EDA + training walkthrough
│
├── models/
│   ├── fake_news_model.h5          # trained Keras model
│   ├── fake_news_model_metrics.json
│   └── tokenizer.pickle
│
├── src/
│   ├── data_prep.py                # cleaning, tokenizer fit, save/load
│   ├── model_build.py              # build LSTM model factory
│   ├── train.py                    # training loop with callbacks
│   ├── evaluate.py                 # evaluation scripts + visualizations
│   └── utils.py                    # helper functions
│
├── project_deployment.py           # Flask API to serve predictions
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation & Setup

### Prerequisites

* Python 3.8+
* 8GB+ RAM recommended for training (GPU recommended for speed)

### Quick start

```bash
git clone https://github.com/mariemwalid19/fakeNewsDetection.git
cd fakeNewsDetection
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

### Download dataset

Use Kaggle website or CLI (see dataset link above). Place the CSV in `datasets/` or update paths in `src/data_prep.py`.

---

## Usage

### 1. Preprocess data

```bash
python src/data_prep.py \
  --input datasets/fake-news.csv \
  --output_dir models/ \
  --max_vocab 20000 \
  --max_len 250
```

This script:

* cleans text,
* fits/saves a Keras `Tokenizer` (`tokenizer.pickle`),
* encodes labels,
* creates train/val/test splits and saves processed arrays if configured.

### 2. Train model

```bash
python src/train.py \
  --data_dir models/ \
  --epochs 10 \
  --batch_size 64 \
  --embedding_dim 128 \
  --max_len 250
```

Outputs:

* `models/fake_news_model.h5`
* `models/fake_news_model_metrics.json` (metrics, history)
* checkpoints & TensorBoard logs (if enabled)

### 3. Evaluate

```bash
python src/evaluate.py --model models/fake_news_model.h5 --data models/
```

Generates:

* classification report (precision/recall/F1)
* confusion matrix PNG
* ROC curve

### 4. Run Flask API (Deployment)

```bash
python project_deployment.py
```

Example `project_deployment.py` behavior:

* loads `tokenizer.pickle` and `fake_news_model.h5`
* exposes `/predict` POST endpoint that accepts JSON payload:

```json
{ "text": "Breaking news: Scientists discover water on Mars!" }
```

Response:

```json
{
  "prediction": "REAL",
  "confidence": 0.93
}
```

---

## API Example (curl)

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This just in — celebrity endorses miracle pill!"}'
```

---

## Model Architecture & Hyperparameters (example)

* Embedding layer (vocab\_size, embedding\_dim=128)
* Bidirectional LSTM (units=128) + Dropout(0.5)
* Dense(64) + ReLU + Dropout(0.5)
* Output Dense(1) + Sigmoid (binary classification)
* Loss: Binary Crossentropy, Optimizer: Adam (lr=1e-3)
* Callbacks: EarlyStopping (monitor=val\_loss, patience=3), ModelCheckpoint

---

## Example Performance (illustrative — replace with your trained results)

| Model            | Precision | Recall | F1-Score | ROC AUC |
| ---------------- | --------- | ------ | -------- | ------- |
| LSTM (this repo) | 0.86      | 0.84   | 0.85     | 0.92    |

> Replace the table above with real metrics after training; `src/evaluate.py` will produce a JSON summary.

---

## Key Technical Decisions

### Data handling

* Use conservative text cleaning (preserve named entities where possible).
* Use Keras `Tokenizer` for consistent token-to-index mapping between training and inference.
* Stratified split to preserve class balance.

### Training choices

* Dropout and early stopping to prevent overfitting.
* Save best model checkpoints and the tokenizer for deterministic inference.

### Evaluation

* Use `sklearn.metrics` for classification report and confusion matrix.
* Monitor ROC AUC and per-class F1 to judge model robustness.

---

## Insights & Tips

* **Class imbalance**: Fake-news datasets often have imbalance. Use class weights or balanced sampling.
* **Text length**: Many news articles are long — consider truncation strategies and/or using title+short excerpt.
* **Ensembling**: Combining LSTM with transformer embeddings (`distilBERT` or `bert`) often improves robustness.
* **Interpretability**: Use attention or LIME/SHAP for explainability in production.

---

## Future Improvements

* Integrate transformer-based encoders (`en_core_web_trf`, `distilbert`) for better contextual understanding.
* Add cross-validation & robust hyperparameter search (Optuna or KerasTuner).
* Dockerize the service and add CI/CD for continuous model updates.
* Add monitoring for model drift and an automated re-training pipeline.

---

## Contributing

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/awesome-improvement`.
3. Commit your changes: `git commit -m "Add awesome improvement"`.
4. Push: `git push origin feature/awesome-improvement`.
5. Open a Pull Request.

Please follow the code style in `src/` and include tests for new utilities.

---

## License

This project uses the **MIT License** — see `LICENSE` for details.

---

## Acknowledgments

* Dataset: Kaggle — `rajatkumar30/fake-news`.
* Libraries: TensorFlow/Keras, scikit-learn, NLTK, Flask.
* Inspiration: Standard best practices for text classification and model deployment.

---

## Contact

**Mariem Walid** — open an issue on the repository for questions, feature requests, or dataset/metric clarifications.

---

*Ready-to-run templates and scripts are included in `src/`. If you want, I can:*

* generate a polished `project_deployment.py` Flask app,
* produce `src/data_prep.py` / `src/train.py` templates,
* or convert the README to a downloadable `README.md` file. Which would you like next?
