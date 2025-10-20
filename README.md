# Fake News Detector — Streamlit App

This repository contains a Streamlit app (`app.py`) and accompanying Jupyter notebook (CapstoneProjectFakeNewsDetector_24thSept-final.ipynb) used to build and compare models for fake news detection. The app lets users paste news text and get a prediction (Fake / Real) using either a Logistic Regression model or a Feedforward Deep Learning model.

Contents
- `app.py` — Streamlit application.
- `CapstoneProjectFakeNewsDetector_24thSept-final.ipynb` — notebook with experiments, training code and comparison plots.
- `requirements.txt` — Python dependencies for Streamlit Cloud.

Notes on TensorFlow
- TensorFlow is included in `requirements.txt` as an optional dependency to enable the Feedforward DL model and LSTM experiments. TensorFlow greatly increases build time and the final image size on Streamlit Cloud. If you don't need the DL models on Streamlit Cloud, remove or comment out the `tensorflow` line in `requirements.txt` prior to deployment.

How to run locally
1. Create a Python virtual environment and activate it.
   - On Windows PowerShell:
     python -m venv .venv; .\.venv\Scripts\Activate.ps1
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   python -m streamlit run app.py

How to deploy to Streamlit Cloud
1. Push this project to a public GitHub repository.
2. Go to https://share.streamlit.io and connect your GitHub account.
3. Choose the repository and branch containing this code and deploy.

If you included TensorFlow in `requirements.txt`, expect a longer build. Consider removing it and pre-training the DL model locally, then serializing weights and loading them in `app.py`.

Troubleshooting
- If the app doesn't appear on http://localhost:8501 when running locally, run the Streamlit command shown above and check the terminal for binding messages or errors. If the local scripts folder is not on PATH, using `python -m streamlit run app.py` works reliably.
- If NLTK complains about missing packages at runtime (stopwords, wordnet), run the following once from Python:
  >>> import nltk
  >>> nltk.download('stopwords')
  >>> nltk.download('wordnet')

Contact
- If you want me to prepare a ready-to-deploy GitHub repo (I will create the repo structure, add these files, and optionally exclude TensorFlow) tell me and I'll create the remaining packaging changes and a git-ready folder.
# Fake News Detector Streamlit App

This Streamlit app wraps the Logistic Regression model from your notebook and provides a simple UI to classify user-input news text as Fake or Real.

Files added
- `app.py`: Streamlit app that downloads the original datasets, trains TF-IDF + LogisticRegression, and exposes a prediction UI.
- `requirements.txt`: Python dependencies.

How to run (PowerShell on Windows)

1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Streamlit app:

```powershell
streamlit run app.py
```

Notes
- The app attempts to download the two CSV files from Google Drive using the file IDs used in the original notebook. If the automatic download fails (network/GDrive permission), upload the `true` and `fake` CSVs using the sidebar file upload controls.
- On first run NLTK will download the `stopwords` and `wordnet` corpora. If your environment blocks downloads, run the following in Python once to install them manually:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

Limitations & next steps
- Training happens at app startup and may take a minute depending on machine CPU and dataset size. For production, consider training offline and loading serialized models with `joblib`.
- For reproducibility, you can save the fitted `vectorizer` and `model` to disk and load them on app startup instead of retraining.
# fake_news_streamlit_app
