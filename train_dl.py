"""Train a small feedforward DL model using TF-IDF features and save artifacts.

This script safely loads functions from app.py, prepares data, trains a Keras model,
and writes dl_model.h5 and tfidf_vectorizer.joblib into the app folder.

Run with a Python interpreter that has TensorFlow installed.
Example:
  conda create -n fake_news_tf python=3.11 -y
  conda run -n fake_news_tf pip install tensorflow scikit-learn pandas joblib nltk
  conda run -n fake_news_tf python train_dl.py
"""
from pathlib import Path
import importlib.util
import sys
import joblib
import numpy as np

APP_P = Path(__file__).resolve().parent / 'app.py'
if not APP_P.exists():
    raise FileNotFoundError(f"app.py not found at {APP_P}")

# Load app.py as a module without executing it as __main__ to avoid streamlit side-effects
spec = importlib.util.spec_from_file_location('app_mod', str(APP_P))
app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)

# Required helpers
for fn in ('download_and_load_data', 'prepare_data', 'fit_model'):
    if not hasattr(app_mod, fn):
        raise RuntimeError(f"{fn} not found in app.py; cannot proceed")

print('Loading data...')
true_df, fake_df = app_mod.download_and_load_data()
if true_df is None or fake_df is None:
    raise RuntimeError('Data download or load failed inside app.py')

print('Preparing dataframe...')
df = app_mod.prepare_data(true_df, fake_df)

print('Fitting TF-IDF and getting train/test splits...')
# fit_model in app.py returns (model, vectorizer, acc, X_train, X_test, y_train, y_test)
fit_result = app_mod.fit_model(df, max_features=5000)
_, vectorizer, acc, X_train, X_test, y_train, y_test = fit_result

# Convert to dense arrays for Keras
X_train_arr = X_train.toarray() if hasattr(X_train, 'toarray') else np.array(X_train)
X_test_arr = X_test.toarray() if hasattr(X_test, 'toarray') else np.array(X_test)

print('Train shape:', X_train_arr.shape, 'Test shape:', X_test_arr.shape)

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception as e:
    raise RuntimeError('TensorFlow import failed. Run this script with a TF-capable interpreter') from e

input_dim = X_train_arr.shape[1]
model = Sequential([
    Dense(256, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print('Training model...')
model.fit(X_train_arr, y_train, validation_split=0.1, epochs=10, batch_size=32, callbacks=[es], verbose=2)

out_dir = Path(__file__).resolve().parent
model_path = out_dir / 'dl_model.h5'
vec_path = out_dir / 'tfidf_vectorizer.joblib'
print('Saving', model_path, 'and', vec_path)
model.save(model_path)
joblib.dump(vectorizer, vec_path)

loss, acc = model.evaluate(X_test_arr, y_test, verbose=0)
print(f'Test loss: {loss:.4f}  Test acc: {acc:.4f}')

print('Done')
