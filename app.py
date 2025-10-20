import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from pathlib import Path

# path for serialized DL model
DL_MODEL_PATH = Path(__file__).parent / 'dl_model.h5'
# path for serialized TF-IDF vectorizer
VECTORIZER_PATH = Path(__file__).parent / 'tfidf_vectorizer.joblib'

# Page config and simple CSS to make UI colorful
st.set_page_config(page_title='Fake News Detector', page_icon='📰', layout='centered')
_style = """
<style>
.stApp {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
body {background: linear-gradient(180deg, #fff 0%, #f7fbff 100%);} 
.stApp header {background: linear-gradient(90deg,#6dd5fa,#2980b9);}
.big-title {font-size:34px; font-weight:700; color:#0b3954}
.card {background: #ffffff; border-radius:10px; padding:14px; box-shadow: 0 4px 14px rgba(11,57,84,0.06);}
.muted {color:#4b5563; font-size:16px}
/* textarea styling */
textarea[aria-label="Paste news text here"] {font-size:16px; line-height:1.4}
.footer {text-align:center; font-size:16px; color:#ffffff; margin-top:18px; background:#87CEEB; padding:14px; border-radius:8px}
.topbar {background: linear-gradient(90deg,#6dd5fa,#2980b9); padding:14px; text-align:center; border-radius:8px; margin-bottom:14px}
.topbar h1 {color:#ffffff; margin:0; font-size:26px}

</style>
"""
st.markdown(_style, unsafe_allow_html=True)

# Top title bar in the header area
st.markdown('<div class="topbar"><h1>Fake News Detector</h1></div>', unsafe_allow_html=True)


@st.cache_resource
def download_and_load_data():
    # URLs / file ids from the original notebook
    try:
        file_id = "1FBHLmRhDtD9zFsuYupajb3ANyBVdG50j"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        true_df = pd.read_csv(url)

        file_id = "1ktQu00sC49tNGX-nvkOODfx0f0jTdwFY"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        fake_df = pd.read_csv(url)
    except Exception as e:
        st.warning(f"Could not download datasets automatically: {e}. You can upload CSVs manually in the sidebar.")
        true_df = pd.DataFrame()
        fake_df = pd.DataFrame()

    return true_df, fake_df


def clean_text(text, lemmatizer, stop_words):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def prepare_data(true_df, fake_df):
    # If download failed, expect user to upload via sidebar
    if true_df.empty or fake_df.empty:
        return None

    true_df['label'] = 0
    fake_df['label'] = 1
    df = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    # combine title + text if title exists
    if 'title' in df.columns:
        df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df = df.rename(columns={c: c for c in df.columns})
    for col in ['title', 'subject', 'date']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors='ignore')
    return df


@st.cache_data(show_spinner=False)
def fit_model(df, max_features=5000):
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    df['cleaned_text'] = df['text'].apply(lambda t: clean_text(t, lemmatizer, stop_words))
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # save sample of test rows for display
    sample = df.iloc[y_test.index] if hasattr(y_test, 'index') else None

    # Return the classical model, vectorizer and test splits. DL model is trained only when requested.
    return model, vectorizer, acc, X_train, X_test, y_train, y_test


def predict_text(text, model, vectorizer, lemmatizer, stop_words):
    cleaned = clean_text(text, lemmatizer, stop_words)
    X = vectorizer.transform([cleaned])
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)
    label = 'Fake' if pred == 1 else 'Real'
    return label, prob, cleaned


def train_feedforward_dl(X_train_arr, y_train, X_val_arr=None, y_val=None, epochs=5):
    # Train a simple feedforward DL model using TF-IDF dense arrays
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
    except Exception:
        raise RuntimeError('TensorFlow is not available in this environment')

    input_dim = X_train_arr.shape[1]
    model_dl = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model_dl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if X_val_arr is not None and y_val is not None:
        history = model_dl.fit(X_train_arr, y_train, epochs=epochs, batch_size=64, validation_data=(X_val_arr, y_val), verbose=0)
    else:
        history = model_dl.fit(X_train_arr, y_train, epochs=epochs, batch_size=64, verbose=0)
    return model_dl, history


def try_load_dl_model():
    """Attempt to load a serialized Keras model if present and TensorFlow is available."""
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model
    except Exception:
        return None

    if DL_MODEL_PATH.exists():
        try:
            model = load_model(str(DL_MODEL_PATH))
            return model
        except Exception as e:
            st.warning(f'Could not load saved DL model: {e}')
            return None
    return None

def try_load_vectorizer():
    """Attempt to load a serialized TF-IDF vectorizer if present."""
    if VECTORIZER_PATH.exists():
        try:
            vect = joblib.load(str(VECTORIZER_PATH))
            return vect
        except Exception as e:
            st.warning(f'Could not load saved TF-IDF vectorizer: {e}')
            return None
    return None


def main():
    # subtitle / instructions
    st.markdown('<div class="muted">Paste an article or news text and choose a model from sidebar to classify it as Fake or Real.</div>', unsafe_allow_html=True)

    true_df, fake_df = download_and_load_data()

    uploaded_true = st.sidebar.file_uploader('Optional: upload true news CSV', type=['csv'])
    uploaded_fake = st.sidebar.file_uploader('Optional: upload fake news CSV', type=['csv'])

    if uploaded_true is not None:
        true_df = pd.read_csv(uploaded_true)
    if uploaded_fake is not None:
        fake_df = pd.read_csv(uploaded_fake)

    df = prepare_data(true_df, fake_df)

    if df is None:
        st.sidebar.info('No datasets available. Upload CSV files for true and fake news in the sidebar to enable training.')
        st.stop()

    st.sidebar.write('Dataset loaded. Choose a model and train below.')

    model_choice = st.sidebar.radio('Select model', ('Logistic Regression', 'FeedforwardDL'))

    train_button = st.sidebar.button('Train / (re)fit selected model')

    model = None
    vectorizer = None
    acc = None
    # try load any previously saved DL model at startup
    dl_loaded = try_load_dl_model()
    if dl_loaded is not None:
        st.session_state['dl_model'] = dl_loaded
    # try load vectorizer if present
    vect_loaded = try_load_vectorizer()
    if vect_loaded is not None:
        st.session_state['vectorizer'] = vect_loaded
    # Fit classical model now (fit_model returns X_train/X_test splits too)
    if train_button:
        with st.spinner('Training selected model...'):
            try:
                model_lr, vectorizer, acc, X_train, X_test, y_train, y_test = fit_model(df)
                st.sidebar.success(f'Logistic model trained. Test accuracy: {acc:.3f}')
                # persist the vectorizer for faster startup
                try:
                    joblib.dump(vectorizer, str(VECTORIZER_PATH))
                    st.sidebar.info(f'Saved TF-IDF vectorizer to {VECTORIZER_PATH.name}')
                except Exception as e:
                    st.sidebar.warning(f'Failed to save TF-IDF vectorizer: {e}')
            except Exception as e:
                st.sidebar.error(f'Failed training: {e}')
                return

            # if user selected DL, try to train FeedforwardDL on TF-IDF dense arrays
            dl_model = None
            if model_choice == 'FeedforwardDL':
                try:
                    X_train_arr = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
                    X_test_arr = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
                    dl_model, history = train_feedforward_dl(X_train_arr, y_train, X_val_arr=X_test_arr, y_val=y_test, epochs=5)
                    y_prob_dl = dl_model.predict(X_test_arr, verbose=0).ravel()
                    y_pred_dl = (y_prob_dl > 0.5).astype('int32')
                    dl_acc = accuracy_score(y_test, y_pred_dl)
                    st.sidebar.success(f'Feedforward DL trained. Test accuracy: {dl_acc:.3f}')
                    # try to persist the trained DL model for future runs
                    try:
                        dl_model.save(str(DL_MODEL_PATH))
                        st.sidebar.info(f'Saved DL model to {DL_MODEL_PATH.name}')
                    except Exception as e:
                        st.sidebar.warning(f'Failed to save DL model: {e}')
                except RuntimeError as re:
                    st.sidebar.error(str(re))
                    dl_model = None

            # expose trained classical model and optional dl model in session state for reuse
            st.session_state['logistic_model'] = model_lr
            st.session_state['vectorizer'] = vectorizer
            st.session_state['dl_model'] = dl_model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['acc'] = acc

    # prepare lemmatizer and stopwords for prediction
    lemmatizer = WordNetLemmatizer()
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    # Use session_state to hold the input so the Clear button can reset it
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = ''

    user_text = st.text_area('Paste news text here', value=st.session_state['input_text'], key='input_text', height=360)
    # Place Predict and Clear side-by-side; use a callback to clear
    def _clear_input():
        st.session_state['input_text'] = ''

    c1, c2 = st.columns([1, 0.35])
    do_predict = c1.button('Predict')
    c2.button('Clear', on_click=_clear_input)

    if do_predict:
        user_text = st.session_state.get('input_text', '')
        if not user_text or user_text.strip() == '':
            st.warning('Please enter news text to classify.')
        else:
            # choose model from session state
            sel_model_name = st.sidebar.radio('Model for prediction', ('Logistic Regression', 'FeedforwardDL'))
            vec = st.session_state.get('vectorizer')
            logm = st.session_state.get('logistic_model')
            dlm = st.session_state.get('dl_model')
            if sel_model_name == 'Logistic Regression':
                if logm is None or vec is None:
                    st.error('Logistic model not trained yet. Train it using the sidebar button.')
                else:
                    label, prob, cleaned = predict_text(user_text, logm, vec, lemmatizer, stop_words)
                    st.success(f'Prediction: {label}  —  Fake probability: {prob:.3f}')
                    with st.expander('Show cleaned text'):
                        st.write(cleaned)
            else:
                if dlm is None or vec is None:
                    st.error('DL model not trained or TensorFlow not available. Train DL using the sidebar or select Logistic Regression.')
                else:
                    cleaned = clean_text(user_text, lemmatizer, stop_words)
                    X_user = vec.transform([cleaned])
                    X_user_arr = X_user.toarray() if hasattr(X_user, 'toarray') else X_user
                    prob = dlm.predict(X_user_arr, verbose=0).ravel()[0]
                    pred = int(prob >= 0.5)
                    label = 'Fake' if pred == 1 else 'Real'
                    st.success(f'Prediction (DL): {label}  —  Fake probability: {prob:.3f}')
                    with st.expander('Show cleaned text'):
                        st.write(cleaned)

    st.markdown('---')
    st.subheader('Model Evaluation (on hold-out test set)')
    # show evaluation only if a trained accuracy exists in session_state
    if st.session_state.get('acc') is not None:
        st.write(f'Accuracy on hold-out test set: {st.session_state.get("acc"):.3f}')
    else:
        st.write('No trained model available. Train a model using the sidebar to see evaluation metrics here.')

    st.sidebar.markdown('---')
    st.sidebar.write('Notes: You can upload CSVs matching the original dataset format (columns: title, text, subject, date).')

    # Footer / credits (prominent, sky-blue background)
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">&copy; Developed by Pokhraj, Ravi, Mansi and Ramya</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
