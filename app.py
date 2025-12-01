import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide"
)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('models/svm_model.joblib')
    vectorizer = joblib.load('models/vectorizer.joblib')
    return model, vectorizer
    

try:
    model, vectorizer = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Title and description
st.title("📧 Email Spam Classifier")
st.markdown("""
This application uses a Support Vector Machine (SVM) classifier to detect spam emails.
Enter your email text below to check if it's spam or not.
""")

# Input text area
email_text = st.text_area("Enter email text:", height=200)

if st.button("Classify Email"):
    if email_text:
        # Transform the text
        text_vectorized = vectorizer.transform([email_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Display result
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("📨 This email is likely SPAM!")
            else:
                st.success("✉️ This email appears to be HAM (not spam).")
        
        with col2:
            st.info(f"Confidence: {probability[int(prediction)]:.2%}")
        
        # Show feature importance (if available)
        st.subheader("Important Features")
        feature_names = vectorizer.get_feature_names_out()
        transformed_text = text_vectorized.toarray()[0]
        
        # Get top 10 features present in the email
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Value': transformed_text
        })
        feature_importance = feature_importance[feature_importance['Value'] > 0]
        feature_importance = feature_importance.sort_values('Value', ascending=False).head(10)
        
        st.bar_chart(feature_importance.set_index('Feature'))
        
    else:
        st.warning("Please enter some text to classify.")

# Add information about the model
import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd

st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")

MODEL_PATH = os.path.join("models", "svm_model.joblib")
VECT_PATH = os.path.join("models", "vectorizer.joblib")

@st.cache_resource
def load_model_and_vectorizer(model_path=MODEL_PATH, vect_path=VECT_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vect_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vect_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)
    return model, vectorizer

try:
    model, vectorizer = load_model_and_vectorizer()
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

st.title("📧 Email Spam Classifier")
st.write("Enter email text below and click `Classify`.")

text = st.text_area("Email text:", height=200)

if st.button("Classify"):
    if not text or text.strip() == "":
        st.warning("Please enter some email text to classify.")
    else:
        X = vectorizer.transform([text])

        # Predict label
        try:
            pred = model.predict(X)[0]
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            raise

        # Derive confidence/probability in a robust way
        confidence = None
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                # model.classes_ gives order of probabilities
                classes = list(model.classes_)
                idx = classes.index(pred)
                confidence = probs[idx]
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X)
                # handle binary vs multiclass shapes
                def _sigmoid(x):
                    return 1.0 / (1.0 + np.exp(-x))

                if isinstance(df, np.ndarray) and df.ndim == 1:
                    # binary: map distance to [0,1]
                    prob_pos = _sigmoid(df[0])
                    # assume classes_[1] corresponds to the "positive" direction
                    if len(model.classes_) >= 2:
                        pos_label = model.classes_[1]
                        confidence = float(prob_pos) if pred == pos_label else float(1 - prob_pos)
                    else:
                        confidence = float(prob_pos)
                else:
                    # multiclass: apply softmax
                    df_row = np.atleast_2d(df)[0]
                    exps = np.exp(df_row - np.max(df_row))
                    probs = exps / exps.sum()
                    classes = list(model.classes_)
                    idx = classes.index(pred)
                    confidence = float(probs[idx])
        except Exception:
            confidence = None

        # Display result
        is_spam_label = None
        try:
            # If labels are numeric (0/1) or strings, we detect a label that implies spam
            if isinstance(pred, (int, np.integer)):
                is_spam_label = (int(pred) == 1)
            else:
                # common label names
                is_spam_label = str(pred).lower() in ("spam", "1", "true", "yes")
        except Exception:
            is_spam_label = False

        if is_spam_label:
            st.error("📨 This email is predicted: SPAM")
        else:
            st.success("✉️ This email is predicted: HAM (not spam)")

        if confidence is not None:
            st.info(f"Confidence: {confidence:.2%}")
        else:
            st.info("Confidence: N/A")

        # Show top contributing features when available
        st.subheader("Top features in this message")

        try:
            feat_names = vectorizer.get_feature_names_out()
            x_arr = X.toarray()[0]
            present_idx = np.where(x_arr > 0)[0]

            if hasattr(model, "coef_"):
                # linear model: compute contribution = feature_value * coef_for_class
                coefs = model.coef_
                # choose coef row corresponding to predicted class if multiclass
                if coefs.ndim == 1:
                    coef_row = coefs
                else:
                    classes = list(model.classes_)
                    coef_row = coefs[classes.index(pred)]

                contributions = coef_row * x_arr
                top_idx = np.argsort(np.abs(contributions))[::-1][:10]
                rows = []
                for i in top_idx:
                    if x_arr[i] <= 0:
                        continue
                    rows.append((feat_names[i], float(contributions[i])))

                if rows:
                    df = pd.DataFrame(rows, columns=["feature", "contribution"]).set_index("feature")
                    st.bar_chart(df)
                else:
                    # fallback to showing top tf-idf tokens
                    top_idx = np.argsort(x_arr)[::-1][:10]
                    rows = [(feat_names[i], float(x_arr[i])) for i in top_idx if x_arr[i] > 0]
                    if rows:
                        df = pd.DataFrame(rows, columns=["feature", "tfidf"]).set_index("feature")
                        st.bar_chart(df)
                    else:
                        st.write("No token features present in this message (after vectorization).")
            else:
                # no coef_: just show top tf-idf tokens
                top_idx = np.argsort(x_arr)[::-1][:10]
                rows = [(feat_names[i], float(x_arr[i])) for i in top_idx if x_arr[i] > 0]
                if rows:
                    df = pd.DataFrame(rows, columns=["feature", "tfidf"]).set_index("feature")
                    st.bar_chart(df)
                else:
                    st.write("No token features present in this message (after vectorization).")
        except Exception as e:
            st.write("Could not compute feature contributions:", e)

# Footer
st.markdown("---")
st.write("Model files expected in `models/svm_model.joblib` and `models/vectorizer.joblib`.")