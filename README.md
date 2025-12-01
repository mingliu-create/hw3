# hw3
**Email Spam Classifier**

- **Description:**: A simple SVM-based email spam classifier with a Streamlit web UI to classify messages as spam or ham.

**Setup**
- **Python:**: Recommended Python 3.10+.
- **Install deps:**: Install project dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

- **Train model (if missing):**: If `models/svm_model.joblib` and `models/vectorizer.joblib` are not present, run the training script to create them.

```bash
python src/train_model.py
```

**Run locally**
- **Start the Streamlit app:**

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

**Deploy to Streamlit Cloud**
- **Push to GitHub:** Ensure your repository is up-to-date on GitHub (this repo already contains an example `app.py`).
- **Create an app on Streamlit Cloud:**
	1. Visit https://share.streamlit.io and sign in.
	2. Click "New app" → connect your GitHub repo → select branch `main` and the `app.py` file → deploy.

- **Model files note:**
	- This repository currently stores trained model files in `models/`. For a cleaner repo you can:
		- Stop tracking those files and add `models/` to `.gitignore`:

```bash
git rm -r --cached models
echo models/ >> .gitignore
git add .gitignore
git commit -m "Stop tracking model artifacts"
git push origin main
```

	- Or use Git LFS to store large binary files instead of committing them directly.

**Project Structure**
- **`app.py`**: Streamlit web application (loads model from `models/`).
- **`src/train_model.py`**: Script to train the SVM and save `models/svm_model.joblib` and `models/vectorizer.joblib`.
- **`models/`**: Directory containing trained model and vectorizer (binary files).
- **`requirements.txt`**: Python dependencies.

**Troubleshooting**
- If Streamlit complains about `set_page_config` being called multiple times, ensure `app.py` contains only one `st.set_page_config` call (this repo's `app.py` was recently fixed to avoid duplicates).
- If the app cannot find model files, run `python src/train_model.py` or place the correct joblib files in `models/`.

**License**
- MIT
"# hw3" 

