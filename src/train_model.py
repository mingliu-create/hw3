import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 建立 models 資料夾
os.makedirs("models", exist_ok=True)

# 讀取資料
url = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
df = pd.read_csv(url, names=["label", "text"])

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# 特徵提取
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 建立模型
model = LinearSVC()
model.fit(X_train_vec, y_train)

# 評估
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# 儲存模型與向量器
joblib.dump(model, "models/svm_model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")

print("✅ Model and vectorizer saved to 'models/'")
