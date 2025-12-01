import joblib
import os

mp = os.path.join('models','svm_model.joblib')
vp = os.path.join('models','vectorizer.joblib')

print('Model path:', mp, 'exists:', os.path.exists(mp))
print('Vectorizer path:', vp, 'exists:', os.path.exists(vp))

try:
    m = joblib.load(mp)
    v = joblib.load(vp)
    X = v.transform(['Hello friend, I hope you are well and this is not spam.'])
    pred = m.predict(X)
    print('PRED:', pred)
    if hasattr(m, 'predict_proba'):
        print('PROBA:', m.predict_proba(X))
    elif hasattr(m, 'decision_function'):
        print('DECISION:', m.decision_function(X))
    else:
        print('No probability/decision function available')
except Exception as e:
    print('LOAD/INFER ERROR:', e)
