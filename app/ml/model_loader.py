import os
import joblib

# 현재 파일: app/ml/model_loader.py
# 두 단계 상위가 프로젝트 루트
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
object_dir = os.path.join(project_root, "object")

vectorizer_path = os.path.join(object_dir, "tfidf_vectorizer.joblib")
model_path = os.path.join(object_dir, "sentiment_model.joblib")

vectorizer = joblib.load(vectorizer_path)
clf = joblib.load(model_path)
