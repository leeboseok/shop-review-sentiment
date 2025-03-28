from fastapi import FastAPI
from fastapi import Query
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List
from collections import Counter
from konlpy.tag import Okt
import joblib
import pymysql
import os
import numpy as np
import uvicorn

app = FastAPI()

# 모델 로드
cur_dir = os.path.dirname(__file__)
vectorizer = joblib.load(os.path.join(cur_dir, 'object', 'tfidf_vectorizer.joblib'))
clf = joblib.load(os.path.join(cur_dir, 'object', 'sentiment_model.joblib'))

# MySQL 연결 설정
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "boseok",
    "password": "iotiot",
    "database": "sentimentreview",
    "port": 3306
}

def get_db_connection():
    try:
        conn = pymysql.connect(
            **DB_CONFIG,
            cursorclass=pymysql.cursors.DictCursor,  # 💡 딕셔너리 형태로 결과 받기
            charset='utf8mb4'
        )
        return conn
    except pymysql.MySQLError as err:
        print(f"MySQL 연결 오류: {err}")
        return None

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vectorizer.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vectorizer.transform([document])
    clf.best_estimator_.partial_fit(X, [y])

def mysql_entry(document, y, proba):
    conn = get_db_connection()
    if conn is None:
        print("DB 연결 실패, 리뷰 저장 불가")
        return
    try:
        cursor = conn.cursor()
        query = "INSERT INTO review (review, sentiment, probability, date) VALUES (%s, %s, %s, NOW())"
        cursor.execute(query, (document, y, proba))
        conn.commit()
        print("리뷰 저장 완료!")
    except pymysql.MySQLError as err:
        print(f"MySQL 오류: {err}")
    finally:
        cursor.close()
        conn.close()


def save_feedback(document, original_sentiment, corrected_sentiment, feedback_type, probability):
    conn = get_db_connection()
    if conn is None:
        print("DB 연결 실패, 피드백 저장 불가")
        return
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO feedback (review, original_sentiment, corrected_sentiment, feedback_type, probability)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (document, original_sentiment, corrected_sentiment, feedback_type, probability))
        conn.commit()
        print("피드백 저장 완료!")
    except pymysql.MySQLError as err:
        print(f"MySQL 오류: {err}")
    finally:
        cursor.close()
        conn.close()

# 요청 모델
class ReviewRequest(BaseModel):
    review_text: str

class FeedbackRequest(BaseModel):
    review_text: str
    sentiment: str
    feedback: str

@app.post("/analyze-sentiment/")
async def analyze_sentiment(request: ReviewRequest):
    sentiment, proba = classify(request.review_text)
    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[sentiment]
    per_proba = round(proba * 100, 2)
    mysql_entry(request.review_text, y, per_proba)

    return {
        "review_text": request.review_text,
        "sentiment": sentiment,
        "probability": per_proba
    }

@app.post("/feedback/")
async def feedback(request: FeedbackRequest):
    inv_label = {'negative': 0, 'positive': 1}
    y_original = inv_label[request.sentiment]
    y_corrected = y_original

    if request.feedback == 'Incorrect':
        y_corrected = int(not y_original)

    # 모델 업데이트
    train(request.review_text, y_corrected)

    # 감정 확률 예측 → 확률 배열 중 가장 높은 값 선택 후 소수점 2자리까지 변환
    prob = clf.predict_proba(vectorizer.transform([request.review_text]))
    per_prob = round(float(np.max(prob)) * 100, 2)

    # 문자열로 다시 변환
    label_map = {0: 'negative', 1: 'positive'}
    sentiment_label = label_map[y_original]
    corrected_label = label_map[y_corrected]

    # 피드백 저장
    save_feedback(
        document=request.review_text,
        original_sentiment=sentiment_label,
        corrected_sentiment=corrected_label,
        feedback_type=request.feedback,
        probability=per_prob
    )

    return {
        "message": "피드백이 저장되고, 모델이 업데이트되었습니다.",
        "original_sentiment": sentiment_label,
        "corrected_sentiment": corrected_label,
        "probability": per_prob
    }

    # DB 저장
    save_feedback(
        document=request.review_text,
        original_sentiment=sentiment_label,
        corrected_sentiment=corrected_label,
        feedback_type=request.feedback,
        probability=per_prob
    )

    return {"message": "피드백이 저장되고, 모델이 업데이트되었습니다."}

@app.post("/analyze-batch/")
async def analyze_batch(reviews: List[ReviewRequest]):
    results = []
    for review in reviews:
        sentiment, proba = classify(review.review_text)
        results.append({
            "review_text": review.review_text,
            "sentiment": sentiment,
            "probability": round(proba * 100, 2)
        })
    return results

@app.post("/analyze-length-filter/")
async def analyze_with_length_filter(request: ReviewRequest):
    if len(request.review_text) < 10:
        return {"error": "리뷰가 너무 짧습니다."}

@app.get("/feedbacks/")
def get_feedbacks(limit: int = 10):
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500, content={"error": "DB 연결 실패"})

    try:
        cursor = conn.cursor()
        query = """
            SELECT id, review, original_sentiment, corrected_sentiment, feedback_type, probability, created_at
            FROM feedback
            ORDER BY created_at DESC
            LIMIT %s
        """
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        return rows
    except pymysql.MySQLError as err:
        return JSONResponse(status_code=500, content={"error": f"MySQL 오류: {err}"})
    finally:
        cursor.close()
        conn.close()

@app.get("/reviews/stats/sentiment")
def get_sentiment_distribution():
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500, content={"error": "DB 연결 실패"})

    try:
        cursor = conn.cursor()
        query = """
            SELECT sentiment, COUNT(*) as count
            FROM review
            GROUP BY sentiment
        """
        cursor.execute(query)
        results = cursor.fetchall()
        distribution = {row['sentiment']: row['count'] for row in results}
        return distribution
    except pymysql.MySQLError as err:
        return JSONResponse(status_code=500, content={"error": f"MySQL 오류: {err}"})
    finally:
        cursor.close()
        conn.close()

from konlpy.tag import Okt

okt = Okt()
stopwords = {'상품', '쇼핑몰', '정말', '도대체', '이거', '그냥', '너무', '완전', '진짜', '때문'}

def clean_and_tokenize(text: str) -> List[str]:
    # 1. 소문자화, 특수문자 제거 등 전처리
    text = text.lower()
    
    # 2. 형태소 분석
    tokens = okt.pos(text, stem=True)
    
    # 3. 명사/형용사만 추출 + 불용어 제거 + 길이 1 이상
    keywords = [
        word for word, tag in tokens
        if tag in ('Noun', 'Adjective') and word not in stopwords and len(word) > 1
    ]
    return keywords

@app.get("/keywords/top3")
def get_top_keywords(limit: int = 3):
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500, content={"error": "DB 연결 실패"})

    try:
        cursor = conn.cursor()
        # 최근 리뷰 500개에서 키워드 분석 (원하는 만큼 조절 가능)
        query = "SELECT review FROM review ORDER BY date DESC LIMIT 500"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        all_keywords = []
        for row in rows:
            review_text = row['review']
            keywords = clean_and_tokenize(review_text)
            all_keywords.extend(keywords)
        
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(limit)

        return {
            "top_keywords": [{"keyword": k, "count": v} for k, v in top_keywords]
        }

    except pymysql.MySQLError as err:
        return JSONResponse(status_code=500, content={"error": f"MySQL 오류: {err}"})
    finally:
        cursor.close()
        conn.close()

@app.put("/review/{review_id}")
def update_review(review_id: int, request: ReviewRequest):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    try:
        cursor = conn.cursor()
        query = "UPDATE review SET review = %s, date = NOW() WHERE id = %s"
        cursor.execute(query, (request.review_text, review_id))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="리뷰를 찾을 수 없습니다.")
        return {"message": "리뷰가 수정되었습니다."}
    except pymysql.MySQLError as err:
        raise HTTPException(status_code=500, detail=f"MySQL 오류: {err}")
    finally:
        cursor.close()
        conn.close()

@app.delete("/review/{review_id}")
def delete_review(review_id: int):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    try:
        cursor = conn.cursor()
        query = "DELETE FROM review WHERE id = %s"
        cursor.execute(query, (review_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="리뷰를 찾을 수 없습니다.")
        return {"message": "리뷰가 삭제되었습니다."}
    except pymysql.MySQLError as err:
        raise HTTPException(status_code=500, detail=f"MySQL 오류: {err}")
    finally:
        cursor.close()
        conn.close()

from fastapi import Query

@app.get("/reviews/search-by-keyword")
def search_reviews_by_keyword(keyword: str = Query(..., min_length=1)):
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500, content={"error": "DB 연결 실패"})

    try:
        cursor = conn.cursor()
        query = """
            SELECT id, review, sentiment, probability, date
            FROM review
            WHERE review LIKE %s
            ORDER BY date DESC
            LIMIT 100
        """
        keyword_pattern = f"%{keyword}%"
        cursor.execute(query, (keyword_pattern,))
        results = cursor.fetchall()
        return results
    except pymysql.MySQLError as err:
        return JSONResponse(status_code=500, content={"error": f"MySQL 오류: {err}"})
    finally:
        cursor.close()
        conn.close()

@app.get("/reviews/stats/weekly")
def get_weekly_review_stats():
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500, content={"error": "DB 연결 실패"})

    try:
        cursor = conn.cursor()
        query = """
            SELECT 
                DATE_FORMAT(date, '%Y-%u주차') AS week, 
                COUNT(*) AS count
            FROM review
            GROUP BY week
            ORDER BY week DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except pymysql.MySQLError as err:
        return JSONResponse(status_code=500, content={"error": f"MySQL 오류: {err}"})
    finally:
        cursor.close()
        conn.close()

@app.get("reviews/stats/monthlysentiment")
def get_monthly_review_stats_sentiment():
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500, content={"error": "DB 연결 실패"})

    try:
        cursor = conn.cursor()
        query = """
            SELECT 
            DATE_FORMAT(date, '%Y-%m') AS month,
            sentiment,
            COUNT(*) as count
            FROM review
            GROUP BY month, sentiment
            ORDER BY month DESC;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except pymysql.MySQLError as err:
        return JSONResponse(status_code=500, content={"error": f"MySQL 오류: {err}"})
    finally:
        cursor.close()
        conn.close()
    
@app.get("/reviews/stats/monthly")
def get_monthly_review_stats():
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500, content={"error": "DB 연결 실패"})

    try:
        cursor = conn.cursor()
        query = """
            SELECT DATE_FORMAT(date, '%Y-%m') AS month, COUNT(*) AS count
            FROM review
            GROUP BY month
            ORDER BY month DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except pymysql.MySQLError as err:
        return JSONResponse(status_code=500, content={"error": f"MySQL 오류: {err}"})
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)