from fastapi import FastAPI
from app.api.api import router

app = FastAPI(
    title="🛍️ 쇼핑몰 리뷰 감성 분석 API",
    description="""
이 API는 쇼핑몰 리뷰에 대해 감성 분석을 수행하고  
피드백을 반영하여 모델을 개선하는 기능을 제공합니다.

- 감성 분석 (긍정/부정)
- 피드백 수집 및 모델 업데이트
- 리뷰 키워드 추출
- 통계 조회 (월별, 주차별, 감정 분포 등)
""",
    version="1.0.0",
    contact={
        "name": "Boseok",
        "url": "https://github.com/leeboseok",
        "email": "boseok.lee@hotmail.com",
    }
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
