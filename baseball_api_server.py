# baseball_api_server.py
"""
⚾ Baseball AI 예측 API 서버 (FastAPI)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict
import os

# FastAPI 앱 생성
app = FastAPI(title="Baseball AI Prediction API")

# CORS 설정 (Next.js에서 접근 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
MODEL_DIR = 'models'

print("🤖 모델 로딩 중...")
try:
    win_model = joblib.load(os.path.join(MODEL_DIR, 'baseball_win_model.pkl'))
    over_model = joblib.load(os.path.join(MODEL_DIR, 'baseball_over_model.pkl'))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    print("✅ 모델 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    win_model = None
    over_model = None
    feature_columns = None

# 요청/응답 모델
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    home_win_prob: float
    away_win_prob: float
    over_prob: float
    under_prob: float
    confidence: str
    grade: str

# API 엔드포인트
@app.get("/")
def root():
    """루트 엔드포인트"""
    return {
        "message": "Baseball AI Prediction API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    """헬스 체크"""
    models_loaded = win_model is not None and over_model is not None
    return {
        "status": "healthy" if models_loaded else "error",
        "models_loaded": models_loaded,
        "features": len(feature_columns) if feature_columns else 0
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """예측 API"""
    try:
        if win_model is None or over_model is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Features 배열로 변환
        X = np.array([[request.features[col] for col in feature_columns]])
        
        # 예측
        home_win_prob = float(win_model.predict_proba(X)[0][1])
        over_prob = float(over_model.predict_proba(X)[0][1])
        
        # 신뢰도 계산
        win_confidence = abs(home_win_prob - 0.5)
        over_confidence = abs(over_prob - 0.5)
        avg_confidence = (win_confidence + over_confidence) / 2
        
        # 등급 결정
        if avg_confidence >= 0.15:
            grade = "PICK"
            confidence = "HIGH"
        elif avg_confidence >= 0.08:
            grade = "GOOD"
            confidence = "MEDIUM"
        else:
            grade = "PASS"
            confidence = "LOW"
        
        return PredictionResponse(
            home_win_prob=home_win_prob,
            away_win_prob=1 - home_win_prob,
            over_prob=over_prob,
            under_prob=1 - over_prob,
            confidence=confidence,
            grade=grade
        )
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 서버 시작...")
    print("📍 http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
