# baseball_api_server.py
"""
⚾ Baseball AI 예측 API 서버 (FastAPI)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, Optional
import os
import subprocess
import threading
import json
from datetime import datetime

# FastAPI 앱 생성
app = FastAPI(title="Baseball AI Prediction API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
MODEL_DIR = 'models/MLB'

# 재학습 상태 추적
retrain_status = {
    "is_training": False,
    "last_trained": None,
    "last_result": None,
}

def load_models():
    """모델 파일 로드"""
    global win_model, over_model, feature_columns
    try:
        win_model = joblib.load(os.path.join(MODEL_DIR, 'baseball_win_model.pkl'))
        over_model = joblib.load(os.path.join(MODEL_DIR, 'baseball_over_model.pkl'))
        feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
        print("✅ 모델 로드 완료!")
        return True
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        win_model = None
        over_model = None
        feature_columns = None
        return False

print("🤖 모델 로딩 중...")
win_model = None
over_model = None
feature_columns = None
load_models()


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

class RetrainRequest(BaseModel):
    secret: str  # 보안용 시크릿 키
    min_games: Optional[int] = 100  # 최소 경기수 (기본 100개)


def run_retrain(min_games: int):
    """백그라운드 재학습 실행"""
    global retrain_status

    retrain_status["is_training"] = True
    retrain_status["last_result"] = None
    print(f"🔄 재학습 시작... (min_games={min_games})")

    try:
        # 학습 스크립트 실행
        result = subprocess.run(
            ["python", "baseball_ai_model_training_v4.py", "--league=MLB", f"--min-games={min_games}"],
            capture_output=True,
            text=True,
            timeout=600  # 10분 타임아웃
        )

        if result.returncode == 0:
            print("✅ 재학습 완료! 모델 리로드 중...")

            # 모델 리로드 (새 모델 파일로 교체)
            load_models()

            # GitHub push (Railway 자동 재배포 트리거)
            push_result = push_models_to_github()

            retrain_status["last_trained"] = datetime.now().isoformat()
            retrain_status["last_result"] = {
                "success": True,
                "github_push": push_result,
                "output": result.stdout[-500:] if result.stdout else "",
            }
            print("🚀 GitHub push 완료 - Railway 재배포 트리거됨")
        else:
            print(f"❌ 재학습 실패: {result.stderr}")
            retrain_status["last_result"] = {
                "success": False,
                "error": result.stderr[-500:] if result.stderr else "Unknown error",
            }

    except subprocess.TimeoutExpired:
        retrain_status["last_result"] = {"success": False, "error": "Timeout (10min)"}
    except Exception as e:
        retrain_status["last_result"] = {"success": False, "error": str(e)}
    finally:
        retrain_status["is_training"] = False


def push_models_to_github():
    """학습된 모델 파일을 GitHub에 push"""
    try:
        github_token = os.environ.get("GITHUB_TOKEN")
        github_repo = os.environ.get("GITHUB_REPO")  # ex: username/baseball-ai-api

        if not github_token or not github_repo:
            return {"success": False, "error": "GITHUB_TOKEN or GITHUB_REPO not set"}

        # git 설정
        subprocess.run(["git", "config", "user.email", "auto-retrain@trendsoccer.com"], check=True)
        subprocess.run(["git", "config", "user.name", "TrendSoccer AutoTrain"], check=True)

        # remote URL에 토큰 포함
        remote_url = f"https://{github_token}@github.com/{github_repo}.git"
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)

        # 모델 파일 add
        subprocess.run(["git", "add", "models/MLB/"], check=True)

        # 변경사항 있는지 확인
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not status.stdout.strip():
            return {"success": True, "message": "No changes to push"}

        # commit & push
        commit_msg = f"Auto-retrain: {datetime.now().strftime('%Y-%m-%d %H:%M')} KST"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)

        return {"success": True, "message": commit_msg}

    except subprocess.CalledProcessError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# API 엔드포인트
@app.get("/")
def root():
    return {
        "message": "Baseball AI Prediction API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    models_loaded = win_model is not None and over_model is not None
    return {
        "status": "healthy" if models_loaded else "error",
        "models_loaded": models_loaded,
        "features": len(feature_columns) if feature_columns else 0,
        "last_trained": retrain_status["last_trained"],
        "is_training": retrain_status["is_training"],
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        if win_model is None or over_model is None:
            raise HTTPException(status_code=500, detail="Models not loaded")

        X = np.array([[request.features[col] for col in feature_columns]])

        home_win_prob = float(win_model.predict_proba(X)[0][1])
        over_prob = float(over_model.predict_proba(X)[0][1])

        win_confidence = abs(home_win_prob - 0.5)
        over_confidence = abs(over_prob - 0.5)
        avg_confidence = (win_confidence + over_confidence) / 2

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


@app.post("/retrain")
def retrain(request: RetrainRequest, background_tasks: BackgroundTasks):
    """
    모델 재학습 엔드포인트
    - 백그라운드에서 학습 실행
    - 완료 후 GitHub push → Railway 자동 재배포
    """
    # 시크릿 검증
    retrain_secret = os.environ.get("RETRAIN_SECRET", "")
    if not retrain_secret or request.secret != retrain_secret:
        raise HTTPException(status_code=401, detail="Invalid secret")

    # 이미 학습 중이면 스킵
    if retrain_status["is_training"]:
        return {
            "success": False,
            "message": "Already training in progress",
            "status": retrain_status,
        }

    # 백그라운드 재학습 시작
    background_tasks.add_task(run_retrain, request.min_games)

    return {
        "success": True,
        "message": "Retrain started in background",
        "min_games": request.min_games,
    }


@app.get("/retrain/status")
def retrain_status_check():
    """재학습 상태 확인"""
    return retrain_status


if __name__ == "__main__":
    import uvicorn
    print("🚀 서버 시작...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
