# baseball_api_server.py
"""
⚾ Baseball AI 예측 API 서버 (FastAPI)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, Optional, Any
import os
import subprocess
import threading
import json
import re
from datetime import datetime
import httpx


def parse_aucs(stdout: str) -> dict:
    """학습 stdout에서 리그별 승부/총점 AUC 파싱.

    학습 스크립트의 요약 라인 형식 예:
      "  MLB (투수 포함): 승부 AUC 0.587 (양호) | 총점 AUC 0.542"
    파싱 실패해도 빈 dict를 반환해 로깅을 막지 않는다.
    """
    aucs = {}
    if not stdout:
        return aucs
    pattern = re.compile(
        r"(MLB|KBO|NPB).*?승부\s*AUC\s*([\d.]+).*?총점\s*AUC\s*([\d.]+)"
    )
    for m in pattern.finditer(stdout):
        league = m.group(1)
        try:
            aucs[league] = {
                "win": round(float(m.group(2)), 3),
                "over": round(float(m.group(3)), 3),
            }
        except ValueError:
            continue
    return aucs

# Supabase 설정
SUPABASE_URL = os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

async def save_retrain_log(league: str, success: bool, games_used: int = 0, error_message: str = None, github_push: bool = False, aucs: dict = None):
    """재학습 결과를 Supabase에 저장"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        url = f"{SUPABASE_URL}/rest/v1/baseball_retrain_logs"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "league": league,
            "success": success,
            "games_used": games_used,
            "error_message": error_message,
            "github_push": github_push,
            "aucs": aucs or {},
        }
        async with httpx.AsyncClient() as client:
            res = await client.post(url, headers=headers, json=payload)
            print(f"📡 재학습 로그 저장: {res.status_code}")
    except Exception as e:
        print(f"⚠️ 재학습 로그 저장 실패: {e}")

async def save_ai_pick_to_db(api_match_id: int, league: str, grade: str, confidence: str, home_win_prob: float, away_win_prob: float):
    """예측 결과를 baseball_odds_latest에 저장"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        url = f"{SUPABASE_URL}/rest/v1/baseball_odds_latest"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        }
        payload = {
            "api_match_id": api_match_id,
            "league": league,
            "ai_pick": grade,
            "ai_pick_confidence": confidence,
            "home_win_prob": round(home_win_prob * 100, 2),
            "away_win_prob": round(away_win_prob * 100, 2),
        }
        async with httpx.AsyncClient() as client:
            res = await client.patch(
                url,
                headers=headers,
                json=payload,
                params={"api_match_id": f"eq.{api_match_id}"}
            )
            print(f"📡 Supabase 응답: {res.status_code} {res.text[:200]}")
    except Exception as e:
        print(f"⚠️ DB 저장 실패 ({api_match_id}): {e}")

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

# 리그별 모델 저장소
SUPPORTED_LEAGUES = ['MLB', 'KBO', 'NPB']
models = {}

# 재학습 상태 추적
retrain_status = {
    "is_training": False,
    "last_trained": None,
    "last_result": None,
}

def load_models():
    """리그별 모델 파일 로드"""
    global models
    for league in SUPPORTED_LEAGUES:
        model_dir = f'models/{league}'
        try:
            models[league] = {
                'win_model': joblib.load(os.path.join(model_dir, 'baseball_win_model.pkl')),
                'over_model': joblib.load(os.path.join(model_dir, 'baseball_over_model.pkl')),
                'feature_columns': joblib.load(os.path.join(model_dir, 'feature_columns.pkl')),
            }
            print(f"✅ {league} 모델 로드 완료!")
        except Exception as e:
            print(f"⚠️ {league} 모델 로드 실패: {e}")

print("🤖 모델 로딩 중...")
models = {}
load_models()


def restore_retrain_status():
    """앱 시작 시 Supabase 최신 재학습 로그로 retrain_status 복원.

    Railway는 재배포/재시작마다 컨테이너가 새로 뜨므로 메모리에만 있던
    retrain_status 가 초기화된다. 그러면 /retrain/status 가 last_trained=null 을
    반환해 모니터링이 매번 '기록 없음'으로 보인다.
    → baseball_retrain_logs(영속) 최신 행을 읽어 메모리 상태를 복구한다.
    실패해도 서버 기동을 막지 않도록 예외를 모두 무시한다.
    """
    global retrain_status
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        url = f"{SUPABASE_URL}/rest/v1/baseball_retrain_logs"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }
        params = {"select": "*", "order": "trained_at.desc", "limit": "1"}
        with httpx.Client(timeout=15) as client:
            res = client.get(url, headers=headers, params=params)
        rows = res.json() if res.status_code == 200 else []
        if not rows:
            print("ℹ️ 복원할 재학습 로그 없음")
            return
        row = rows[0]
        retrain_status["last_trained"] = row.get("trained_at")
        retrain_status["last_result"] = {
            "success": row.get("success"),
            "github_push": {"success": row.get("github_push")},
            "aucs": row.get("aucs") or {},
            "error": row.get("error_message"),
            "restored_from_log": True,  # 라이브 결과가 아니라 DB에서 복원됐음을 표시
        }
        print(f"♻️ 재학습 상태 복원: last_trained={row.get('trained_at')}, "
              f"success={row.get('success')}, github_push={row.get('github_push')}")
    except Exception as e:
        print(f"⚠️ 재학습 상태 복원 실패(무시): {e}")


restore_retrain_status()


# 요청/응답 모델
class PredictionRequest(BaseModel):
    features: Dict[str, float]
    league: Optional[str] = 'MLB'  # MLB / KBO / NPB
    match_id: Optional[int] = None  # DB 저장용

class PredictionResponse(BaseModel):
    home_win_prob: float
    away_win_prob: float
    over_prob: float
    under_prob: float
    confidence: str
    grade: str

class RetrainRequest(BaseModel):
    secret: str
    min_games: Optional[int] = 20
    league: Optional[str] = 'ALL'  # MLB / KBO / NPB / ALL


def run_retrain(min_games: int, league: str = "ALL"):
    """백그라운드 재학습 실행"""
    import asyncio
    global retrain_status

    retrain_status["is_training"] = True
    retrain_status["last_result"] = None
    print(f"🔄 재학습 시작... (min_games={min_games}, league={league})")

    success = False
    error_message = None
    github_push_success = False
    aucs = {}

    try:
        # 학습 스크립트 실행
        result = subprocess.run(
            ["python", "baseball_ai_model_training_v5.py", f"--league={league}", f"--min-games={min_games}"],
            capture_output=True,
            text=True,
            timeout=900  # 15분 타임아웃 (6/17·6/19 학습 10분 초과 실패 대응)
        )

        if result.returncode == 0:
            print("✅ 재학습 완료! 모델 리로드 중...")

            # 모델 리로드 (새 모델 파일로 교체)
            load_models()

            # GitHub push (전체 리그 모델 포함)
            push_result = push_models_to_github()
            github_push_success = push_result.get("success", False)

            # 학습 출력에서 리그별 AUC 파싱 (실패해도 무시)
            aucs = parse_aucs(result.stdout)

            retrain_status["last_trained"] = datetime.now().isoformat()
            retrain_status["last_result"] = {
                "success": True,
                "github_push": push_result,
                "aucs": aucs,
                "output": result.stdout[-500:] if result.stdout else "",
            }
            success = True
            print("🚀 GitHub push 완료 - Railway 재배포 트리거됨")
        else:
            error_message = result.stderr[-500:] if result.stderr else result.stdout[-500:] if result.stdout else "Unknown error"
            print(f"❌ 재학습 실패: {error_message}")
            retrain_status["last_result"] = {
                "success": False,
                "error": error_message,
            }

    except subprocess.TimeoutExpired:
        error_message = "Timeout (15min)"
        retrain_status["last_result"] = {"success": False, "error": error_message}
    except Exception as e:
        error_message = str(e)
        retrain_status["last_result"] = {"success": False, "error": error_message}
    finally:
        retrain_status["is_training"] = False
        # Supabase에 결과 저장
        asyncio.run(save_retrain_log(
            league=league,
            success=success,
            games_used=min_games,
            error_message=error_message,
            github_push=github_push_success,
            aucs=aucs,
        ))


def push_models_to_github():
    """학습된 모델 파일을 GitHub에 push.

    주의: Railway(Nixpacks)는 코드를 git clone이 아니라 '스냅샷 복사'로 배포하므로
    실행 컨테이너에는 .git 저장소가 없다. 따라서 현재 디렉터리에서 git 명령을
    실행하면 'not a git repository'(exit 128)로 실패한다.
    → 임시 디렉터리에 레포를 새로 clone하고, 학습된 models/ 를 그쪽으로 복사한 뒤
      commit/push 하는 방식으로 .git 부재 문제를 우회한다.
    """
    import shutil
    import tempfile

    tmp_dir = None
    try:
        github_token = os.environ.get("GITHUB_TOKEN")
        github_repo = os.environ.get("GITHUB_REPO")  # ex: tri-kkk/baseball-ai-api

        if not github_token or not github_repo:
            return {"success": False, "error": "GITHUB_TOKEN or GITHUB_REPO not set"}

        if not os.path.isdir("models"):
            return {"success": False, "error": "models/ 디렉터리 없음 (학습 산출물 누락)"}

        remote_url = f"https://x-access-token:{github_token}@github.com/{github_repo}.git"

        tmp_dir = tempfile.mkdtemp(prefix="model-push-")
        repo_dir = os.path.join(tmp_dir, "repo")

        def run(args):
            return subprocess.run(args, check=True, capture_output=True, text=True)

        # 1) 최신 main만 얕게 clone (모델 갱신만 필요)
        run(["git", "clone", "--depth", "1", "--branch", "main", remote_url, repo_dir])

        # 2) clone된 레포 안에서 사용자 설정
        run(["git", "-C", repo_dir, "config", "user.email", "auto-retrain@trendsoccer.com"])
        run(["git", "-C", repo_dir, "config", "user.name", "TrendSoccer AutoTrain"])

        # 3) 학습된 models/ 를 clone 레포로 통째 복사 (기존 모델 폴더 교체)
        dst_models = os.path.join(repo_dir, "models")
        if os.path.isdir(dst_models):
            shutil.rmtree(dst_models)
        shutil.copytree("models", dst_models)

        # 4) 변경사항 확인
        run(["git", "-C", repo_dir, "add", "models/"])
        status = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            capture_output=True, text=True,
        )
        if not status.stdout.strip():
            return {"success": True, "message": "No changes to push"}

        # 5) commit & push
        commit_msg = f"Auto-retrain: {datetime.now().strftime('%Y-%m-%d %H:%M')} KST"
        run(["git", "-C", repo_dir, "commit", "-m", commit_msg])
        run(["git", "-C", repo_dir, "push", "origin", "HEAD:main"])

        return {"success": True, "message": commit_msg}

    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip() if hasattr(e, "stderr") else ""
        return {"success": False, "error": f"{e} | {detail}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
# API 엔드포인트
@app.get("/")
def root():
    return {
        "message": "Baseball AI Prediction API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/files")
def list_files():
    import os
    files = os.listdir(".")
    return {"files": sorted(files)}

@app.get("/health")
def health_check():
    loaded = {lg: lg in models for lg in SUPPORTED_LEAGUES}
    return {
        "status": "healthy" if any(loaded.values()) else "error",
        "models_loaded": loaded,
        "features": {lg: len(models[lg]['feature_columns']) for lg in models},
        "last_trained": retrain_status["last_trained"],
        "is_training": retrain_status["is_training"],
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        league = request.league or 'MLB'
        if league not in models:
            league = 'MLB'  # fallback

        m = models[league]
        win_model = m['win_model']
        over_model = m['over_model']
        feature_columns = m['feature_columns']

        if win_model is None or over_model is None:
            raise HTTPException(status_code=500, detail="Models not loaded")

        # KBO/NPB는 투수 feature 없음 → 있는 feature만 사용
        X = np.array([[request.features.get(col, 0.0) for col in feature_columns]])

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

        # DB에 ai_pick 저장
        if request.match_id:
            print(f"💾 DB 저장 시도: match_id={request.match_id}, grade={grade}")
            print(f"🔑 SUPABASE_URL: {SUPABASE_URL[:30] if SUPABASE_URL else 'EMPTY'}")
            print(f"🔑 SUPABASE_KEY: {'SET' if SUPABASE_KEY else 'EMPTY'}")
            await save_ai_pick_to_db(
                api_match_id=request.match_id,
                league=league,
                grade=grade,
                confidence=confidence,
                home_win_prob=home_win_prob,
                away_win_prob=1 - home_win_prob,
            )
            print(f"✅ DB 저장 완료: match_id={request.match_id}")

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
    background_tasks.add_task(run_retrain, request.min_games, request.league or "ALL")

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
