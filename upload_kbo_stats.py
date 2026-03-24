"""
KBO 투수 스탯 업로드 스크립트
사용법: python upload_kbo_stats.py

환경변수 필요:
  SUPABASE_URL=https://xxx.supabase.co
  SUPABASE_SERVICE_KEY=eyJ...
"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from pathlib import Path

# 스크립트 위치 기준으로 .env 로드
env_path = Path(__file__).parent / '.env.local'
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 환경변수를 찾을 수 없어요. .env 파일 확인해주세요.")
    print(f"   .env 경로: {env_path}")
    print(f"   SUPABASE_URL: {SUPABASE_URL}")
    raise SystemExit(1)
EXCEL_FILE   = "KBO_2025_투수스탯_완성.xlsx"
SEASON       = "2025"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 컬럼 매핑 (엑셀 → DB) ────────────────────────────
COL_MAP = {
    "선수명": "name",
    "팀명":   "team",
    "ERA":    "era",
    "G":      "games",
    "W":      "wins",
    "L":      "losses",
    "SV":     "saves",
    "HLD":    "holds",
    "WPCT":   "wpct",

    "H":      "hits",
    "HR":     "home_runs",
    "BB":     "walks",
    "HBP":    "hit_by_pitch",
    "SO":     "strikeouts",
    "R":      "runs",
    "ER":     "earned_runs",
    "WHIP":   "whip",
}

def load_all_sheets(filepath: str) -> pd.DataFrame:
    """전체 시트 + 팀별 시트 합쳐서 로드, 중복 제거"""
    xl = pd.ExcelFile(filepath)
    sheets_to_read = [s for s in xl.sheet_names if s not in ("📋 입력 가이드",)]

    frames = []
    for sheet in sheets_to_read:
        df = pd.read_excel(filepath, sheet_name=sheet, header=1, dtype=str)
        df = df.dropna(subset=["선수명"])
        df = df[df["선수명"].str.strip() != ""]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # 선수명+팀명 기준 중복 제거 (전체 시트가 있으면 팀별 시트와 겹침)
    combined = combined.drop_duplicates(subset=["선수명", "팀명"])
    return combined

def clean_row(row: pd.Series) -> dict:
    """숫자 변환 + 결측 처리"""
    r = {}
    for kr, en in COL_MAP.items():
        val = row.get(kr, None)
        if pd.isna(val) or str(val).strip() in ("", "-", "nan"):
            r[en] = None
        else:
            try:
                r[en] = float(str(val).replace(",", ""))
                # 정수형 컬럼
                if en in ("games", "wins", "losses", "saves", "holds",
                          "hits", "home_runs", "walks", "hit_by_pitch",
                          "strikeouts", "runs", "earned_runs"):
                    r[en] = int(r[en])
            except ValueError:
                r[en] = str(val).strip()
    r["season"] = SEASON
    r["league"] = "KBO"
    return r

def upload(df: pd.DataFrame):
    rows = [clean_row(row) for _, row in df.iterrows()]

    success = 0
    errors  = 0
    for r in rows:
        try:
            supabase.table("kbo_pitcher_stats").upsert(
                r,
                on_conflict="name,team,season"   # unique key
            ).execute()
            print(f"  ✅ {r['name']} ({r['team']})")
            success += 1
        except Exception as e:
            print(f"  ❌ {r.get('name')} — {e}")
            errors += 1

    print(f"\n완료: {success}명 업로드, {errors}건 오류")

if __name__ == "__main__":
    print(f"📂 파일 로드: {EXCEL_FILE}")
    df = load_all_sheets(EXCEL_FILE)
    print(f"📊 총 {len(df)}명 감지\n")
    upload(df)
