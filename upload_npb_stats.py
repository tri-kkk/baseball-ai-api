"""
NPB 투수 스탯 업로드 스크립트
사용법: python upload_npb_stats.py

환경변수 필요 (.env.local):
  SUPABASE_URL / NEXT_PUBLIC_SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY / SUPABASE_SERVICE_KEY
"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / '.env.local'
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 환경변수를 찾을 수 없어요.")
    raise SystemExit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
EXCEL_FILE = "NPB_2025_투수스탯.xlsx"
SEASON     = "2025"

def load_data(filepath):
    xl = pd.ExcelFile(filepath)
    frames = []
    for sheet in xl.sheet_names:
        if sheet in ('📋 입력 가이드',):
            continue
        df = pd.read_excel(filepath, sheet_name=sheet, header=1, dtype=str)
        df = df.dropna(subset=['선수명'])
        df = df[df['선수명'].str.strip() != '']
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=['선수명', '팀명'])

def to_num(val, is_int=False):
    if pd.isna(val) or str(val).strip() in ('', '-', 'nan', ''):
        return None
    try:
        f = float(str(val).replace(',', ''))
        return int(f) if is_int else f
    except:
        return None

def clean_row(row):
    return {
        'name':              str(row.get('선수명', '')).strip(),
        'team':              str(row.get('팀명', '')).strip(),
        'season':            SEASON,
        'league':            'NPB',
        'pitch_hand':        str(row.get('투구손', '')).strip() or None,
        'games':             to_num(row.get('등판'), True),
        'wins':              to_num(row.get('승리'), True),
        'losses':            to_num(row.get('패배'), True),
        'saves':             to_num(row.get('세이브'), True),
        'holds':             to_num(row.get('홀드'), True),
        'hp':                to_num(row.get('HP'), True),
        'complete_games':    to_num(row.get('완투'), True),
        'shutouts':          to_num(row.get('완봉승'), True),
        'no_walks':          to_num(row.get('무사구'), True),
        'wpct':              to_num(row.get('승률')),
        'batters_faced':     to_num(row.get('타자'), True),
        'hits':              to_num(row.get('안타'), True),
        'home_runs':         to_num(row.get('홈런'), True),
        'walks':             to_num(row.get('사구'), True),
        'intentional_walks': to_num(row.get('고의4구'), True),
        'hit_by_pitch':      to_num(row.get('사구(HBP)'), True),
        'strikeouts':        to_num(row.get('삼진'), True),
        'wild_pitches':      to_num(row.get('폭투'), True),
        'balks':             to_num(row.get('보크'), True),
        'runs':              to_num(row.get('실점'), True),
        'earned_runs':       to_num(row.get('자책점'), True),
        'era':               to_num(row.get('방어율')),
    }

def upload(df):
    success = errors = 0
    for _, row in df.iterrows():
        r = clean_row(row)
        if not r['name'] or not r['team']:
            continue
        try:
            supabase.table('npb_pitcher_stats').upsert(
                r, on_conflict='name,team,season'
            ).execute()
            print(f"  ✅ {r['name']} ({r['team']}) {'좌' if r['pitch_hand'] == '좌' else '우'}투")
            success += 1
        except Exception as e:
            print(f"  ❌ {r.get('name')} — {e}")
            errors += 1
    print(f"\n완료: {success}명 업로드, {errors}건 오류")

if __name__ == '__main__':
    print(f"📂 파일 로드: {EXCEL_FILE}")
    df = load_data(EXCEL_FILE)
    print(f"📊 총 {len(df)}명 감지\n")
    upload(df)
