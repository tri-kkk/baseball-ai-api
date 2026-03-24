"""
MLB game_pk backfill 스크립트
- MLB Stats API 날짜별 schedule 조회 → gamePk + 팀명
- 우리 DB 날짜 + 팀명 매핑 → mlb_game_pk 저장
- 실행: python backfill_mlb_game_pk.py
"""

import requests
import time
from datetime import date, timedelta
from supabase import create_client

# =====================================================
# 설정 (본인 값으로 변경)
# =====================================================
SUPABASE_URL = "https://riqvjiiwjyynvhuynisv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJpcXZqaWl3anl5bnZodXluaXN2Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTYzODI0MiwiZXhwIjoyMDc3MjE0MjQyfQ.FgeTgtbg3RR1zZ9Qo1i7gtxwJ54jbRZK6HXxx9PaV-A"

START_DATE = date(2025, 9, 5)
END_DATE   = date(2026, 3, 10)

# =====================================================
# MLB Stats API 팀명 → DB 팀명 변환 테이블
# =====================================================
# 예외 변환만 정의 (나머지는 그대로 사용)
MLB_EXCEPTIONS: dict = {
    "St. Louis Cardinals": "St.Louis Cardinals",
    "American League":     None,
    "National League":     None,
}

def get_mlb_schedule(date_str: str) -> list[dict]:
    """MLB Stats API에서 날짜별 경기 목록 조회"""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        dates = data.get("dates", [])
        games = dates[0].get("games", []) if dates else []
        return games
    except Exception as e:
        print(f"  ⚠️ API 오류 ({date_str}): {e}")
        return []

def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    total_days = (END_DATE - START_DATE).days + 1
    total_matched = 0
    total_skipped = 0
    total_no_match = 0

    print(f"MLB game_pk backfill 시작")
    print(f"기간: {START_DATE} ~ {END_DATE} ({total_days}일)")
    print("=" * 60)

    current = START_DATE
    day_num = 0

    while current <= END_DATE:
        date_str = current.strftime("%Y-%m-%d")
        day_num += 1

        # MLB Stats API 조회
        games = get_mlb_schedule(date_str)

        if not games:
            current += timedelta(days=1)
            time.sleep(0.1)
            continue

        # 해당 날짜 DB 경기 조회
        db_result = supabase.from_("baseball_matches") \
            .select("api_match_id, home_team, away_team, mlb_game_pk") \
            .eq("match_date", date_str) \
            .eq("league", "MLB") \
            .execute()

        db_matches = db_result.data or []

        if not db_matches:
            current += timedelta(days=1)
            time.sleep(0.1)
            continue

        # DB 경기를 팀명 기준으로 빠르게 조회할 수 있도록 딕셔너리화
        # key: (home_team, away_team)
        db_map: dict[tuple, dict] = {}
        for m in db_matches:
            key = (m["home_team"], m["away_team"])
            db_map[key] = m

        matched_today = 0

        for game in games:
            game_pk: int = game.get("gamePk")
            home_raw: str = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away_raw: str = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")

            # 팀명 변환 (예외만 처리, 나머지 그대로)
            if home_raw in MLB_EXCEPTIONS:
                home_db = MLB_EXCEPTIONS[home_raw]
            else:
                home_db = home_raw
            if away_raw in MLB_EXCEPTIONS:
                away_db = MLB_EXCEPTIONS[away_raw]
            else:
                away_db = away_raw

            # 올스타전 등 None 팀 스킵
            if home_db is None or away_db is None:
                continue

            # DB에서 매칭
            db_match = db_map.get((home_db, away_db))

            if not db_match:
                # 매핑 실패 - 로그 출력 (팀명 변환 테이블 보완 필요)
                print(f"  ❌ 매핑 실패: {date_str} | {home_raw} vs {away_raw} → DB: {home_db} vs {away_db}")
                total_no_match += 1
                continue

            # 이미 채워진 경우 스킵
            if db_match.get("mlb_game_pk"):
                total_skipped += 1
                continue

            # mlb_game_pk 업데이트
            update_res = supabase.from_("baseball_matches") \
                .update({"mlb_game_pk": game_pk}) \
                .eq("api_match_id", db_match["api_match_id"]) \
                .execute()

            if update_res.data:
                matched_today += 1
                total_matched += 1
            else:
                print(f"  ⚠️ 업데이트 실패: api_match_id={db_match['api_match_id']}")

        if matched_today > 0 or day_num % 30 == 0:
            print(f"[{day_num}/{total_days}] {date_str}: {len(games)}경기 조회, {matched_today}개 매핑")

        current += timedelta(days=1)
        time.sleep(0.15)  # API rate limit 방지

    print("\n" + "=" * 60)
    print(f"완료!")
    print(f"  매핑 성공: {total_matched}개")
    print(f"  이미 존재: {total_skipped}개")
    print(f"  매핑 실패: {total_no_match}개 (팀명 변환 테이블 보완 필요)")

if __name__ == "__main__":
    main()
