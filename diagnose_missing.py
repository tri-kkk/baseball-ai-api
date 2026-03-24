"""
매핑 실패 경기 진단 - MLB Stats API에 실제로 있는지 확인
"""
import requests
import time
from supabase import create_client

SUPABASE_URL = "https://riqvjiiwjyynvhuynisv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJpcXZqaWl3anl5bnZodXluaXN2Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTYzODI0MiwiZXhwIjoyMDc3MjE0MjQyfQ.FgeTgtbg3RR1zZ9Qo1i7gtxwJ54jbRZK6HXxx9PaV-A"

def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 전체 누락 경기 (페이지네이션으로 전체 조회)
    all_missing = []
    offset = 0
    while True:
        result = supabase.from_("baseball_matches") \
            .select("api_match_id, home_team, away_team, match_date") \
            .eq("league", "MLB") \
            .eq("status", "FT") \
            .is_("mlb_game_pk", "null") \
            .order("match_date") \
            .range(offset, offset + 999) \
            .execute()
        batch = result.data or []
        all_missing.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    missing = all_missing
    print(f"전체 누락 {len(missing)}개 처리 중...\n")

    # DB 팀명 → API 팀명 변환 (St.Louis 예외)
    DB_TO_API = {
        "St.Louis Cardinals": "St. Louis Cardinals",
    }

    not_in_api = []
    found_different = []

    for m in missing:
        date_str = m["match_date"]
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        try:
            res = requests.get(url, timeout=10)
            data = res.json()
            dates = data.get("dates", [])
            games = dates[0].get("games", []) if dates else []

            # 해당 날짜 모든 팀명 수집
            api_matchups = {
                (g["teams"]["home"]["team"]["name"], g["teams"]["away"]["team"]["name"]): g["gamePk"]
                for g in games
            }

            home = DB_TO_API.get(m["home_team"], m["home_team"])
            away = DB_TO_API.get(m["away_team"], m["away_team"])

            if (home, away) in api_matchups:
                pk = api_matchups[(home, away)]
                found_different.append({**m, "gamePk": pk})
                pass  # 성공은 출력 생략
            else:
                # 팀명 부분 매칭 시도
                partial = [(h, a, pk) for (h, a), pk in api_matchups.items()
                           if home.split()[-1] in h or away.split()[-1] in a]
                if partial:
                    print(f"  🔶 부분매칭: {date_str} | DB:{home} vs {away}")
                    for h, a, pk in partial:
                        print(f"       API: {h} vs {a} → gamePk={pk}")
                else:
                    not_in_api.append(m)
                    print(f"  ❌ API에 없음: {date_str} | {home} vs {away}")

        except Exception as e:
            print(f"  ⚠️ 오류: {e}")

        time.sleep(0.2)

    print(f"\n{'='*50}")
    print(f"API에 있음 (업데이트 가능): {len(found_different)}개")
    print(f"API에 없음 (더블헤더 등): {len(not_in_api)}개")

    # 업데이트 가능한 것들 바로 저장
    if found_different:
        print(f"\n업데이트 가능한 {len(found_different)}개 저장 중...")
        for m in found_different:
            try:
                supabase.from_("baseball_matches") \
                    .update({"mlb_game_pk": m["gamePk"]}) \
                    .eq("api_match_id", m["api_match_id"]) \
                    .execute()
                print(f"  ✅ 저장: {m['home_team']} vs {m['away_team']} → {m['gamePk']}")
            except Exception as e:
                print(f"  ❌ 실패: {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    main()
