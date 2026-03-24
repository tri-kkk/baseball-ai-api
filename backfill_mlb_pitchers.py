"""
MLB 선발투수 backfill - mlb_game_pk 기반
- DB에서 mlb_game_pk 있는 경기 전체 조회
- MLB Stats API schedule?gamePk= 로 probablePitcher 조회
- home/away_starting_pitcher, ERA, WHIP, K 저장
- 실행: python backfill_mlb_pitchers.py
"""

import requests
import time
from supabase import create_client

SUPABASE_URL = "https://riqvjiiwjyynvhuynisv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJpcXZqaWl3anl5bnZodXluaXN2Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTYzODI0MiwiZXhwIjoyMDc3MjE0MjQyfQ.FgeTgtbg3RR1zZ9Qo1i7gtxwJ54jbRZK6HXxx9PaV-A"

def get_pitcher_stats(player_id: int, fallback_name: str, season: int) -> dict:
    """선수 ID로 시즌 스탯 조회"""
    try:
        res = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=pitching&season={season}",
            timeout=10
        )
        if not res.ok:
            return {"name": fallback_name, "era": None, "whip": None, "strikeouts": None}

        data = res.json()
        stats = data.get("stats", [{}])[0].get("splits", [{}])[0].get("stat", {})
        full_name = data.get("people", [{}])[0].get("fullName", fallback_name)

        if stats:
            return {
                "name": full_name or fallback_name,
                "era": float(stats["era"]) if stats.get("era") else None,
                "whip": float(stats["whip"]) if stats.get("whip") else None,
                "strikeouts": stats.get("strikeOuts"),
            }
    except Exception as e:
        print(f"    ⚠️ 스탯 조회 실패 ({player_id}): {e}")

    return {"name": fallback_name, "era": None, "whip": None, "strikeouts": None}

def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 전체 경기 조회 (mlb_game_pk 있는 것, 페이지네이션)
    print("DB에서 경기 목록 로드 중...")
    all_matches = []
    offset = 0
    while True:
        result = supabase.from_("baseball_matches") \
            .select("api_match_id, mlb_game_pk, home_team, away_team, match_date, home_starting_pitcher") \
            .eq("league", "MLB") \
            .eq("status", "FT") \
            .filter("mlb_game_pk", "not.is", "null") \
            .order("match_date") \
            .range(offset, offset + 999) \
            .execute()
        batch = result.data or []
        all_matches.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    # 이미 투수 데이터 있는 경기 제외
    missing = [m for m in all_matches if not m.get("home_starting_pitcher")]
    print(f"전체: {len(all_matches)}개 | 투수 미입력: {len(missing)}개\n")
    print("=" * 60)

    success = 0
    skipped = 0
    errors = 0
    current_year = 2026

    for i, match in enumerate(missing):
        game_pk = match["mlb_game_pk"]
        api_match_id = match["api_match_id"]

        try:
            # gamePk로 선발투수 조회
            res = requests.get(
                f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePk={game_pk}&hydrate=probablePitcher",
                timeout=10
            )
            if not res.ok:
                skipped += 1
                continue

            data = res.json()
            dates = data.get("dates", [])
            game = dates[0].get("games", [{}])[0] if dates else {}

            home_pitcher_info = game.get("teams", {}).get("home", {}).get("probablePitcher", {})
            away_pitcher_info = game.get("teams", {}).get("away", {}).get("probablePitcher", {})

            home_id = home_pitcher_info.get("id")
            away_id = away_pitcher_info.get("id")
            home_name = home_pitcher_info.get("fullName", "")
            away_name = away_pitcher_info.get("fullName", "")

            if not home_id and not away_id:
                skipped += 1
                continue

            # 경기 시즌 파악
            match_year = int(match["match_date"][:4])

            # 스탯 조회
            home_stats = get_pitcher_stats(home_id, home_name, match_year) if home_id else {"name": home_name, "era": None, "whip": None, "strikeouts": None}
            away_stats = get_pitcher_stats(away_id, away_name, match_year) if away_id else {"name": away_name, "era": None, "whip": None, "strikeouts": None}

            # DB 업데이트
            update_data = {}
            if home_stats.get("name"):
                update_data["home_starting_pitcher"] = home_stats["name"]
                if home_stats["era"] is not None: update_data["home_pitcher_era"] = home_stats["era"]
                if home_stats["whip"] is not None: update_data["home_pitcher_whip"] = home_stats["whip"]
                if home_stats["strikeouts"] is not None: update_data["home_pitcher_k"] = home_stats["strikeouts"]
            if away_stats.get("name"):
                update_data["away_starting_pitcher"] = away_stats["name"]
                if away_stats["era"] is not None: update_data["away_pitcher_era"] = away_stats["era"]
                if away_stats["whip"] is not None: update_data["away_pitcher_whip"] = away_stats["whip"]
                if away_stats["strikeouts"] is not None: update_data["away_pitcher_k"] = away_stats["strikeouts"]

            if update_data:
                for retry in range(3):
                    try:
                        supabase.from_("baseball_matches") \
                            .update(update_data) \
                            .eq("api_match_id", api_match_id) \
                            .execute()
                        success += 1
                        break
                    except Exception as e:
                        if retry < 2:
                            time.sleep(3)
                        else:
                            print(f"  ❌ 저장 실패: {api_match_id} - {e}")
                            errors += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"  ⚠️ 오류 ({game_pk}): {e}")
            errors += 1

        # 진행 상황 출력 (100개마다)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(missing)}] 성공: {success} | 스킵: {skipped} | 오류: {errors}")

        time.sleep(0.2)  # API rate limit

    print(f"\n{'='*60}")
    print(f"완료!")
    print(f"  성공: {success}개")
    print(f"  스킵 (투수 미발표): {skipped}개")
    print(f"  오류: {errors}개")

if __name__ == "__main__":
    main()
