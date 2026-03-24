# baseball_ai_model_training_v3.py
"""
Baseball AI 예측 모델 학습 스크립트 v3

변경사항 (v2 대비):
- MLB/KBO/NPB 멀티 리그 지원
- 리그별 모델 파일 분리 저장 (models/MLB/, models/KBO/, models/NPB/)
- 선취점 승률 / 역전율 / 리드 지킴 feature 추가
- 오버 기준: MLB 9.0, KBO 8.5, NPB 8.5
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# =======================
# 1. 인자 파싱
# =======================

parser = argparse.ArgumentParser()
parser.add_argument('--league', type=str, default='MLB', choices=['MLB', 'KBO', 'NPB', 'ALL'])
parser.add_argument('--min-games', type=int, default=100)
args = parser.parse_args()

LEAGUE = args.league
MIN_GAMES = args.min_games

OVER_LINE = {'MLB': 9.0, 'KBO': 8.5, 'NPB': 8.5}

# 리그별 정규시즌 시작 기준 (스프링 트레이닝 제외)
REGULAR_SEASON_START = {'MLB': '2024-03-28', 'KBO': '2022-04-01', 'NPB': '2022-03-25'}
LEAGUES_TO_TRAIN = ['MLB', 'KBO', 'NPB'] if LEAGUE == 'ALL' else [LEAGUE]

# =======================
# 2. 환경 설정
# =======================

load_dotenv('.env.local')
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("오류: .env.local 파일에 Supabase 설정이 없습니다!")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

WINDOW = 10
FEATURE_COLUMNS = [
    # 기본 stats
    'home_win_pct', 'home_avg_scored', 'home_avg_conceded',
    'home_avg_hits', 'home_home_win_pct', 'home_recent_form', 'home_run_diff',
    'away_win_pct', 'away_avg_scored', 'away_avg_conceded',
    'away_avg_hits', 'away_away_win_pct', 'away_recent_form', 'away_run_diff',
    'win_pct_diff', 'scored_diff', 'conceded_diff', 'form_diff', 'run_diff_diff',
    'total_avg_scored',
    # 선취점 / 역전 stats
    'home_first_score_win_rate',   # 홈팀이 선취점 시 승률
    'home_comeback_rate',          # 홈팀 역전 성공률 (뒤지다가 이긴 비율)
    'home_blown_lead_rate',        # 홈팀 역전당한 비율 (앞서다가 진 비율)
    'away_first_score_win_rate',   # 원정팀이 선취점 시 승률
    'away_comeback_rate',          # 원정팀 역전 성공률
    'away_blown_lead_rate',        # 원정팀 역전당한 비율
    'first_score_win_rate_diff',   # 선취점 승률 차이
    'comeback_rate_diff',          # 역전율 차이
]


# =======================
# 3. 이닝 데이터 파싱
# =======================

def parse_inning_stats(inning_data: dict) -> dict:
    """
    이닝 데이터에서 선취점팀, 역전 여부 추출
    반환: { first_scorer: 'home'|'away'|None, had_comeback: bool, home_led: bool, away_led: bool }
    """
    if not inning_data:
        return {'first_scorer': None, 'home_comeback': False, 'away_comeback': False, 'home_blown_lead': False, 'away_blown_lead': False}

    home_innings = inning_data.get('home', {})
    away_innings = inning_data.get('away', {})

    # 이닝별 누적 점수 계산
    home_cumsum = 0
    away_cumsum = 0
    first_scorer = None
    home_ever_led = False
    away_ever_led = False

    for inning in range(1, 10):
        k = str(inning)
        h = home_innings.get(k)
        a = away_innings.get(k)

        # null 이닝 스킵 (연장 등)
        if h is None or a is None:
            continue

        # 원정팀 공격 먼저 (야구 규칙: 어웨이 선공)
        away_cumsum += (a or 0)
        if first_scorer is None and away_cumsum > 0:
            first_scorer = 'away'

        home_cumsum += (h or 0)
        if first_scorer is None and home_cumsum > 0:
            first_scorer = 'home'

        # 리드 추적
        if home_cumsum > away_cumsum:
            home_ever_led = True
        elif away_cumsum > home_cumsum:
            away_ever_led = True

    # 최종 스코어로 역전 판정
    # 홈 역전: 원정이 리드한 적 있는데 홈이 최종 승리
    # 원정 역전: 홈이 리드한 적 있는데 원정이 최종 승리
    home_won = home_cumsum > away_cumsum
    away_won = away_cumsum > home_cumsum

    return {
        'first_scorer': first_scorer,
        'home_comeback': home_won and away_ever_led,  # 홈팀 역전승
        'away_comeback': away_won and home_ever_led,  # 원정팀 역전승
        'home_blown_lead': away_won and home_ever_led,  # 홈팀 역전당함
        'away_blown_lead': home_won and away_ever_led,  # 원정팀 역전당함
    }


# =======================
# 4. 데이터 로드
# =======================

def load_data(league: str):
    print(f"\n{league} 데이터 로드 중...")
    all_matches = []
    page_size = 1000
    offset = 0

    while True:
        response = supabase.table('baseball_matches') \
            .select(
                'id, league, season, match_date, '
                'home_team, away_team, '
                'home_score, away_score, '
                'home_hits, away_hits, '
                'inning, status'
            ) \
            .eq('status', 'FT') \
            .eq('league', league) \
            .not_.is_('home_score', 'null') \
            .not_.is_('away_score', 'null') \
            .gte('match_date', REGULAR_SEASON_START.get(league, '2022-01-01')) \
            .order('match_date', desc=False) \
            .range(offset, offset + page_size - 1) \
            .execute()

        batch = response.data
        if not batch:
            break
        all_matches.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    print(f"  {len(all_matches)}개 경기 로드 완료")
    return all_matches


# =======================
# 5. Rolling Feature 생성
# =======================

def get_team_rolling_stats(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling feature + 선취점/역전 feature 생성"""

    # 이닝 stats 미리 계산
    inning_stats_list = []
    for _, row in df.iterrows():
        stats = parse_inning_stats(row.get('inning'))
        inning_stats_list.append(stats)
    df = df.copy()
    df['_first_scorer'] = [s['first_scorer'] for s in inning_stats_list]
    df['_home_comeback'] = [s['home_comeback'] for s in inning_stats_list]
    df['_away_comeback'] = [s['away_comeback'] for s in inning_stats_list]
    df['_home_blown'] = [s['home_blown_lead'] for s in inning_stats_list]
    df['_away_blown'] = [s['away_blown_lead'] for s in inning_stats_list]

    # 팀별 경기 인덱싱
    team_home_games = {}
    team_away_games = {}

    for team in pd.concat([df['home_team'], df['away_team']]).unique():
        team_home_games[team] = df[df['home_team'] == team].copy()
        team_away_games[team] = df[df['away_team'] == team].copy()

    def safe_mean(series, default=0.5):
        return series.mean() if len(series) > 0 else default

    def team_stats(team: str, before_date: pd.Timestamp) -> dict:
        hg = team_home_games.get(team, pd.DataFrame())
        if len(hg) > 0:
            hg = hg[hg['match_date'] < before_date].tail(window).copy()
            hg['scored'] = hg['home_score']
            hg['conceded'] = hg['away_score']
            hg['hits'] = hg['home_hits']
            hg['won'] = (hg['scored'] > hg['conceded']).astype(int)
            hg['is_home'] = 1
            hg['first_scored'] = (hg['_first_scorer'] == 'home').astype(int)
            hg['won_after_first'] = ((hg['_first_scorer'] == 'home') & (hg['won'] == 1)).astype(int)
            hg['comeback'] = hg['_home_comeback'].astype(int)
            hg['blown'] = hg['_home_blown'].astype(int)

        ag = team_away_games.get(team, pd.DataFrame())
        if len(ag) > 0:
            ag = ag[ag['match_date'] < before_date].tail(window).copy()
            ag['scored'] = ag['away_score']
            ag['conceded'] = ag['home_score']
            ag['hits'] = ag['away_hits']
            ag['won'] = (ag['scored'] > ag['conceded']).astype(int)
            ag['is_home'] = 0
            ag['first_scored'] = (ag['_first_scorer'] == 'away').astype(int)
            ag['won_after_first'] = ((ag['_first_scorer'] == 'away') & (ag['won'] == 1)).astype(int)
            ag['comeback'] = ag['_away_comeback'].astype(int)
            ag['blown'] = ag['_away_blown'].astype(int)

        valid_frames = [f for f in [hg, ag] if len(f) > 0]
        if not valid_frames:
            return {
                'win_pct': 0.5, 'avg_scored': 4.5, 'avg_conceded': 4.5,
                'avg_hits': 8.0, 'home_win_pct': 0.5, 'away_win_pct': 0.5,
                'recent_form': 0.5, 'run_diff': 0.0, 'games_played': 0,
                'first_score_win_rate': 0.5, 'comeback_rate': 0.2, 'blown_lead_rate': 0.2,
            }

        all_games = pd.concat(valid_frames).sort_values('match_date').tail(window)
        home_only = all_games[all_games['is_home'] == 1]
        away_only = all_games[all_games['is_home'] == 0]
        recent_5 = all_games.tail(5)

        # 선취점 시 승률
        first_scored_games = all_games[all_games['first_scored'] == 1]
        first_score_win_rate = safe_mean(first_scored_games['won_after_first']) if len(first_scored_games) > 0 else 0.65

        # 역전율 (comeback: 뒤지다 이긴 비율)
        lost_lead_games = all_games[all_games['won'] == 0]  # 진 경기 중
        comeback_rate = safe_mean(all_games['comeback'], default=0.2)

        # 역전당한 비율
        blown_lead_rate = safe_mean(all_games['blown'], default=0.2)

        return {
            'win_pct': safe_mean(all_games['won']),
            'avg_scored': safe_mean(all_games['scored'], 4.5),
            'avg_conceded': safe_mean(all_games['conceded'], 4.5),
            'avg_hits': all_games['hits'].mean() if all_games['hits'].notna().any() else 8.0,
            'home_win_pct': safe_mean(home_only['won']) if len(home_only) > 0 else 0.5,
            'away_win_pct': safe_mean(away_only['won']) if len(away_only) > 0 else 0.5,
            'recent_form': safe_mean(recent_5['won']) if len(recent_5) > 0 else 0.5,
            'run_diff': (all_games['scored'] - all_games['conceded']).mean(),
            'games_played': len(all_games),
            'first_score_win_rate': first_score_win_rate,
            'comeback_rate': comeback_rate,
            'blown_lead_rate': blown_lead_rate,
        }

    records = []
    total = len(df)

    for idx, row in df.iterrows():
        home_stats = team_stats(row['home_team'], row['match_date'])
        away_stats = team_stats(row['away_team'], row['match_date'])

        records.append({
            'id': row['id'],
            'match_date': row['match_date'],
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            # 기본 stats
            'home_win_pct': home_stats['win_pct'],
            'home_avg_scored': home_stats['avg_scored'],
            'home_avg_conceded': home_stats['avg_conceded'],
            'home_avg_hits': home_stats['avg_hits'],
            'home_home_win_pct': home_stats['home_win_pct'],
            'home_recent_form': home_stats['recent_form'],
            'home_run_diff': home_stats['run_diff'],
            'home_games_played': home_stats['games_played'],
            'away_win_pct': away_stats['win_pct'],
            'away_avg_scored': away_stats['avg_scored'],
            'away_avg_conceded': away_stats['avg_conceded'],
            'away_avg_hits': away_stats['avg_hits'],
            'away_away_win_pct': away_stats['away_win_pct'],
            'away_recent_form': away_stats['recent_form'],
            'away_run_diff': away_stats['run_diff'],
            'away_games_played': away_stats['games_played'],
            # 선취점 / 역전 stats
            'home_first_score_win_rate': home_stats['first_score_win_rate'],
            'home_comeback_rate': home_stats['comeback_rate'],
            'home_blown_lead_rate': home_stats['blown_lead_rate'],
            'away_first_score_win_rate': away_stats['first_score_win_rate'],
            'away_comeback_rate': away_stats['comeback_rate'],
            'away_blown_lead_rate': away_stats['blown_lead_rate'],
            'first_score_win_rate_diff': home_stats['first_score_win_rate'] - away_stats['first_score_win_rate'],
            'comeback_rate_diff': home_stats['comeback_rate'] - away_stats['comeback_rate'],
        })

        if (idx + 1) % 500 == 0:
            print(f"  처리 중: {idx + 1}/{total}...")

    print(f"  처리 완료: {total}/{total}")
    return pd.DataFrame(records)


# =======================
# 6. 모델 학습
# =======================

def train_league(league: str):
    print(f"\n{'='*50}")
    print(f"  {league} 모델 학습")
    print(f"{'='*50}")

    all_matches = load_data(league)
    if len(all_matches) < MIN_GAMES:
        print(f"  데이터 부족: {len(all_matches)}개 (최소 {MIN_GAMES}개) - 스킵")
        return None

    df = pd.DataFrame(all_matches)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)
    print(f"날짜 범위: {df['match_date'].min().date()} ~ {df['match_date'].max().date()}")

    # 이닝 데이터 있는 경기 비율
    inning_coverage = df['inning'].notna().mean()
    print(f"이닝 데이터 커버리지: {inning_coverage:.1%}")

    print("Rolling feature 생성 중...")
    df_features = get_team_rolling_stats(df, WINDOW)

    # 파생 feature
    df_features['win_pct_diff'] = df_features['home_win_pct'] - df_features['away_win_pct']
    df_features['scored_diff'] = df_features['home_avg_scored'] - df_features['away_avg_scored']
    df_features['conceded_diff'] = df_features['home_avg_conceded'] - df_features['away_avg_conceded']
    df_features['form_diff'] = df_features['home_recent_form'] - df_features['away_recent_form']
    df_features['run_diff_diff'] = df_features['home_run_diff'] - df_features['away_run_diff']
    df_features['total_avg_scored'] = df_features['home_avg_scored'] + df_features['away_avg_scored']

    over_line = OVER_LINE.get(league, 8.5)
    df_features['home_win'] = (df_features['home_score'] > df_features['away_score']).astype(int)
    df_features['total_runs'] = df_features['home_score'] + df_features['away_score']
    df_features['over'] = (df_features['total_runs'] > over_line).astype(int)

    df_model = df_features[
        (df_features['home_games_played'] >= 5) &
        (df_features['away_games_played'] >= 5)
    ].copy()

    print(f"학습 데이터: {len(df_model)}개 | 홈승률: {df_model['home_win'].mean():.1%}")

    X = df_model[FEATURE_COLUMNS].fillna(0)
    y_win = df_model['home_win']
    y_over = df_model['over']

    X_train, X_test, y_win_train, y_win_test = train_test_split(
        X, y_win, test_size=0.2, random_state=42, shuffle=False
    )
    _, _, y_over_train, y_over_test = train_test_split(
        X, y_over, test_size=0.2, random_state=42, shuffle=False
    )

    print(f"학습: {len(X_train)}개 / 검증: {len(X_test)}개")

    win_model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    win_model.fit(X_train, y_win_train)

    over_model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    over_model.fit(X_train, y_over_train)

    win_auc = roc_auc_score(y_win_test, win_model.predict_proba(X_test)[:, 1])
    win_acc = accuracy_score(y_win_test, win_model.predict(X_test))
    over_auc = roc_auc_score(y_over_test, over_model.predict_proba(X_test)[:, 1])
    over_acc = accuracy_score(y_over_test, over_model.predict(X_test))

    print(f"승부 예측 - 정확도: {win_acc:.2%}  AUC: {win_auc:.3f}")
    print(f"총점 예측 - 정확도: {over_acc:.2%}  AUC: {over_auc:.3f}")

    # Feature Importance 상위 10개
    print(f"\nFeature Importance 상위 10개:")
    fi = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': win_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in fi.head(10).iterrows():
        bar = '|' * int(row['importance'] * 200)
        print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")

    # 모델 저장
    model_dir = f'models/{league}'
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(win_model, f'{model_dir}/baseball_win_model.pkl')
    joblib.dump(over_model, f'{model_dir}/baseball_over_model.pkl')
    joblib.dump(FEATURE_COLUMNS, f'{model_dir}/feature_columns.pkl')

    metadata = {
        'trained_at': datetime.now().isoformat(),
        'version': 'v3',
        'league': league,
        'over_line': over_line,
        'window': WINDOW,
        'features': FEATURE_COLUMNS,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'win_accuracy': float(win_acc),
        'win_auc': float(win_auc),
        'over_accuracy': float(over_acc),
        'over_auc': float(over_auc),
        'inning_coverage': float(inning_coverage),
    }
    with open(f'{model_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ {league} 모델 저장: {model_dir}/")
    return metadata


# =======================
# 메인 실행
# =======================

print(f"Baseball AI 모델 학습 v3")
print(f"리그: {LEAGUE} | 최소경기: {MIN_GAMES}")
print(f"시작: {datetime.now()}")

results = {}
for league in LEAGUES_TO_TRAIN:
    result = train_league(league)
    if result:
        results[league] = result

print(f"\n{'='*50}")
print(f"전체 완료: {datetime.now()}")
for league, meta in results.items():
    quality = "양호" if meta['win_auc'] > 0.55 else "보통" if meta['win_auc'] > 0.52 else "미흡"
    print(f"  {league}: 승부 AUC {meta['win_auc']:.3f} ({quality}) | 총점 AUC {meta['over_auc']:.3f}")
