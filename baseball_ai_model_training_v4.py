# baseball_ai_model_training_v4.py
"""
Baseball AI 예측 모델 학습 스크립트 v4

변경사항 (v3 대비):
- MLB 전용 선발투수 feature 추가 (ERA, WHIP, K)
- KBO/NPB는 v3 feature set 유지 (투수 데이터 미수집)
- 리그별 FEATURE_COLUMNS 분리 관리
- 투수 null → 리그 평균으로 대체
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
REGULAR_SEASON_START = {'MLB': '2024-03-28', 'KBO': '2022-04-01', 'NPB': '2022-03-25'}
LEAGUES_TO_TRAIN = ['MLB', 'KBO', 'NPB'] if LEAGUE == 'ALL' else [LEAGUE]

# =======================
# 2. 환경 설정
# =======================

load_dotenv('.env.local')  # fallback for local
SUPABASE_URL = os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("오류: Supabase 환경변수가 설정되지 않았습니다!")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

WINDOW = 10

# 공통 feature (KBO/NPB/MLB 모두 사용)
BASE_FEATURE_COLUMNS = [
    'home_win_pct', 'home_avg_scored', 'home_avg_conceded',
    'home_avg_hits', 'home_home_win_pct', 'home_recent_form', 'home_run_diff',
    'away_win_pct', 'away_avg_scored', 'away_avg_conceded',
    'away_avg_hits', 'away_away_win_pct', 'away_recent_form', 'away_run_diff',
    'win_pct_diff', 'scored_diff', 'conceded_diff', 'form_diff', 'run_diff_diff',
    'total_avg_scored',
    'home_first_score_win_rate', 'home_comeback_rate', 'home_blown_lead_rate',
    'away_first_score_win_rate', 'away_comeback_rate', 'away_blown_lead_rate',
    'first_score_win_rate_diff', 'comeback_rate_diff',
]

# MLB 전용 투수 feature
MLB_PITCHER_FEATURES = [
    'home_pitcher_era', 'home_pitcher_whip', 'home_pitcher_k',
    'away_pitcher_era', 'away_pitcher_whip', 'away_pitcher_k',
    'pitcher_era_diff', 'pitcher_whip_diff', 'pitcher_k_diff',
]

# 리그별 feature set
FEATURE_COLUMNS_BY_LEAGUE = {
    'MLB': BASE_FEATURE_COLUMNS + MLB_PITCHER_FEATURES,
    'KBO': BASE_FEATURE_COLUMNS,
    'NPB': BASE_FEATURE_COLUMNS,
}

# 투수 null 대체값 (MLB 전용)
PITCHER_DEFAULTS = {
    'MLB': {'era': 4.20, 'whip': 1.30, 'k': 150},
}


# =======================
# 3. 이닝 데이터 파싱
# =======================

def parse_inning_stats(inning_data: dict) -> dict:
    if not inning_data:
        return {'first_scorer': None, 'home_comeback': False, 'away_comeback': False,
                'home_blown_lead': False, 'away_blown_lead': False}

    home_innings = inning_data.get('home', {})
    away_innings = inning_data.get('away', {})

    home_cumsum = 0
    away_cumsum = 0
    first_scorer = None
    home_ever_led = False
    away_ever_led = False

    for inning in range(1, 10):
        k = str(inning)
        h = home_innings.get(k)
        a = away_innings.get(k)
        if h is None or a is None:
            continue

        away_cumsum += (a or 0)
        if first_scorer is None and away_cumsum > 0:
            first_scorer = 'away'

        home_cumsum += (h or 0)
        if first_scorer is None and home_cumsum > 0:
            first_scorer = 'home'

        if home_cumsum > away_cumsum:
            home_ever_led = True
        elif away_cumsum > home_cumsum:
            away_ever_led = True

    home_won = home_cumsum > away_cumsum
    away_won = away_cumsum > home_cumsum

    return {
        'first_scorer': first_scorer,
        'home_comeback': home_won and away_ever_led,
        'away_comeback': away_won and home_ever_led,
        'home_blown_lead': away_won and home_ever_led,
        'away_blown_lead': home_won and away_ever_led,
    }


# =======================
# 4. 데이터 로드
# =======================

def load_data(league: str):
    print(f"\n{league} 데이터 로드 중...")
    all_matches = []
    page_size = 1000
    offset = 0

    # MLB만 투수 컬럼 추가 select
    pitcher_cols = (
        ', home_pitcher_era, home_pitcher_whip, home_pitcher_k,'
        ' away_pitcher_era, away_pitcher_whip, away_pitcher_k'
    ) if league == 'MLB' else ''

    while True:
        response = supabase.table('baseball_matches') \
            .select(
                f'id, league, season, match_date, '
                f'home_team, away_team, '
                f'home_score, away_score, '
                f'home_hits, away_hits, '
                f'inning, status{pitcher_cols}'
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

def get_team_rolling_stats(df: pd.DataFrame, window: int, league: str) -> pd.DataFrame:
    inning_stats_list = [parse_inning_stats(row.get('inning')) for _, row in df.iterrows()]
    df = df.copy()
    df['_first_scorer'] = [s['first_scorer'] for s in inning_stats_list]
    df['_home_comeback'] = [s['home_comeback'] for s in inning_stats_list]
    df['_away_comeback'] = [s['away_comeback'] for s in inning_stats_list]
    df['_home_blown'] = [s['home_blown_lead'] for s in inning_stats_list]
    df['_away_blown'] = [s['away_blown_lead'] for s in inning_stats_list]

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
        else:
            hg = pd.DataFrame()

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
        else:
            ag = pd.DataFrame()

        valid_frames = [f for f in [hg, ag] if len(f) > 0]
        if not valid_frames:
            return {
                'win_pct': 0.5, 'avg_scored': 4.5, 'avg_conceded': 4.5,
                'avg_hits': 8.0, 'home_win_pct': 0.5, 'away_win_pct': 0.5,
                'recent_form': 0.5, 'run_diff': 0.0, 'games_played': 0,
                'first_score_win_rate': 0.65, 'comeback_rate': 0.2, 'blown_lead_rate': 0.2,
            }

        all_games = pd.concat(valid_frames).sort_values('match_date').tail(window)
        home_only = all_games[all_games['is_home'] == 1]
        away_only = all_games[all_games['is_home'] == 0]
        recent_5 = all_games.tail(5)

        first_scored_games = all_games[all_games['first_scored'] == 1]
        first_score_win_rate = safe_mean(first_scored_games['won_after_first']) if len(first_scored_games) > 0 else 0.65

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
            'comeback_rate': safe_mean(all_games['comeback'], default=0.2),
            'blown_lead_rate': safe_mean(all_games['blown'], default=0.2),
        }

    records = []
    total = len(df)

    for idx, row in df.iterrows():
        home_stats = team_stats(row['home_team'], row['match_date'])
        away_stats = team_stats(row['away_team'], row['match_date'])

        record = {
            'id': row['id'],
            'match_date': row['match_date'],
            'home_score': row['home_score'],
            'away_score': row['away_score'],
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
            'home_first_score_win_rate': home_stats['first_score_win_rate'],
            'home_comeback_rate': home_stats['comeback_rate'],
            'home_blown_lead_rate': home_stats['blown_lead_rate'],
            'away_first_score_win_rate': away_stats['first_score_win_rate'],
            'away_comeback_rate': away_stats['comeback_rate'],
            'away_blown_lead_rate': away_stats['blown_lead_rate'],
            'first_score_win_rate_diff': home_stats['first_score_win_rate'] - away_stats['first_score_win_rate'],
            'comeback_rate_diff': home_stats['comeback_rate'] - away_stats['comeback_rate'],
        }

        # MLB만 투수 stats 추가
        if league == 'MLB':
            record.update({
                'home_pitcher_era': row.get('home_pitcher_era'),
                'home_pitcher_whip': row.get('home_pitcher_whip'),
                'home_pitcher_k': row.get('home_pitcher_k'),
                'away_pitcher_era': row.get('away_pitcher_era'),
                'away_pitcher_whip': row.get('away_pitcher_whip'),
                'away_pitcher_k': row.get('away_pitcher_k'),
            })

        records.append(record)

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

    inning_coverage = df['inning'].notna().mean()
    print(f"이닝 데이터 커버리지: {inning_coverage:.1%}")

    if league == 'MLB':
        pitcher_coverage = df['home_pitcher_era'].notna().mean()
        print(f"투수 데이터 커버리지: {pitcher_coverage:.1%}")
    else:
        pitcher_coverage = 0.0
        print(f"투수 데이터: 미수집 (추후 추가 예정)")

    print("Rolling feature 생성 중...")
    df_features = get_team_rolling_stats(df, WINDOW, league)

    # 파생 feature
    df_features['win_pct_diff'] = df_features['home_win_pct'] - df_features['away_win_pct']
    df_features['scored_diff'] = df_features['home_avg_scored'] - df_features['away_avg_scored']
    df_features['conceded_diff'] = df_features['home_avg_conceded'] - df_features['away_avg_conceded']
    df_features['form_diff'] = df_features['home_recent_form'] - df_features['away_recent_form']
    df_features['run_diff_diff'] = df_features['home_run_diff'] - df_features['away_run_diff']
    df_features['total_avg_scored'] = df_features['home_avg_scored'] + df_features['away_avg_scored']

    # MLB 전용: 투수 null 처리 + 파생 feature
    if league == 'MLB':
        defaults = PITCHER_DEFAULTS['MLB']
        df_features['home_pitcher_era'] = df_features['home_pitcher_era'].fillna(defaults['era'])
        df_features['away_pitcher_era'] = df_features['away_pitcher_era'].fillna(defaults['era'])
        df_features['home_pitcher_whip'] = df_features['home_pitcher_whip'].fillna(defaults['whip'])
        df_features['away_pitcher_whip'] = df_features['away_pitcher_whip'].fillna(defaults['whip'])
        df_features['home_pitcher_k'] = df_features['home_pitcher_k'].fillna(defaults['k'])
        df_features['away_pitcher_k'] = df_features['away_pitcher_k'].fillna(defaults['k'])
        # ERA/WHIP 낮을수록 유리 → 원정 - 홈 (양수 = 홈 유리)
        df_features['pitcher_era_diff'] = df_features['away_pitcher_era'] - df_features['home_pitcher_era']
        df_features['pitcher_whip_diff'] = df_features['away_pitcher_whip'] - df_features['home_pitcher_whip']
        df_features['pitcher_k_diff'] = df_features['home_pitcher_k'] - df_features['away_pitcher_k']

    feature_columns = FEATURE_COLUMNS_BY_LEAGUE[league]
    over_line = OVER_LINE.get(league, 8.5)
    df_features['home_win'] = (df_features['home_score'] > df_features['away_score']).astype(int)
    df_features['total_runs'] = df_features['home_score'] + df_features['away_score']
    df_features['over'] = (df_features['total_runs'] > over_line).astype(int)

    df_model = df_features[
        (df_features['home_games_played'] >= 5) &
        (df_features['away_games_played'] >= 5)
    ].copy()

    print(f"학습 데이터: {len(df_model)}개 | 홈승률: {df_model['home_win'].mean():.1%}")
    print(f"사용 feature: {len(feature_columns)}개 {'(투수 포함)' if league == 'MLB' else '(투수 제외)'}")

    X = df_model[feature_columns].fillna(0)
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

    print(f"\nFeature Importance 상위 10개:")
    fi = pd.DataFrame({
        'feature': feature_columns,
        'importance': win_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in fi.head(10).iterrows():
        bar = '|' * int(row['importance'] * 200)
        print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")

    model_dir = f'models/{league}'
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(win_model, f'{model_dir}/baseball_win_model.pkl')
    joblib.dump(over_model, f'{model_dir}/baseball_over_model.pkl')
    joblib.dump(feature_columns, f'{model_dir}/feature_columns.pkl')

    metadata = {
        'trained_at': datetime.now().isoformat(),
        'version': 'v4',
        'league': league,
        'over_line': over_line,
        'window': WINDOW,
        'features': feature_columns,
        'pitcher_features_included': league == 'MLB',
        'train_size': len(X_train),
        'test_size': len(X_test),
        'win_accuracy': float(win_acc),
        'win_auc': float(win_auc),
        'over_accuracy': float(over_acc),
        'over_auc': float(over_auc),
        'inning_coverage': float(inning_coverage),
        'pitcher_coverage': float(pitcher_coverage),
    }
    with open(f'{model_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ {league} 모델 저장: {model_dir}/")
    return metadata


# =======================
# 메인 실행
# =======================

print(f"Baseball AI 모델 학습 v4")
print(f"리그: {LEAGUE} | 최소경기: {MIN_GAMES}")
print(f"  MLB: 투수 feature 포함 ({len(FEATURE_COLUMNS_BY_LEAGUE['MLB'])}개)")
print(f"  KBO/NPB: 투수 feature 제외 ({len(FEATURE_COLUMNS_BY_LEAGUE['KBO'])}개)")
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
    pitcher_note = " (투수 포함)" if meta['pitcher_features_included'] else " (투수 제외)"
    print(f"  {league}{pitcher_note}: 승부 AUC {meta['win_auc']:.3f} ({quality}) | 총점 AUC {meta['over_auc']:.3f}")
