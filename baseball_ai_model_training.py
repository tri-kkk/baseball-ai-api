# baseball_ai_model_training.py
"""
⚾ Baseball AI 예측 모델 학습 스크립트 (Supabase Client 버전)

목적:
1. 승부 예측 (홈팀 승리 확률)
2. 총점 예측 (Over/Under 8.5)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# =======================
# 1. 환경 설정
# =======================

# .env.local 로드
load_dotenv('.env.local')

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 오류: .env.local 파일에 Supabase 설정이 없습니다!")
    print("필요한 환경변수:")
    print("  - NEXT_PUBLIC_SUPABASE_URL")
    print("  - SUPABASE_SERVICE_ROLE_KEY")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("⚾ Baseball AI 모델 학습 시작...")
print(f"시작 시간: {datetime.now()}")
print(f"Supabase URL: {SUPABASE_URL[:30]}...")

# =======================
# 2. 데이터 로드
# =======================

print("\n📊 데이터 로드 중...")

# 2.1 경기 데이터 로드 (페이지네이션)
print("  경기 데이터 로딩...")
all_matches = []
page_size = 1000
offset = 0

while True:
    matches_response = supabase.table('baseball_matches') \
        .select('*') \
        .in_('season', ['2024', '2025']) \
        .eq('status', 'FT') \
        .not_.is_('home_score', 'null') \
        .not_.is_('home_hits', 'null') \
        .range(offset, offset + page_size - 1) \
        .execute()
    
    batch = matches_response.data
    if not batch:
        break
    
    all_matches.extend(batch)
    print(f"    로드됨: {len(all_matches)}개...")
    
    if len(batch) < page_size:
        break
    
    offset += page_size

matches_data = all_matches
print(f"  ✅ 경기 데이터: {len(matches_data)}개")

# 2.2 팀 통계 로드
print("  팀 통계 로딩...")
stats_response = supabase.table('baseball_team_season_stats') \
    .select('*') \
    .in_('season', ['2024', '2025']) \
    .limit(1000) \
    .execute()

stats_data = stats_response.data
print(f"  ✅ 팀 통계: {len(stats_data)}개")

# DataFrame 변환
df_matches = pd.DataFrame(matches_data)
df_stats = pd.DataFrame(stats_data)

print(f"✅ 데이터 로드 완료: {len(df_matches)}개 경기")

# =======================
# 3. 데이터 조인 및 전처리
# =======================

print("\n🔧 데이터 조인 중...")

# 팀명 매핑 함수
def normalize_team_name(team_name):
    """팀명 정규화"""
    mapping = {
        'Oakland Athletics': 'Athletics',
        'Cleveland Indians': 'Cleveland Guardians'
    }
    return mapping.get(team_name, team_name)

# 팀명 정규화
df_matches['home_team_normalized'] = df_matches['home_team'].apply(normalize_team_name)
df_matches['away_team_normalized'] = df_matches['away_team'].apply(normalize_team_name)

# 홈팀 통계 조인
df_merged = df_matches.merge(
    df_stats,
    left_on=['home_team_normalized', 'season'],
    right_on=['team_name', 'season'],
    how='left',
    suffixes=('', '_home_stats')
)

# 원정팀 통계 조인
df_merged = df_merged.merge(
    df_stats,
    left_on=['away_team_normalized', 'season'],
    right_on=['team_name', 'season'],
    how='left',
    suffixes=('_home', '_away')
)

# 결측치 제거
df_final = df_merged.dropna(subset=[
    'win_pct_home', 'win_pct_away',
    'team_era_home', 'team_era_away',
    'team_runs_per_game_home', 'team_runs_per_game_away'
])

print(f"✅ 조인 완료: {len(df_final)}개 경기 (결측치 제거 후)")

# =======================
# 4. Feature Engineering
# =======================

print("\n🔧 Feature Engineering...")

# 4.1 Target 변수 생성
df_final['home_win'] = (df_final['home_score'] > df_final['away_score']).astype(int)
df_final['total_runs'] = df_final['home_score'] + df_final['away_score']
df_final['over_8_5'] = (df_final['total_runs'] > 8.5).astype(int)

# 4.2 계산 Features
df_final['win_pct_diff'] = df_final['win_pct_home'] - df_final['win_pct_away']
df_final['era_diff'] = df_final['team_era_away'] - df_final['team_era_home']  # 낮을수록 좋음
df_final['rpg_diff'] = df_final['team_runs_per_game_home'] - df_final['team_runs_per_game_away']

# 홈 어드밴티지
df_final['home_home_win_pct'] = df_final['home_wins_home'] / (df_final['home_wins_home'] + df_final['home_losses_home'])
df_final['away_away_win_pct'] = df_final['away_wins_away'] / (df_final['away_wins_away'] + df_final['away_losses_away'])

# 최근 10경기 승률
def parse_last_10(record):
    """최근 10경기 기록 파싱"""
    try:
        if pd.isna(record):
            return 0.5
        wins, losses = map(int, str(record).split('-'))
        return wins / (wins + losses) if (wins + losses) > 0 else 0.5
    except:
        return 0.5

df_final['home_recent_form'] = df_final['last_10_record_home'].apply(parse_last_10)
df_final['away_recent_form'] = df_final['last_10_record_away'].apply(parse_last_10)

# 결측치 처리
df_final = df_final.fillna({
    'home_home_win_pct': 0.5,
    'away_away_win_pct': 0.5,
    'home_recent_form': 0.5,
    'away_recent_form': 0.5,
})

print(f"✅ Feature Engineering 완료")

# =======================
# 5. 모델 학습
# =======================

print("\n🤖 모델 학습 시작...")

# Feature 선택
feature_columns = [
    'win_pct_home', 'win_pct_away', 'win_pct_diff',
    'team_era_home', 'team_era_away', 'era_diff',
    'team_runs_per_game_home', 'team_runs_per_game_away', 'rpg_diff',
    'home_home_win_pct', 'away_away_win_pct',
    'home_recent_form', 'away_recent_form',
]

X = df_final[feature_columns]
y_win = df_final['home_win']
y_over = df_final['over_8_5']

# Train/Test Split
X_train, X_test, y_win_train, y_win_test = train_test_split(
    X, y_win, test_size=0.2, random_state=42
)

_, _, y_over_train, y_over_test = train_test_split(
    X, y_over, test_size=0.2, random_state=42
)

print(f"   학습 데이터: {len(X_train)}개")
print(f"   검증 데이터: {len(X_test)}개")

# Model 1: 승부 예측
print("\n📊 Model 1: 승부 예측 학습 중...")

win_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

win_model.fit(X_train, y_win_train)

y_win_pred = win_model.predict(X_test)
y_win_proba = win_model.predict_proba(X_test)[:, 1]

win_accuracy = accuracy_score(y_win_test, y_win_pred)
win_auc = roc_auc_score(y_win_test, y_win_proba)

print(f"   ✅ 승부 예측 정확도: {win_accuracy:.2%}")
print(f"   ✅ AUC Score: {win_auc:.3f}")

# Model 2: 총점 예측
print("\n📊 Model 2: 총점 예측 학습 중...")

over_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

over_model.fit(X_train, y_over_train)

y_over_pred = over_model.predict(X_test)
y_over_proba = over_model.predict_proba(X_test)[:, 1]

over_accuracy = accuracy_score(y_over_test, y_over_pred)
over_auc = roc_auc_score(y_over_test, y_over_proba)

print(f"   ✅ 총점 예측 정확도: {over_accuracy:.2%}")
print(f"   ✅ AUC Score: {over_auc:.3f}")

# =======================
# 6. Feature Importance
# =======================

print("\n📈 Feature Importance (승부 예측):")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': win_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# =======================
# 7. 모델 저장
# =======================

print("\n💾 모델 저장 중...")

# 저장 디렉토리 생성
os.makedirs('models', exist_ok=True)

# 모델 저장
joblib.dump(win_model, 'models/baseball_win_model.pkl')
joblib.dump(over_model, 'models/baseball_over_model.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')

# 메타 정보 저장
metadata = {
    'trained_at': datetime.now().isoformat(),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'win_accuracy': float(win_accuracy),
    'win_auc': float(win_auc),
    'over_accuracy': float(over_accuracy),
    'over_auc': float(over_auc),
    'features': feature_columns,
    'seasons': ['2024', '2025']
}

import json
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ 모델 저장 완료!")
print(f"   - baseball_win_model.pkl")
print(f"   - baseball_over_model.pkl")
print(f"   - feature_columns.pkl")
print(f"   - metadata.json")

# =======================
# 8. 샘플 예측
# =======================

print("\n🎯 샘플 예측 테스트...")

if len(X_test) > 0:
    sample_game = X_test.iloc[0:1]
    
    win_prob = win_model.predict_proba(sample_game)[0][1]
    over_prob = over_model.predict_proba(sample_game)[0][1]
    
    print(f"   홈팀 승리 확률: {win_prob:.1%}")
    print(f"   총점 Over 8.5 확률: {over_prob:.1%}")

# =======================
# 9. 완료
# =======================

print(f"\n✅ 학습 완료!")
print(f"종료 시간: {datetime.now()}")
print(f"\n📊 최종 결과:")
print(f"   승부 예측 정확도: {win_accuracy:.2%}")
print(f"   총점 예측 정확도: {over_accuracy:.2%}")
print(f"   학습 데이터: {len(X_train)}개 경기")
print(f"\n다음 단계: Next.js API에 모델 통합")
