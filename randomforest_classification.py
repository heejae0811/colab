import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 0: 데이터 불러오기
df = pd.read_csv('/content/soccer_agility_training_data.csv')

# Step 1: 사용할 feature 컬럼 정의 (자동 또는 수동)
metrics = ['Pre_Slalom', 'Post_Slalom', 'Pre_Zigzag', 'Post_Zigzag', 'Pre_Illinois', 'Post_Illinois']

# Step 2: 타겟 라벨 숫자 변환
df['Group'] = df['Group'].map({'Control': 0, 'Intervention': 1})

# Step 3: 피처 숫자형으로 변환
for col in metrics:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 4: 결측값 제거
df = df.dropna(subset=metrics + ['Group'])

# Step 5: X, y 정의
X = df[metrics]
y = df['Group']

# Step 6: 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# (선택) Step 7: 스케일링 (랜덤 포레스트는 꼭 필요하지 않지만 비교 위해 적용해도 됨)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Step 8: 랜덤 포레스트 분류기 모델 생성 및 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 9: 예측
y_pred = rf_model.predict(X_test)

# Step 10: 성능 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Step 11: 결과 출력
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
