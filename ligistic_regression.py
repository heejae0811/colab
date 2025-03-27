import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 0: 데이터 불러오기
df = pd.read_csv('/content/soccer_agility_training_data.csv')

# Step 1: 분석에 사용할 컬럼 정의
metrics = ['Pre_Slalom', 'Post_Slalom', 'Pre_Zigzag', 'Post_Zigzag', 'Pre_Illinois', 'Post_Illinois']

# Step 2: Group 문자 라벨을 숫자로 변환
df['Group'] = df['Group'].map({'Control': 0, 'Intervention': 1})

# Step 3: metrics 컬럼을 숫자형으로 변환
for col in metrics:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 4: NaN 제거
df = df.dropna(subset=metrics + ['Group'])

# Step 5: 입력(X)과 타겟(y) 정의
X = df[metrics]
y = df['Group']

# Step 6: 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: 모델 학습 및 예측
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Step 8: 성능 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Step 9: 결과 출력
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)