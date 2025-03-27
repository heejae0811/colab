# KNN을 위한 라이브러리
# 이진 분류(Binary Classification), 지도학습
# 운동 기록(matric)을 바탕으로 이 참가자가 어떤 그룹에 속했는지 예측하는 지도학습 분류 모델
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Feature와 Target 설정
X = df[metrics]  # 위에서 사용한 metrics 리스트 사용
y = df['Group']  # Intervention / Control

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: KNN 모델 훈련
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Step 5: 예측 및 평가
y_pred = knn.predict(X_test_scaled)

# Step 6: 결과 출력
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))