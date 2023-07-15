import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 데이터셋 로드 및 전처리 함수
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    dataset = []
    labels = []
    for line in lines:
        line = line.strip()
        text, label = line[:-2], int(line[-1])
        dataset.append(text)
        labels.append(label)
    return dataset, labels

def preprocess_text(text):
    # 소문자 변환
    text = text.lower()
    # 특수 문자 제거
    text = re.sub(r"[^\w\s]", "", text)
    return text

# 데이터셋 로드
dataset, labels = load_dataset('./data/dataset.txt')

# 텍스트 전처리
preprocessed_dataset = [preprocess_text(text) for text in dataset]

# 특징 추출
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(preprocessed_dataset)

# 모델 훈련
model = LogisticRegression()
model.fit(features, labels)

# 입력 텍스트 분류 함수
def classify_text(text):
    # 텍스트 전처리
    preprocessed_text = preprocess_text(text)
    # 특징 추출
    features = vectorizer.transform([preprocessed_text])
    # 분류 예측
    prediction = model.predict(features)[0]
    if prediction == 1:
        return "욕설입니다."
    else:
        return "욕설이 아닙니다."

# 사용자 입력 텍스트 분류
user_input = input("텍스트를 입력하세요: ")
result = classify_text(user_input)
print(result)
