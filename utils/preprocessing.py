import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

def preprocess_text(text):
    """
    텍스트 전처리 함수
    - URL 제거
    - 사용자 언급(@)과 해시태그(#) 제거
    - 특수 문자 및 숫자 제거
    - 소문자로 변환
    """
    # URL 제거
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 사용자 언급(@)과 해시태그(#) 제거
    text = re.sub(r'\@\w+|\#', '', text)
    # 특수 문자 및 숫자 제거
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # 소문자로 변환
    return text.lower().strip()

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(file_path="data/tweet_emotions.csv"):
    data = pd.read_csv(file_path)
    data['clean_text'] = data['content'].apply(preprocess_text)
    
    # 긍정적 감정을 정의
    positive_sentiments = ['enthusiasm', 'love', 'happiness', 'fun', 'relief']
    
    # 'sentiment' 값을 이진 값으로 변환
    data['feel'] = data['sentiment'].apply(lambda x: 1 if x in positive_sentiments else 0)
    
    return data

# 긴급 점수 계산 함수
def calculate_emergency_score(text):
    emergency_keywords = ['help', 'danger', 'urgent', 'scared', 'assistance']
    return sum(word in text for word in emergency_keywords)

def split_data(data, test_size=0.2, random_state=42):
    X = data[['clean_text', 'emergency_score']]
    y = data['feel']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) 