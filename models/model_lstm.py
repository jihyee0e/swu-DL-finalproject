import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import time

# GloVe 임베딩 로드 함수
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except ValueError:
                continue
    return embeddings_index

# LSTM 모델을 위한 데이터 준비
def prepare_lstm_data(X_train, X_val, y_train, y_val, glove_file='data/glove.6B.100d.txt'):
    # GloVe 임베딩 로드
    embeddings_index = load_glove_embeddings(glove_file)
    
    # Tokenizer 생성
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train['clean_text'])
    
    # 텍스트를 시퀀스로 변환
    X_train_seq = tokenizer.texts_to_sequences(X_train['clean_text'])
    X_val_seq = tokenizer.texts_to_sequences(X_val['clean_text'])
    
    # 패딩
    max_len = max(len(seq) for seq in X_train_seq)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    
    # 긴급 점수 추가
    X_train_features = np.hstack((X_train_padded, np.array(X_train['emergency_score']).reshape(-1, 1)))
    X_val_features = np.hstack((X_val_padded, np.array(X_val['emergency_score']).reshape(-1, 1)))
    
    # SMOTE 적용
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_features, y_train)
    
    # GloVe 임베딩 행렬 생성
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return {
        'X_train_resampled': X_train_resampled,
        'X_val_features': X_val_features,
        'y_train_resampled': y_train_resampled,
        'y_val': y_val,
        'embedding_matrix': embedding_matrix,
        'tokenizer': tokenizer,
        'max_len': max_len
    }

# LSTM 모델 생성 함수
def create_lstm_model(embedding_matrix, max_len, vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], 
                 input_length=max_len, trainable=False),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# LSTM 모델과 토크나이저 저장
def save_lstm_model(model, data_dict, model_path='models/lstm_model.h5', tokenizer_path='models/lstm_tokenizer.pkl'):
    # 모델 저장 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    
    # 모델 저장
    model.save(model_path)
    
    # 토크나이저와 데이터 정보 저장
    save_data = {
        'tokenizer': data_dict['tokenizer'],
        'max_len': data_dict['max_len'],
        'embedding_matrix': data_dict['embedding_matrix']
    }
    
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"LSTM 모델이 저장되었습니다: {model_path}")
    print(f"토크나이저가 저장되었습니다: {tokenizer_path}")

# 저장된 LSTM 모델과 토크나이저 로드
def load_lstm_model(model_path='models/lstm_model.h5', tokenizer_path='models/lstm_tokenizer.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None, None
    
    try:
        # 모델 로드
        model = load_model(model_path)
        
        # 토크나이저와 데이터 정보 로드
        with open(tokenizer_path, 'rb') as f:
            save_data = pickle.load(f)
        
        data_dict = {
            'tokenizer': save_data['tokenizer'],
            'max_len': save_data['max_len'],
            'embedding_matrix': save_data['embedding_matrix']
        }
        
        print(f"저장된 LSTM 모델을 로드했습니다: {model_path}")
        return model, data_dict
    
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None, None

# LSTM 모델 학습 (저장된 모델이 있으면 로드, 없으면 학습)
def train_lstm_model(X_train, X_val, y_train, y_val, glove_file='data/glove.6B.100d.txt', force_train=False):
    # 저장된 모델 확인
    if not force_train:
        model, data_dict = load_lstm_model()
        if model is not None:
            return model, None, data_dict, 0  # 학습 시간 0초
    
    print("=== GloVe+LSTM 모델 학습 시작 ===")
    start_time = time.time()
    
    # 데이터 준비
    data_dict = prepare_lstm_data(X_train, X_val, y_train, y_val, glove_file)
    
    # 모델 생성
    model = create_lstm_model(
        data_dict['embedding_matrix'], 
        data_dict['max_len'], 
        len(data_dict['tokenizer'].word_index) + 1
    )
    
    # 클래스 가중치 계산
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(data_dict['y_train_resampled']), 
        y=data_dict['y_train_resampled']
    )
    class_weights_dict = dict(zip(np.unique(data_dict['y_train_resampled']), class_weights))
    
    # 모델 학습
    history = model.fit(
        data_dict['X_train_resampled'], data_dict['y_train_resampled'],
        validation_data=(data_dict['X_val_features'], data_dict['y_val']),
        epochs=10,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
    )
    
    training_time = time.time() - start_time
    print(f"모델 학습 완료! (소요시간: {training_time:.1f}초)")
    
    # 모델 저장
    save_lstm_model(model, data_dict)
    
    return model, history, data_dict, training_time

# LSTM 모델 예측 함수
def predict_lstm(model, test_sentences, data_dict):
    # 텍스트 전처리
    from utils.preprocessing import preprocess_text
    test_sentences_processed = [preprocess_text(sentence) for sentence in test_sentences]
    
    # 텍스트를 시퀀스로 변환
    test_sequences = data_dict['tokenizer'].texts_to_sequences(test_sentences_processed)
    
    # 패딩
    test_padded = pad_sequences(test_sequences, maxlen=data_dict['max_len'], 
                               padding='post', truncating='post')
    
    # 긴급 점수 추가
    emergency_scores = np.array([0] * len(test_sentences)).reshape(-1, 1)  # 테스트용으로 0으로 설정
    test_features = np.hstack((test_padded, emergency_scores))
    
    # 예측
    predictions = model.predict(test_features)
    return predictions 