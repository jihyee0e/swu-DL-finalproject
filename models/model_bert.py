import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os
import time

# BERT 레이어 클래스
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_model_name="bert-base-uncased", **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(bert_model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]  # [CLS] 토큰 출력

def tokenize_data(texts, tokenizer, max_len=50):  # BERT 토크나이저를 이용한 텍스트 전처리
    return tokenizer(
        list(texts),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )

def prepare_bert_data(X_train, X_val, y_train, y_val):  # BERT 모델을 위한 데이터 준비
    # Tokenizer 설정
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 데이터 토크나이징
    train_encodings = tokenize_data(X_train, tokenizer)
    val_encodings = tokenize_data(X_val, tokenizer)
    
    # 입력 데이터 준비
    train_input_ids = train_encodings['input_ids']
    train_attention_mask = train_encodings['attention_mask']
    val_input_ids = val_encodings['input_ids']
    val_attention_mask = val_encodings['attention_mask']
    
    return {
        'train_input_ids': train_input_ids,
        'train_attention_mask': train_attention_mask,
        'val_input_ids': val_input_ids,
        'val_attention_mask': val_attention_mask,
        'tokenizer': tokenizer
    }

def create_bert_model():  # BERT 모델 생성
    # 입력 정의
    input_ids = tf.keras.Input(shape=(50,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(50,), dtype=tf.int32, name="attention_mask")
    
    # BERT 레이어 추가
    bert_output = BertLayer()([input_ids, attention_mask])
    
    # 이후 레이어
    x = tf.keras.layers.Dropout(0.3)(bert_output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # 모델 생성
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# BERT 모델과 토크나이저 저장
def save_bert_model(model, data_dict, model_path='models/bert_model.h5', tokenizer_path='models/bert_tokenizer.pkl'):
    # 모델 저장 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    
    # 모델 저장
    model.save(model_path)
    
    # 토크나이저 저장
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(data_dict['tokenizer'], f)
    
    print(f"BERT 모델이 저장되었습니다: {model_path}")
    print(f"토크나이저가 저장되었습니다: {tokenizer_path}")

# 저장된 BERT 모델과 토크나이저 로드
def load_bert_model(model_path='models/bert_model.h5', tokenizer_path='models/bert_tokenizer.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None, None
    
    try:
        # 모델 로드
        model = tf.keras.models.load_model(model_path, custom_objects={'BertLayer': BertLayer})
        
        # 토크나이저 로드
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        data_dict = {'tokenizer': tokenizer}
        
        print(f"저장된 BERT 모델을 로드했습니다: {model_path}")
        return model, data_dict
    
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None, None

# BERT 모델 학습 (저장된 모델이 있으면 로드, 없으면 학습)
def train_bert_model(X_train, X_val, y_train, y_val, force_train=False):
    # 저장된 모델 확인
    if not force_train:
        model, data_dict = load_bert_model()
        if model is not None:
            return model, None, data_dict, 0  # 학습 시간 0초
    
    print("=== BERT 모델 학습 시작 ===")
    start_time = time.time()
    
    # 데이터 준비
    data_dict = prepare_bert_data(X_train, X_val, y_train, y_val)
    
    # 모델 생성
    model = create_bert_model()
    
    # 콜백 설정
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2
    )
    
    # 모델 학습
    history = model.fit(
        [data_dict['train_input_ids'], data_dict['train_attention_mask']], y_train,
        validation_data=([data_dict['val_input_ids'], data_dict['val_attention_mask']], y_val),
        epochs=10,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr]
    )
    
    training_time = time.time() - start_time
    print(f"모델 학습 완료! (소요시간: {training_time:.1f}초)")
    
    # 모델 저장
    save_bert_model(model, data_dict)
    
    return model, history, data_dict, training_time

# BERT 모델 예측 함수
def predict_bert(model, test_sentences, data_dict):
    # 텍스트 전처리
    from utils.preprocessing import preprocess_text
    test_sentences_processed = [preprocess_text(sentence) for sentence in test_sentences]
    
    # BERT 입력 형식으로 변환
    test_encodings = tokenize_data(test_sentences_processed, data_dict['tokenizer'])
    
    # 예측
    predictions = model.predict([test_encodings['input_ids'], test_encodings['attention_mask']])
    return predictions 