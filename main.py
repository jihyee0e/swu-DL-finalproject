import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import time
import argparse
import sys
# 유틸리티 임포트
from utils.preprocessing import load_and_preprocess_data, calculate_emergency_score, split_data
from utils.visualization import plot_training_history, print_classification_report, print_prediction_results, compare_model_performance
# 모델 임포트
from models.model_lstm import train_lstm_model, predict_lstm
from models.model_bert import train_bert_model, predict_bert

def main():
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='긴급도 분류 모델 비교 프로그램')
    parser.add_argument('--train', action='store_true', help='강제로 모델을 새로 학습')
    parser.add_argument('--predict-only', action='store_true', help='저장된 모델로 예측만 실행')
    parser.add_argument('--test-sentences', nargs='+', 
                       default=[
                           "I am very scared right now", 
                           "It's a beautiful day!", 
                            "Help me! I'm in danger.", 
                            "Everything is fine.", 
                            "I need urgent assistance now!"],
                       help='테스트할 문장들')
    
    args = parser.parse_args()
    
    # print("🚀 긴급도 분류 모델 비교 프로그램 시작")
    # print("=" * 50)
    
    if args.predict_only:
        print("📊 저장된 모델로 예측만 실행합니다...")
        run_prediction_only(args.test_sentences)
        return
    
    # 데이터 로드 및 전처리
    print("📊 데이터 로드 및 전처리 중...")
    data = load_and_preprocess_data("tweet_emotions.csv")
    
    # 긴급 점수 추가
    data['emergency_score'] = data['clean_text'].apply(calculate_emergency_score)
    
    # 데이터 분할
    X_train, X_val, y_train, y_val = split_data(data)
    
    print(f"데이터 로드 완료: {len(data)}개 샘플")
    print(f"학습 데이터: {len(X_train)}개, 검증 데이터: {len(X_val)}개")
    print()
    
    # 테스트 문장 정의
    test_sentences = args.test_sentences
    
    # # GloVe+LSTM 모델 학습 및 예측
    # print("🔧 GloVe+LSTM 모델 처리 중...")
    # lstm_model, lstm_history, lstm_data_dict, lstm_time = train_lstm_model(
    #     X_train, X_val, y_train, y_val, force_train=args.train
    # )
    
    # # LSTM 예측
    # lstm_predictions = predict_lstm(lstm_model, test_sentences, lstm_data_dict)
    
    # # LSTM 검증 데이터 예측
    # lstm_val_pred = lstm_model.predict(lstm_data_dict['X_val_features'])
    # lstm_val_pred_class = (lstm_val_pred > 0.5).astype(int)
    
    # print()
    
    # BERT 모델 학습 및 예측
    print("🤖 BERT 모델 처리 중...")
    bert_model, bert_history, bert_data_dict, bert_time = train_bert_model(
        X_train['clean_text'], X_val['clean_text'], y_train, y_val, force_train=args.train
    )
    
    # BERT 예측
    bert_predictions = predict_bert(bert_model, test_sentences, bert_data_dict)
    
    # BERT 검증 데이터 예측
    bert_val_encodings = bert_data_dict['tokenizer'](
        list(X_val['clean_text']),
        max_length=50,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    bert_val_pred = bert_model.predict([bert_val_encodings['input_ids'], bert_val_encodings['attention_mask']])
    bert_val_pred_class = (bert_val_pred > 0.5).astype(int)
    
    print()
    
    # 결과 출력
    print("📈 결과 분석")
    print("=" * 50)
    
    # 예측 결과 출력
    # print_prediction_results(test_sentences, lstm_predictions, "GloVe+LSTM 모델")
    print_prediction_results(test_sentences, bert_predictions, "BERT 모델")
    
    # Classification Report 출력
    # print_classification_report(y_val, lstm_val_pred_class, "GloVe+LSTM")
    print_classification_report(y_val, bert_val_pred_class, "BERT")
    
    # 성능 비교
    # lstm_accuracy = np.mean(lstm_val_pred_class.flatten() == y_val)
    bert_accuracy = np.mean(bert_val_pred_class.flatten() == y_val)
    # compare_model_performance(lstm_accuracy, bert_accuracy)
    
    # 학습 시간 비교
    # print("=== 학습 시간 비교 ===")
    # if lstm_time > 0:
        # print(f"GloVe+LSTM 학습 시간: {lstm_time:.1f}초")
    # else:
        # print("GloVe+LSTM: 저장된 모델 사용")
    
    if bert_time > 0:
        print(f"BERT 학습 시간: {bert_time:.1f}초")
    else:
        print("BERT: 저장된 모델 사용")
    
    # if lstm_time > 0 and bert_time > 0:
    #     print(f"BERT가 LSTM보다 {bert_time/lstm_time:.1f}배 더 오래 걸림")
    # print()
    
    # 시각화 (학습 히스토리가 있을 때만)
    # if lstm_history is not None or bert_history is not None:
    #     print("📊 학습 과정 시각화")
    #     if lstm_history is not None:
    #         plot_training_history(lstm_history, "GloVe+LSTM")
    #     if bert_history is not None:
    #         plot_training_history(bert_history, "BERT")
    
    # print("✅ 모든 분석 완료!")

def run_prediction_only(test_sentences):
    #LSTM 모델 로드
    # from models.model_lstm import load_lstm_model
    # lstm_model, lstm_data_dict = load_lstm_model()
    
    # if lstm_model is None:
    #     print("❌ 저장된 LSTM 모델을 찾을 수 없습니다.")
    #     print("먼저 모델을 학습시켜주세요.")
    #     return
    
    # BERT 모델 로드
    from models.model_bert import load_bert_model
    bert_model, bert_data_dict = load_bert_model()
    
    if bert_model is None:
        print("❌ 저장된 BERT 모델을 찾을 수 없습니다.")
        print("먼저 모델을 학습시켜주세요.")
        return
    
    # 예측 실행
    # print("🔧 GloVe+LSTM 모델로 예측 중...")
    # lstm_predictions = predict_lstm(lstm_model, test_sentences, lstm_data_dict)
    
    print("🤖 BERT 모델로 예측 중...")
    bert_predictions = predict_bert(bert_model, test_sentences, bert_data_dict)
    
    # 결과 출력
    print("\n📈 예측 결과")
    print("=" * 50)
    # print_prediction_results(test_sentences, lstm_predictions, "GloVe+LSTM 모델")
    print_prediction_results(test_sentences, bert_predictions, "BERT 모델")
    
    print("✅ 예측 완료!")

if __name__ == "__main__":
    main() 