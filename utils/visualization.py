import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# 모델 학습 과정 심화 시각화 함수
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 모델 예측 결과 시각화 함수
def print_classification_report(y_true, y_pred, model_name):
    print(f"=== {model_name} Classification Report ===")
    print(classification_report(y_true, y_pred))
    print()

# 예측 결과 출력 함수
def print_prediction_results(test_sentences, predictions, model_name):
    print(f"=== {model_name} 예측 결과 ===")
    for i, sentence in enumerate(test_sentences):
        urgency = "긴급" if predictions[i][0] < 0.5 else "일반"
        print(f"텍스트: {sentence} => 예측된 긴급도: {urgency} (점수: {predictions[i][0]:.2f})")
    print()

# 모델 성능 비교 함수
def compare_model_performance(lstm_accuracy, bert_accuracy):
    print("=== 모델 성능 비교 ===")
    print(f"GloVe+LSTM 정확도: {lstm_accuracy:.1%}")
    print(f"BERT 정확도: {bert_accuracy:.1%}")
    
    if bert_accuracy > lstm_accuracy:
        diff = bert_accuracy - lstm_accuracy
        print(f"최고 성능 모델: BERT (+{diff:.1%})")
    else:
        diff = lstm_accuracy - bert_accuracy
        print(f"최고 성능 모델: GloVe+LSTM (+{diff:.1%})")
    print() 