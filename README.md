### 4-2 딥러닝및응용 기말프로젝트

### 텍스트 기반의 감정 분석과 긴급도 탐지 모델링 <br>
: 긴급 상황을 텍스트 데이터를 통해 예측 & 사전 예방적인 조치를 위한 기초를 다지는 시스템 개발
- 데이터셋: kaggle - Emotion Detection from Text<br>
![image](https://github.com/user-attachments/assets/6b4a62e8-a3ed-4df2-ae32-d8412b05bfc5)

- **데이터 전처리**
  - 텍스트 정제: URL, 해시태그, 사용자언급(@) 제거<br>
    ![image](https://github.com/user-attachments/assets/3211a2c8-242f-47a8-a538-86d6beb0a042) <br>
    ![image](https://github.com/user-attachments/assets/62332219-0942-4f1b-bc47-67dac0987fdc) <br>
  - 특수문자 및 숫자 제거: 불필요한 문자 제거<br>
    ![image](https://github.com/user-attachments/assets/1b7de5ef-0287-4d29-a48e-59b4114e5de3)<br>
  - 소문자 변환: 텍스트를 소문자로 변환하여 일관성 유지<br>
    ![image](https://github.com/user-attachments/assets/fe09c845-5e62-440a-8082-8b95b47d7f45)<br>
  - 토큰화 및 패딩: 모델 학습을 위한 토큰화 및 시퀀스 길이 맞추기<br>
    ![image](https://github.com/user-attachments/assets/d73d163d-e350-4cd6-9971-043bcb606215)<br>

- **모델**
  (1) Glove + LSTM
    - GloVe 임베딩: 사전 훈련된 GloVe 임베딩은 단어 간의 의미적 관계를 잘 포착 → 텍스트의 의미를 효과적으로 표현
    - LSTM: 순차적인 데이터에서 좋은 성능을 발휘 → 이전 단어들로부터 중요한 정보를 기억하고, 감정 변화 패턴을 잘 학습<br>
  => 결과)<br>
    정확도 70%, recall: 0.73, f1-score: 0.77<br>
    ![image](https://github.com/user-attachments/assets/a9480d37-6912-4cbd-9690-d1810e6d5757)<br>
    샘플 데이터 예측<br>
    ![image](https://github.com/user-attachments/assets/2256fa37-a522-4de4-aff9-5ee92d95092d)<br>
    -> 긴급 상황으로 분류해야 할 텍스트가 일반 상황으로 예측되거나, 일반적인 텍스트가 긴급 상황으로 잘못 예측되는 경우가 전체적으로 발생<br>
       이는 모델이 문맥의 미세한 차이를 제대로 반영하지 못한 결과로, 긴급도를 정확히 예측하는 데 있어 개선이 필요함을 시사
    
  (2) BERT
    - 단어의 의미를 문맥에 따라 파악할 수 있어, 감정 분석처럼 문장의 뉘앙스를 이해하는 작업에 매우 유리
    - 사전 훈련된 모델을 fine-tuning하여 성능을 더욱 최적화할 수 있기 때문에, 특정 도메인이나 데이터셋에 맞게 모델을 세부 조정 가능<br>
  => 결과)<br>
    정확도 75%, recall: 0.89, f1-score: 0.83<br>
    ![image](https://github.com/user-attachments/assets/26959f24-faea-4651-98d3-1d52e5e3b64f)<br>
    샘플 데이터 예측<br>
    ![image](https://github.com/user-attachments/assets/9853380c-8418-4431-93f7-6647609bba35)<br>
    -> 일반적인 문장과 긴급 문장을 잘 구분하는 경향이 보임.<br>
      Everything is fine과 같이 일반 문장이 긴급 상황으로 잘못 예측되는 오류가 발생하였지만, 이를 제외하면 전반적으로 긴급과 일반을 정확히 구분<br>
      예측 결과는 모델이 100% 완벽하지 않지만, 긴급 상황 탐지에서 중요한 차이를 잘 구별하고 있다는 점에서 긍정적인 신호<br>
    
- 결론 및 한계, 향후 계획
  - 결론: 텍스트 기반 감정 분석 모델을 활용한 긴급 상황 예측 가능 → 사전 예방적 조치와 대응 전략 수립을 위한 중요한 기초 데이터를 제공 가능성 o
  - 한계
    - 간접적인 표현이나 비유적 언어를 제대로 파악하지 못함
    - 텍스트 데이터만 사용한 점에서 멀티모달 데이터를 결합한 예측 성능 향상에는 한계
  - 향후 계획
    - 데이터 증강 -> 다양한 감정과 긴급 상황을 다룬 데이터셋을 추가하여 모델의 범용성 증대
    - 멀티모달 데이터 결합 -> 텍스트 외에도 이미지, 음성 등 다양한 데이터를 결합하여 더 정교한 예측 가능 기대

 
  
