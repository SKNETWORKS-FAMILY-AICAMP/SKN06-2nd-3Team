# SKN06-2nd-3Team

## 2차 프로젝트 - 통신사 고객 이탈 분석 및 예측 📈

  ### 팀명 
  **가나디즈**ᘳ´• ᴥ •`ᘰ 
  ### 팀원 👥
 
 ![스크린샷 2024-11-14 185809](https://github.com/user-attachments/assets/1890f376-14ac-4e97-a455-78d45cb71a38)
 ![스크린샷 2024-11-14 185834](https://github.com/user-attachments/assets/c02a6911-c1fd-4940-bd6d-f3a37de77a90)
 ![스크린샷 2024-11-14 185730](https://github.com/user-attachments/assets/d0d11852-4f3c-4462-8e9d-7a2f71419db8)
 ![스크린샷 2024-11-14 191022](https://github.com/user-attachments/assets/8d273c62-cddc-4c7c-829c-9e1478881266)

  | 이세화૮ ･ ﻌ･ა   | &nbsp; &nbsp; &nbsp;   &nbsp;     | 김동훈◖⚆ᴥ⚆◗ |&nbsp; &nbsp; &nbsp;  &nbsp;       | 안형진 ૮ ºﻌºა |&nbsp; &nbsp; &nbsp;      | 전수연υ´• ﻌ •`υ         |
 

______________________________________________________________________________________________________

## 프로젝트 개요

### 소개
통신사 개별 고객 정보와 이탈 여부 데이터를 통해 이탈 가능성을 유추하는 모델을 만들어 평가하는 프로젝트.<br/>

### 필요성
- **비용 효율성**: 새로운 고객을 확보하는 비용이 기존 고객을 유지하는 비용보다 약 5배 더 많이 듭니다.<br/>
  따라서 기존 고객 유지는 마케팅 비용을 크게 절감할 수 있습니다.<br/>
- **시장 포화**: 이동통신 시장이 성숙기에 접어들면서 신규 고객 확보가 어려워졌습니다. 따라서 기존 고객 유지가 더욱 중요해졌습니다.<br/>
- **고객 데이터 활용**: 장기 고객의 데이터를 활용하여 더 나은 서비스와 맞춤형 마케팅을 제공할 수 있습니다.<br/>
따라서 데이터 분석을 통해 고객의 이탈을 줄이는 방향으로 서비스를 개선해나가야 할 것입니다. 

### 목표
통신사 고객 이탈 예측을 통한 통신 시장에 대한 고찰 및 개선
______________________________________________________________________________________________________


## 01. EDA💻

### 1-1. Target data 분포 
<br>
<br>
<img width="359" alt="churn_distribution" src="https://github.com/user-attachments/assets/555fd403-907c-4eb6-a43c-bde136c4c56d">
<br>
<br>

### 1-2. Feature 분포
<br>
<br>
<img width="936" alt="boxplot_1" src="https://github.com/user-attachments/assets/699c2f6e-3bc6-47c8-9ba2-a63d2389a861">
<img width="939" alt="boxplot_2" src="https://github.com/user-attachments/assets/089d548b-a595-4142-8d56-2c9e691db059">
<img width="938" alt="boxplot_3" src="https://github.com/user-attachments/assets/f3eabacc-9392-4cc9-87bd-323d6ccaaaa3">
<br>
<br>

### 1-3. Correlation Matrix (수치형 상관관계)
<br>
<br>
<img width="524" alt="heatmap" src="https://github.com/user-attachments/assets/568e41f7-2d78-4d14-bfee-8b995012cac2">
<br>
<br>




## 02. 데이터 전처리💻

### 2-1. Customer ID 삭제
<br>
- ID 삭제
<br>

### 2-2. feature 선택
<br>

**불균형** : 하나의 값이 95% 이상
<br>
이런 column은 모델 효율성 저하, Overfitting 등의 문제를 야기하므로 해당 열을 삭제함.
<br>
<br>
<img width="774" alt="drop_95" src="https://github.com/user-attachments/assets/478bdb0b-dc55-41e9-b04c-347ca17effe2">
<br>
<br>

**도메인 지식**
<br>
도메인 지식으로 해당 열은 삭제함.
<br>
<br>
'OffPeakCallsInOut': 비피크 시간대의 통화 수<br>
'HandsetRefurbished': 중고 휴대폰 여부<br>
'TruckOwner': 트럭 소유 여부<br>
'RVOwner': RV 소유 여부<br>
'RespondsToMailOffers': 우편 제안에 대한 응답 여부<br>
'OwnsComputer': 컴퓨터 소유 여부<br>
'ThreewayCalls': 3자 통화 사용 횟수<br>
<br>
<br>

### 2-3. 결측치 처리
<br>

**null 값**
<br>
null 값 개수 확인 결과 전체 데이터 개수의 적은 비율을 차지하고 있어 null값이 포함된 행은 삭제함
<br>
<br>
<img width="172" alt="null_drop" src="https://github.com/user-attachments/assets/9ea1dcdc-1dde-4519-8745-f577eb7b8f77">
<br>
<br>

**unknown 값**
<br>
unknown 값이 있는 열 모두 비율이 높아 해당 열은 삭제함
<br>
<br>
<img width="138" alt="unknown_drop" src="https://github.com/user-attachments/assets/1d4186e2-e780-4f53-8e74-b2ce74ebde08">
<br>
<br>

**0 값**
<br>
0값이 있는 열은 0값이 얼마나 있는지 평균,중앙값과 함께 히스토그램으로 확인함<br>
평균,중앙값,다른 값의 크기로 보아 숫자 0이 의미있는 값이 아닌 결측치라고 판단되는 'AgeHH1','AgeHH2' 열 삭제함
<br>
<br>
('AgeHH1','AgeHH2')
<br>
<img width="817" alt="zero_drop" src="https://github.com/user-attachments/assets/261e4c9b-9cb0-4a73-9065-355b3d3cd4f2">
<br>
(의미 있는 0값 예시)<br>
<img width="812" alt="zero_nondrop" src="https://github.com/user-attachments/assets/7791e057-1009-450c-99c2-c76276192fe8">
<br>
<br>

### 2-4. 이상치 처리
<br>

**데이터 변환**
<br>
이상치 처리에 앞서 필요한 값들 데이터 변환
<br>

(1) 'Churn' 컬럼 변환
<br>
"Yes": 1, "No": 0<br>

(2) 'ServiceArea' 컬럼 변환
<br>
'ServiceArea' column의 고유값은 748개임. '지역명+숫자3자리' 형태. 지역명중에서도 앞의 3자리가 도시 이름이어서 도시 이름으로 바꾼 후, 각각의 도시를 동부,중부,서부 3가지로 나눠 고유값 3개로 줄임<br>
'East': ['BOS', 'PHI', 'NYC', 'PIT', 'MIA', 'ATL', 'HAR', 'NSH', 'NCR', 'NNY'],<br>
'Center': ['CHI', 'DET', 'STL', 'DAL', 'HOU', 'KCY', 'OMA', 'IND', 'INH', 'IPM', 'AWI', 'FLN', 'OHI', 'OHH'],<br>
'West': ['LAX', 'SFR', 'SEA', 'SAN', 'PHX', 'DEN', 'SLC', 'LAU', 'NEV', 'NMC', 'NMX', 'NVU', 'HWI', 'SHE', 'SDA', 'SEW', 'SFU', 'SLU']<br>

(3) 'CreditRating' 컬럼 변환
<br>
문자열+숫자 형태의 값 정수로 변환
"1-Highest": 1, "2-High": 2, "3-Good": 3,"4-Medium": 4, "5-Low": 5, "6-VeryLow": 6,"7-Lowest": 7
<br>
<br>

**이상치 처리 (수치형 데이터)**
<br>
이상치 대체값을 구분하는 함수 생성. column 리스트를 함수에 넣어 각 열 별로 어떤 대체값을 사용할 것이지 결정
<br>
<br>
<img width="356" alt="grouping_column" src="https://github.com/user-attachments/assets/86e93dee-bffc-41de-a4a1-dad240d752b9">
<br>
skew > 1 or skew < -1 : 중앙값으로 대체<br>
상한값(Upper Bound)과 하한값(Lower Bound) 사이의 값이 95% 이상인 경우 : IQR 대체<br>
Q1~Q3 범위에 값이 75% 이상 포함되는 경우:  Q1~Q3 범위 대체<br>
최댓값(max)이 상한값(Upper Bound)을 초과하는 경우 : 상한값 대체<br>
최솟값(min)이 하한값(Lower Bound) 미만인 경우 : 하한값 대체<br>

**이상치 처리 (범주형 데이터)**
<br>
5% 미만의 값은 이상치로 판단하고 최반값으로 대체
<br>
<br>
<img width="215" alt="replacement_mode" src="https://github.com/user-attachments/assets/a27f1f61-5f9b-48cd-bfbe-19f7e2f863ad">
<br>
<br>

### 2-5. 원핫인코딩
<br>
과도한 차원 증가를 막고자 범주가 10개 미만인 열만 원핫인코딩 진행
<br>
<br>
______________________________________________________________________________________________________

## 03. 모델 💻

### 3-1. 머신러닝 모델
**DecisionTreeClassifier 모델**:<br>
특성을 기반으로 데이터를 재귀적으로 분할하여 트리 구조의 분류 모델을 만듭니다. 해석이 쉽지만 과적합 위험이 있습니다.<br>
![DT](https://github.com/user-attachments/assets/a8b4800a-8cd8-47ec-9e5d-2b43d7abd8f2)
<br>
<br>
**Random Forest 모델**:<br>
여러 개의 결정 트리를 독립적으로 학습시키고 그 결과를 종합하는 앙상블 방법입니다.<br>
클래스 가중치(class_weight='balanced')를 설정하여 데이터 불균형 문제를 어느 정도 해결했습니다.<br>
이 모델에서 F1 점수가 0.46으로 제일 높았습니다.<br>
![RF](https://github.com/user-attachments/assets/2556a26a-ca96-4185-a7a1-dccb26006644)
<br>
<br>


**GradientBoostingClassifier 모델**:<br>
여러 개의 약한 학습기(주로 결정 트리)를 순차적으로 학습시켜 강한 분류기를 만드는 앙상블 방법입니다. 각 단계에서 이전 모델의 오차를 보완하는 방식으로 학습합니다.<br>
![GB](https://github.com/user-attachments/assets/0acf6910-bd39-4417-9a30-cb6b6301d5e6)<br><br>


**LGBMClassifier 모델**:<br>
그래디언트 부스팅의 또 다른 최적화 구현입니다. 리프 중심 트리 성장 전략을 사용하여 더 빠른 학습과 더 나은 성능을 제공합니다.<br>
![LBM](https://github.com/user-attachments/assets/a0420ea4-b13e-4a45-a02b-cd81d7a50f07)<br><br>


**KNeighborsClassifier모델**:<br>
새로운 데이터 포인트에 대해 가장 가까운 k개의 이웃 데이터 포인트들의 클래스를 참조하여 분류를 수행하는 비모수적 방법입니다.<br>
![KNN](https://github.com/user-attachments/assets/fbb15a74-e447-4d7b-97c0-767599665349)<br><br>


**LogisticRegression모델**:<br>
선형 분류 모델로, 입력 특성의 선형 결합을 사용하여 클래스 확률을 예측합니다.<br>
이진 분류와 다중 분류에 모두 사용될 수 있으며, 해석이 쉽고 계산 비용이 적습니다.<br>
역시나 클래스 가중치(class_weight='balanced')를 설정하니 f1점수가 두번째로 높게 나왔습니다.<br>
![LR](https://github.com/user-attachments/assets/096e3ef2-0d06-4ebe-9242-0680c325c8a0)<br><br>


#### RandomizedSearchCV를 이용한 성능개선:
 F1점수가 가장 높게 나온 Random Forest 모델을 선정하여 하이퍼 파라미터 튜닝을 진행하였습니다.<br>
 결과는 f1점수가 0.47로 아주 미약한 상승(0.1)하였습니다.<br>
 ![스크린샷 2024-11-16 174133](https://github.com/user-attachments/assets/7d42c706-ac6c-451e-862f-ff6c512eecb2)<br><br>


#### 최종 모델의 confusion_matrix:
![스크린샷 2024-11-16 182904](https://github.com/user-attachments/assets/3b1b1a7d-0b47-4035-906b-010dd286cf0a)


#### 분석 지표:
- Precision
- Recall
- F1 Score
- ROC-AUC Score



### 3-2. 딥러닝 모델


**변수 처리**

    # 타겟 분리
    X = data.drop(columns=['Churn'])  
    y = data['Churn']
    
    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 오버샘플링(불균형 해결)
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # 텐서로 변환
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


**모델**

    # 모델 정의
    class SimpleNN(nn.Module):
        def __init__(self, input_size):
            super(SimpleNN, self).__init__()
            self.layer1 = nn.Linear(input_size, 512)
            self.layer2 = nn.Linear(512, 256)
            self.layer3 = nn.Linear(256, 128)
            self.layer4 = nn.Linear(128, 64)
            self.layer5 = nn.Linear(64, 32)
            self.output = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer3(x))
            x = torch.relu(self.layer4(x))
            x = torch.relu(self.layer5(x))
            x = self.output(x)
            return x

    # 하이퍼파라미터 그리드 
    learning_rates = [0.001, 0.0001]
    epochs_list = [20, 100, 500]
    thresholds = [0.5, 0.7]
    grid = list(product(learning_rates, epochs_list, thresholds))


<br>
<br>
모델 시간이 오래 걸려 lr=0.001, epoch 무한 조기종료, threshold=0.7 설정한 모델로 결과값 출력

**결과**
<br>
<br>
<img width="572" alt="dl_result" src="https://github.com/user-attachments/assets/ce974445-5460-4b7e-890a-98f862335cba">
<br>
<br>
recall이 precision보다 더 중요하다고 판단되기는 하나 recall값이 1.0으로 지나치게 높게 나왔고 precision이 낮게 나옴
## 04. 결과 💻

______________________________________________________________________________________________________



## 한 줄 회고 📝
 - 김동훈 :쉬고 싶어요..
 - 안형진
 - 이세화
 - 전수연: 이번 단위 기간동안 배운 전반적인 부분을 정리하는 시간이었다.
  
