# streamlit run app.py
import streamlit as st
import process_encoding as pe
import joblib
import numpy as np
import pandas as pd
import sys
import os

    
# Streamlit 앱 설정
st.set_page_config(layout="wide", page_title="고객 이탈률 예측", page_icon="🔍")
st.title('고객 이탈률 예측')

@st.cache_resource
def load_model():
    return joblib.load('models/best_RF.pkl')

def evaluate_conditions(children_in_hh, handset_web_capable, buys_via_mail_order, non_us_travel, has_credit_card, new_cellphone_user,prizm_code):
    return {
        'ChildrenInHH_No': not children_in_hh,
        'ChildrenInHH_Yes': children_in_hh,
        'HandsetWebCapable_No': not handset_web_capable,
        'HandsetWebCapable_Yes': handset_web_capable,
        'BuysViaMailOrder_No': not buys_via_mail_order,
        'BuysViaMailOrder_Yes': buys_via_mail_order,
        'NonUSTravel_No': not non_us_travel,
        'NonUSTravel_Yes': non_us_travel,
        'HasCreditCard_No': not has_credit_card,
        'HasCreditCard_Yes': has_credit_card,
        'NewCellphoneUser_No': not new_cellphone_user,
        'NewCellphoneUser_Yes': new_cellphone_user,
        'PrizmCode_Other': prizm_code == 'Other',
        'PrizmCode_Surburban': prizm_code == 'Surburban',
        'PrizmCode_Town': prizm_code == 'Town',
    }


# 모델 및 전처리기 로드
model = load_model()

# 입력
st.sidebar.header('입력 정보')
MonthlyRevenue = st.sidebar.number_input('월별 수익', min_value=-999.0, value=0.0)
MonthlyMinutes = st.sidebar.number_input('월별 통화 분수', min_value=0.0, value=0.0)
TotalRecurringCharge = st.sidebar.number_input('총 정기 요금', min_value=-999.0, value=0.0)
DirectorAssistedCalls = st.sidebar.number_input('상담사 도움을 받은 건수', min_value=0.0, value=0.0)
OverageMinutes = st.sidebar.number_input('초과 통화 분수', min_value=0.0, value=0.0)
RoamingCalls = st.sidebar.number_input('로밍 통화 수', min_value=0.0, value=0.0)
PercChangeMinutes = st.sidebar.number_input('통화 분수의 변동률', min_value=-999.0, value=0.0)
PercChangeRevenues = st.sidebar.number_input('수익의 변동률', min_value=-999.0, value=0.0)
DroppedCalls = st.sidebar.number_input('통화 중 끊어진 통화 수', min_value=0.0, value=0.0)
BlockedCalls = st.sidebar.number_input('차단된 통화 수', min_value=0.0, value=0.0)
UnansweredCalls = st.sidebar.number_input('응답받지 못한 통화 수', min_value=0.0, value=0.0)
CustomerCareCalls = st.sidebar.number_input(' 고객 센터 통화 수', min_value=0.0, value=0.0)
ReceivedCalls = st.sidebar.number_input('받은 통화 수', min_value=0.0, value=0.0)
OutboundCalls = st.sidebar.number_input('발신 통화 수', min_value=0.0, value=0.0)
InboundCalls = st.sidebar.number_input('수신 통화 수', min_value=0.0, value=0.0)
PeakCallsInOut = st.sidebar.number_input('피크 시간대의 발신 및 수신 통화 수', min_value=0.0, value=0.0)
DroppedBlockedCalls = st.sidebar.number_input('끊어진 및 차단된 통화 수', min_value=0.0, value=0.0)
CallWaitingCalls = st.sidebar.number_input('통화 대기 중인 통화 수', min_value=0.0, value=0.0)
MonthsInService  = st.sidebar.number_input('서비스 가입 월수', min_value=0, value=0)
UniqueSubs  = st.sidebar.number_input('고유 가입자 수', min_value=0, value=0)
ActiveSubs  = st.sidebar.number_input('활성 가입자 수', min_value=0, value=0)
Handsets = st.sidebar.number_input('휴대폰 수', min_value=0, value=0)
CurrentEquipmentDays = st.sidebar.number_input('현재 장비 사용 일수', min_value=0.0, value=0.0,step=1.0)
HandsetModels = st.sidebar.number_input('휴대폰 모델 수', min_value=0, value=0, step = 1)
IncomeGroup = st.sidebar.number_input('소득 그룹', min_value=0, max_value=9, value=0, step=1)

# 범주형
ChildrenInHH = ["Yes", "No"]
service_area_options = [
    'BOS', 'PHI', 'NYC', 'PIT', 'MIA', 'ATL', 'HAR', 'NSH', 'NCR', 'NNY',
    'CHI', 'DET', 'STL', 'DAL', 'HOU', 'KCY', 'OMA', 'IND', 'INH', 'IPM', 'AWI', 'FLN', 'OHI', 'OHH',
    'LAX', 'SFR', 'SEA', 'SAN', 'PHX', 'DEN', 'SLC', 'LAU', 'NEV', 'NMC', 'NMX', 'NVU', 'HWI', 'SHE', 'SDA', 'SEW', 'SFU', 'SLU'
]
handset_web_capable_options = ["Yes", "No"]
has_credit_card_options = ["Yes", "No"]
new_cellphone_user_options = ["Yes", "No"]
credit_rating_options = ["1-Highest", "2-High", "3-Good", "4-Medium", "5-Low", "6-VeryLow", "7-Lowest"]
prizm_code_options = ["Town", "Suburban", "Rural", "Other"]
occupation_options = ["Professional", "Crafts", "Clerical", "Self", "Retired", "Student", "Homemaker", "Other"]

# 컬럼별 선택 위젯
ChildrenInHH = st.sidebar.selectbox('가구 내 자녀:', ChildrenInHH)
service_area = st.sidebar.selectbox("서비스 지역:", service_area_options)
handset_web_capable = st.sidebar.selectbox("웹 사용 가능한 휴대폰 여부:", handset_web_capable_options)
has_credit_card = st.sidebar.selectbox("신용카드 보유 여부:", has_credit_card_options)
new_cellphone_user = st.sidebar.selectbox("신규 휴대폰 사용자 여부:", new_cellphone_user_options)
credit_rating = st.sidebar.selectbox("신용 등급:", credit_rating_options)
prizm_code = st.sidebar.selectbox("프리즘 코드:", prizm_code_options)
occupation = st.sidebar.selectbox("직업:", occupation_options)
# 예측 버튼
btn = st.sidebar.button('예측')
if btn:
    input_data = pd.DataFrame({
        'MonthlyRevenue': [MonthlyRevenue],
        'MonthlyMinutes': [MonthlyMinutes],
        'TotalRecurringCharge': [TotalRecurringCharge],
        'DirectorAssistedCalls': [DirectorAssistedCalls],
        'OverageMinutes': [OverageMinutes],
        'RoamingCalls': [RoamingCalls],
        'PercChangeMinutes': [PercChangeMinutes],
        'PercChangeRevenues': [PercChangeRevenues],
        'DroppedCalls': [DroppedCalls],
        'BlockedCalls': [BlockedCalls],
        'UnansweredCalls': [UnansweredCalls],
        'CustomerCareCalls': [CustomerCareCalls],
        'ReceivedCalls': [ReceivedCalls],
        'OutboundCalls': [OutboundCalls],
        'InboundCalls': [InboundCalls],
        'PeakCallsInOut': [PeakCallsInOut],
        'DroppedBlockedCalls': [DroppedBlockedCalls],
        'CallWaitingCalls': [CallWaitingCalls],
        'MonthsInService': [MonthsInService],
        'UniqueSubs': [UniqueSubs],
        'ActiveSubs': [ActiveSubs],
        'Handsets': [Handsets],
        'CurrentEquipmentDays': [CurrentEquipmentDays],
        'ChildrenInHH': [ChildrenInHH],
        'HandsetModels': [HandsetModels],
        'ServiceArea': [service_area],
        'HandsetWebCapable': [handset_web_capable],
        'HasCreditCard': [has_credit_card],
        'NewCellphoneUser': [new_cellphone_user],
        'IncomeGroup': [IncomeGroup],
        'CreditRating': [credit_rating],
        'PrizmCode': [prizm_code],
        'Occupation': [occupation],
        'NonUSTravel_NO' : True,
        'NonUSTravel-YES' : False,
        'BuysViaMailOrder_No' : True,
        'BuysViaMailOrder_Yes' : False,
        'NotNewCellphoneUser_No' : True,          
        'NotNewCellphoneUser_Yes' : False,
        'ChildrenInHH_No': not ChildrenInHH,
        'ChildrenInHH_Yes': ChildrenInHH,
        'HandsetWebCapable_No': not handset_web_capable,
        'HandsetWebCapable_Yes': handset_web_capable,
        'HasCreditCard_No': not has_credit_card,
        'HasCreditCard_Yes': has_credit_card,
        'NewCellphoneUser_No': not new_cellphone_user,
        'NewCellphoneUser_Yes': new_cellphone_user,
        'PrizmCode_Surburban': prizm_code == 'Surburban',
        'PrizmCode_Town': prizm_code == 'Town'

    })
    preprocessor = pe.processor(input_data)
    
    # 예측 수행
    prediction = model.predict(preprocessor.to_numpy())
    prediction_proba = model.predict_proba(preprocessor)[:, 1]

    # 결과 출력
    st.header('예측 결과')
    st.subheader(f'이탈여부 예측: {"이탈" if prediction[0] == 1 else "이탈X"}')
    st.write(f'이탈 확률: {prediction_proba[0]*100:.2f}%')