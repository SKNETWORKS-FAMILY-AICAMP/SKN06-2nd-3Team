import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def convertint(X):
    X.Churn = X.Churn.replace("Yes", 1)
    X.Churn = X.Churn.replace("No", 0)
    
    region_dict = {
    'East': ['BOS', 'PHI', 'NYC', 'PIT', 'MIA', 'ATL', 'HAR', 'NSH', 'NCR', 'NNY'],
    'Center': ['CHI', 'DET', 'STL', 'DAL', 'HOU', 'KCY', 'OMA', 'IND', 'INH', 'IPM', 'AWI', 'FLN', 'OHI', 'OHH'],
    'West': ['LAX', 'SFR', 'SEA', 'SAN', 'PHX', 'DEN', 'SLC', 'LAU', 'NEV', 'NMC', 'NMX', 'NVU', 'HWI', 'SHE', 'SDA', 'SEW', 'SFU', 'SLU']
    }

    def map_region(service_area):
        txt = service_area[:3]
        for region, cities in region_dict.items():
            if txt in [city[:3] for city in cities]:
                return region
            
    X.ServiceArea = X.ServiceArea.apply(map_region)
            
    try:
        X.CreditRating = X.CreditRating.replace("1-Highest", 1)
        X.CreditRating = X.CreditRating.replace("2-High", 2)
        X.CreditRating = X.CreditRating.replace("3-Good", 3)
        X.CreditRating = X.CreditRating.replace("4-Medium", 4)
        X.CreditRating = X.CreditRating.replace("5-Low", 5)
        X.CreditRating = X.CreditRating.replace("6-VeryLow", 6)
        X.CreditRating = X.CreditRating.replace("7-Lowest", 7)
    except:
        print("변환 안됨")
    
    
# 원핫인코딩
def ohot(df, col):
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df[col]).toarray()
    ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(col), index=df.index)
    df = pd.concat([df.drop(columns=col), ohe_df], axis=1)
    
    return df

# 이상치처리_IQR
def outlier_iqr(df, whis=2.0):
    q1q3 = ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge', 'DirectorAssistedCalls', 'OverageMinutes', 'PercChangeMinutes', 'PercChangeRevenues', 'Handsets', 'HandsetModels']
    median_list = ['RoamingCalls']
    max_list = ['DroppedCalls', 'BlockedCalls', 'UnansweredCalls', 'CustomerCareCalls', 'ReceivedCalls', 'OutboundCalls', 'InboundCalls', 'PeakCallsInOut', 'OffPeakCallsInOut', 'DroppedBlockedCalls', 'CallForwardingCalls', 'CallWaitingCalls', 'UniqueSubs', 'ActiveSubs']
    
    for i in df.columns:
        X = df[i]
        q1 = np.nanquantile(X, q=0.25)
        q3 = np.nanquantile(X, q=0.75)
        IQR = q3 - q1
        lower_bound = q1 - IQR * whis
        upper_bound = q3 + IQR * whis
        
        if i in median_list:
            df.loc[df[i] < lower_bound, i] = X.median()
            df.loc[df[i] > upper_bound, i] = X.median()
            
        if i in q1q3:
            df.loc[df[i] < lower_bound, i] = q1
            df.loc[df[i] > upper_bound, i] = q3
            
        if i in max_list:
            df.loc[df[i] > upper_bound, i] = q3
        
    return df

def load_df(return_X_y=False, validset=True, test_size = 0.2, random_state=42):
    traindf = pd.read_csv("./dataset/cell2celltrain.csv")

    # 결측치 제거
    drop_col = ['CustomerID', 'ThreewayCalls', 'CallForwardingCalls', 'AgeHH1', 'AgeHH2', 'TruckOwner', 'RVOwner', 'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings', 'NonUSTravel', 'OwnsComputer', 'RetentionCalls', 'RetentionOffersAccepted', 'ReferralsMadeBySubscriber', 'OwnsMotorcycle', 'MadeCallToRetentionTeam', 'HandsetPrice', 'Homeownership', 'MaritalStatus']
    traindf = traindf.drop(columns=drop_col)
    traindf.dropna(axis=0, inplace=True)
    
    # 이상치 처리
    num_col = traindf.select_dtypes(include=[np.number]).columns
    traindf[num_col] = outlier_iqr(traindf[num_col])
    
    # 정수 변환 및 인코딩
    convertint(traindf)
    ohot_list = ['ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'HasCreditCard', 'NewCellphoneUser', 'NotNewCellphoneUser', 'PrizmCode', 'Occupation', 'ServiceArea', 'BuysViaMailOrder']
    traindf = ohot(traindf, ohot_list)
    
    if return_X_y:
        y = traindf.Churn
        X = traindf.drop(columns="Churn")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if validset:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test
        return X_train, X_test, y_train, y_test
    return traindf

if __name__ == "__main__":
    print(load_df())