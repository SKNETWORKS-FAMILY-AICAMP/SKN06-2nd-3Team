import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def dropcol(df):
    columns_to_drop = []
    df = df.drop(columns=['CustomerID', 'Homeownership','HandsetPrice','MaritalStatus', 'OffPeakCallsInOut','HandsetRefurbished','TruckOwner',
                          'RVOwner','RespondsToMailOffers','OwnsComputer','ThreewayCalls','AgeHH1','AgeHH2',])
    for column in df.columns:
        value_counts = df[column].value_counts(normalize=True)
        max_value_percentage = value_counts.max()
        if max_value_percentage >= 0.95:
            columns_to_drop.append((column, max_value_percentage))
    df = df.drop(columns=[col[0] for col in columns_to_drop])
    df.dropna(axis=0, inplace=True)
    return df

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
        
def ifmode(df, catecol):
    threshold = 0.05 
    col_replace = []
    for col in catecol:
        value_counts = df[col].value_counts(normalize=True)
        min_percentage = value_counts.min()
        if min_percentage < threshold:
            col_replace.append(col)
            print(col_replace)
        
        for column in col_replace:
            mode_value = df[column].mode()[0] 
            value_counts = df[column].value_counts(normalize=True)
            rare_values = value_counts[value_counts < 0.05].index
            df[column] = df[column].replace(rare_values, mode_value)
    return df
    
# 원핫인코딩
def ohot(df, column):
    threshold = 10  
    ohecol = []
    for col in column:
        unique_count = df[col].nunique()
        if unique_count <= threshold:
            ohecol.append(col)
    df = pd.get_dummies(df, columns=ohecol)
    # ohe = OneHotEncoder()
    # ohe_array = ohe.fit_transform(df[ohecol]).toarray()
    # ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(ohecol), index=df.index)
    # df = pd.concat([df.drop(columns=ohecol), ohe_df], axis=1)
    
    return df

# 이상치처리_IQR
def outlier_iqr(df, whis=2.0):
    median_group = []
    iqr_group = []
    q1_q3_group = []
    upper_bound_group = []
    lower_bound_group = []

    for i in df.columns:
        X = df[i]
        q1 = np.nanquantile(X, q=0.25)
        q3 = np.nanquantile(X, q=0.75)
        IQR = q3 - q1
        lower_bound = q1 - whis * IQR
        upper_bound = q3 + whis * IQR
        
        if X.skew() > 1 or X.skew() < -1:
            median_group.append(i)
        elif X.between(lower_bound, upper_bound).mean() > 0.95:
            iqr_group.append(i)
        elif X.between(q1, q3).mean() > 0.75:
            q1_q3_group.append(i)
        elif X.max() > upper_bound:
            upper_bound_group.append(i)
        elif X.min() < lower_bound:
            lower_bound_group.append(i)

    for column in median_group:
        median = df[column].median()
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = median


    for column in iqr_group:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        IQR = q3 - q1
        lower_bound = q1 - 2.0 * IQR
        upper_bound = q3 + 2.0 * IQR
        df.loc[df[column] < lower_bound, column] = q1
        df.loc[df[column] > upper_bound, column] = q3

    for column in upper_bound_group:
        q3 = df[column].quantile(0.75)
        IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
        upper_bound = q3 + 1.5 * IQR
        df.loc[df[column] > upper_bound, column] = q3



    for column in lower_bound_group:
        q1 = df[column].quantile(0.25)
        IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
        lower_bound = q1 - 1.5 * IQR
        df.loc[df[column] < lower_bound, column] = q1

    return df
    

def load_df(return_X_y=False, validset=True, test_size = 0.2, random_state=42):
    traindf = pd.read_csv("./dataset/cell2celltrain.csv")

    # 결측치 제거
    traindf = dropcol(traindf)
    
    # 정수 변환
    convertint(traindf)
    
    # 이상치 처리
    num_col = traindf.select_dtypes(include=['int64', 'float64']).columns
    cate_col = traindf.select_dtypes(include=['object', 'category']).columns
    cate_col = [i for i in cate_col if i not in "Churn"]
    traindf[num_col] = outlier_iqr(traindf[num_col])
    traindf[cate_col] = ifmode(traindf[cate_col], cate_col)
    
    # 인코딩
    traindf = ohot(traindf, cate_col)
    
    if return_X_y:
        y = traindf.Churn
        X = traindf.drop(columns="Churn")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if validset:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test
        return X_train, X_test, y_train, y_test
    return traindf

def processor(inputdata):
    num_col = inputdata.select_dtypes(include=[np.number]).columns
    cate_col = inputdata.select_dtypes(include=['object', 'category']).columns
    inputdata[num_col] = outlier_iqr(inputdata[num_col])
    inputdata[cate_col] = ifmode(inputdata[cate_col], cate_col)
    inputdata = ohot(inputdata, cate_col)
    return inputdata

if __name__ == "__main__":
    load_df().to_csv('./chk.csv', index=False)
    # print(load_df().columns)