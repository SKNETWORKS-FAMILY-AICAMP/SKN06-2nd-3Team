import pandas as pd

def load_dataset():
    # load dataset
    data = pd.read_csv('dataset/train_result.csv')
    # 컬럼명 소문자로 변경
    data.columns = data.columns.str.lower()
    # target 컬럼을 y로, 나머지를 X로
    X = data.drop(columns='churn')
    y = data['churn']
    
    return X, y
    
