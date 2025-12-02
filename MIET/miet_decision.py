import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# target_dam = '궁내' # 대곡 , 궁내
column=['time','서울시(대곡교)','성남시(성남북초교)','광주시(남한산초교)','성남시(대장동)','성남시(구미초교)','성남시(한국학중앙연구원)','성남시(궁내교)_WL','성남시(궁내교)_Q','서울시(대곡교)_WL','서울시(대곡교)_Q','대곡교_Ti','궁내교_Ti']
col_rf = ['서울시(대곡교)','성남시(성남북초교)','광주시(남한산초교)','성남시(대장동)','성남시(구미초교)','성남시(한국학중앙연구원)']

#파일 경로
# path = f'../yearly_dataset/2014_dataset.csv'
for year in range(2014, 2025 + 1): #파일 경로 2014-2025 반복
    path = f'../yearly_dataset/{year}_dataset.csv'


    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")

    # MIET autocorrection 분석 (자기상관계수 분석)
    lags = range(1, 100)

    df['rf_sum'] = df[col_rf].sum(axis=1)
    print(df)

    # MIET autocorrection 분석 (자기상관계수 분석)
    lags = range(1, 100)

    autocorrs = [df['rf_sum'].autocorr(lag=lag) for lag in lags] # lag 1~99까지의 자기상관계수
    miet_autocorr = lags[np.argmin(np.abs(autocorrs))] # 43이 영향을 미치는 곳을 알 수가 없음;;
    print(f"Suggested MIET based on autocorrelation: {miet_autocorr} hours")
    MIET = miet_autocorr 

    results = []
    end_index = None

    # 강우 사상 루프
    for index, row in df.iterrows():
        # 강우량이 0보다 크면 (end_index가 정의되지 않은 최초 강우사상탐지)
        if end_index is None and row['rf_sum'] > 0:
            start_index = index # 시작 인덱스를 설정
            # 강우가 0 이상인 시점부터 +MIET 개까지로 강우기간 설정
            # end_index = df.index[df.index.get_loc(index) + MIET] 
            if df.index.get_loc(index) + MIET < len(df):
                end_index = df.index[df.index.get_loc(index) + MIET]
            else:
                end_index = df.index[-1]  # 데이터프레임 끝까지 설정
            print(f"Rainfall Start: {start_index}, Rainfall End: {end_index}")
            results_v2 = [start_index, end_index]
            results.append(results_v2)
        # 새로운 강우 이벤트 탐지 
        elif end_index is not None and row['rf_sum'] > 0 and index > end_index:
            start_index = index # 시작 인덱스를 설정        
            # 강우가 0 이상인 시점부터 +MIET 개까지로 강우기간 설정
            # end_index = df.index[df.index.get_loc(index) + MIET] 
            if df.index.get_loc(index) + MIET < len(df):
                end_index = df.index[df.index.get_loc(index) + MIET]
            else:
                end_index = df.index[-1]  # 데이터프레임 끝까지 설정
            print(f"Rainfall Start: {start_index}, Rainfall End: {end_index}")
            results_v2 = [start_index, end_index]
            results.append(results_v2)


    results = pd.DataFrame(results, columns=["Start Index", "End Index"])
    year = path.split('/')[-1].split('_')[0]
    results.to_csv(f'./Rainfall_event_{year}({MIET}).csv')