import os
import pandas as pd

# 기본 
miet_data = [86, 61, 80, 96, 92, 74, 62, 98, 99, 69, 95, 97]

# 궁내교 컬럼만 유지 할 때
# miet_data = [85, 63, 82, 97, 92, 73, 59, 99, 88, 70, 95, 96]

years = range(2014, 2025 + 1)


#호우 조건
TI_SUM = 110    # 110mm 이상
WINDOW_HOURS = 12  # 12시간 누적

TI_COLS = ["궁내교_Ti", "대곡교_Ti"]  # Tiessen 강우 컬럼

#궁내교 컬럼
# keep_cols = [
#     '성남시(대장동)',
#     '성남시(구미초교)',
#     '성남시(한국학중앙연구원)',
#     '성남시(궁내교)_WL',
#     '성남시(궁내교)_Q',
#     '궁내교_Ti'
# ]


# Target_dam = '궁내' # 대곡 , 궁내 
for idx, year in enumerate(years):
    path = f'../yearly_dataset/{year}_dataset.csv'
    MIET = miet_data[idx]
    

    # Create folders if they don't exist
    rainfall_folder = f'./{year} 강우사상({MIET})'
    non_rainfall_folder = f'./{year} 무강우사상({MIET})'

    os.makedirs(rainfall_folder, exist_ok=True)
    os.makedirs(non_rainfall_folder, exist_ok=True)

    # miet를 기준으로 분리된 강우 사상 csv
    event = pd.read_csv(f'./Rainfall_event_{year}({MIET}).csv', index_col=0)

    rainfall = pd.read_csv(f'{path}', index_col=0, encoding="utf-8-sig")

    # Ti 컬럼 숫자화 (문자/결측 대비)
    for c in TI_COLS:
        rainfall[c] = pd.to_numeric(rainfall[c], errors="coerce").fillna(0)

    for i in range(len(event.index)):
        event_1 = list(event.iloc[i])
        
        print(i, event.index.get_loc(i), len(event) )
        if event.index.get_loc(i) < len(event)-1:
            event_2 = list(event.iloc[i + 1])    
        else:
            event_2 = list(event.iloc[-1])    
        print(i, event_1, event_2)

        # 기본
        rainfall_event = rainfall.loc[event_1[0]:event_1[1]].copy()

        # 각 Tiessen 기준으로 "12시간 누적 최대" 계산
        max12 = {}
        for col in TI_COLS:
            value = rainfall_event[col].rolling(WINDOW_HOURS, min_periods=WINDOW_HOURS).sum().max()
            max12[col] = 0 if pd.isna(value) else float(value)

        # 저장 조건: (둘 중 하나라도) 12시간 누적이 110mm 이상
        if (max12["궁내교_Ti"] >= TI_SUM) or (max12["대곡교_Ti"] >= TI_SUM):
            rainfall_event.to_csv(f'{rainfall_folder}/{year} {i+1}번 강우사상.csv', encoding="utf-8-sig")
            print(f'{year} {i+1}번 강우사상 저장')


        # 기본 저장 -----------------------------------------------------------------
        
        # rainfall_event.to_csv(f'{rainfall_folder}/{year} {i+1}번 강우사상.csv', encoding="utf-8-sig")
        # print(f'{year} {i+1}번 강우사상 저장')

        # non_rainfall_event = rainfall.loc[event_1[1]:event_2[0]]

        # non_rainfall_event.to_csv(f'{non_rainfall_folder}/{year} {i+1}번 무강우사상.csv', encoding="utf-8-sig")
        # print(f'{year} {i+1}번 무강우사상 저장')


        # 궁내교 컬럼만 유지하면서 강우 분리저장 ---------------------------------------

        # rainfall_event = rainfall.loc[event_1[0]:event_1[1], keep_cols]

        # rainfall_event.to_csv(f'{rainfall_folder}/{year} {i+1}번 강우사상.csv', encoding="utf-8-sig")
        # print(f'{year} {i+1}번 강우사상 저장')

        # ----------------------------------------------------------------------------