import os
import pandas as pd

miet_data = [86, 61, 80, 96, 92, 74, 62, 98, 99, 69, 95, 97]
years = range(2014, 2025 + 1)

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

    for i in range(len(event.index)):
        event_1 = list(event.iloc[i])
        
        print(i, event.index.get_loc(i), len(event) )
        if event.index.get_loc(i) < len(event)-1:
            event_2 = list(event.iloc[i + 1])    
        else:
            event_2 = list(event.iloc[-1])    
        print(i, event_1, event_2)
        rainfall_event = rainfall.loc[event_1[0]:event_1[1]]

        rainfall_event.to_csv(f'{rainfall_folder}/{year} {i+1}번 강우사상.csv')
        print(f'{year} {i+1}번 강우사상 저장')

        non_rainfall_event = rainfall.loc[event_1[1]:event_2[0]]

        non_rainfall_event.to_csv(f'{non_rainfall_folder}/{year} {i+1}번 무강우사상.csv')
        print(f'{year} {i+1}번 무강우사상 저장')
