import os
import pandas as pd

from collect import hrfcodict_list
from util import thissen_DG, thissen_GN
from const_sungnam import (RF_OBSCD, RF_COL_NAME, DG_THISSEN_AREA, GN_THISSEN_AREA, 
                            DG_TI_COL, GN_TI_COL, WL_OBSCD, WL_COL_NAME,)


# --------------------------------- 1. 홍수통제소 API 호출 유틸 함수 ---------------------------------


# 한 강우관측소의 1년 데이터(강수량) 조회 & 병합
def fetch_rf (code: str, col_name: str, year: int) -> pd.DataFrame:

    dfs = []

    date_ranges = [
    (f"{year}01010000", f"{year}01312350"),  # 1월
    (f"{year}02010000", f"{year}02292350"),  # 2월 (윤년 반영하려면 별도 코드 필요)
    (f"{year}03010000", f"{year}03312350"),  # 3월
    (f"{year}04010000", f"{year}04302350"),  # 4월
    (f"{year}05010000", f"{year}05312350"),  # 5월
    (f"{year}06010000", f"{year}06302350"),  # 6월
    (f"{year}07010000", f"{year}07312350"),  # 7월
    (f"{year}08010000", f"{year}08312350"),  # 8월
    (f"{year}09010000", f"{year}09302350"),  # 9월
    (f"{year}10010000", f"{year}10312350"),  # 10월
    (f"{year}11010000", f"{year}11302350"),  # 11월
    (f"{year}12010000", f"{year}12312350"),  # 12월
]


    #데이터 조회
    for(start_date, end_date) in date_ranges:         
        data_list = hrfcodict_list('rf', code, start_date, end_date)
        print(start_date, end_date)
        print(data_list[0], data_list[-1])
        if not data_list:
            continue
        df = pd.DataFrame(data_list)
        df['time'] = pd.to_datetime(df['time']) # time 컬럼 datetime으로 변환
        df = df.rename(columns={'rf': col_name}) # rf 컬럼명 관측소명으로 변경
        dfs.append(df)
    
    # 데이터 null 예외처리
    if not dfs: 
        return pd.DataFrame(columns=['time', col_name])
    
    #상반기 하반기 데이터 병합
    year_data = pd.concat(dfs, ignore_index=True) 

    #중복 시간 제거
    year_data = year_data.drop_duplicates(subset=['time']).sort_values('time') 

    return year_data


# 한 수위관측소의 1년 데이터(수위/유량) 조회 & 병합
def fetch_wl (code: str, wl_name: str, year: int) -> pd.DataFrame:

    dfs = []

    date_ranges = [
    (f"{year}01010000", f"{year}01312350"),  # 1월
    (f"{year}02010000", f"{year}02282350"),  # 2월 (윤년 반영하려면 별도 코드 필요)
    (f"{year}03010000", f"{year}03312350"),  # 3월
    (f"{year}04010000", f"{year}04302350"),  # 4월
    (f"{year}05010000", f"{year}05312350"),  # 5월
    (f"{year}06010000", f"{year}06302350"),  # 6월
    (f"{year}07010000", f"{year}07312350"),  # 7월
    (f"{year}08010000", f"{year}08312350"),  # 8월
    (f"{year}09010000", f"{year}09302350"),  # 9월
    (f"{year}10010000", f"{year}10312350"),  # 10월
    (f"{year}11010000", f"{year}11302350"),  # 11월
    (f"{year}12010000", f"{year}12312350"),  # 12월
]


    # 유량 컬럼명 (예: 성남시(궁내교)_WL -> 성남시(궁내교)_Q)
    q_name = wl_name.replace('_WL', '_Q')

    #데이터 조회
    for(start_date, end_date) in date_ranges: 
        data_list = hrfcodict_list('wl', code, start_date, end_date)
        
        print(start_date, end_date)
        print(data_list[0], data_list[-1])
        if not data_list:
            continue
        df = pd.DataFrame(data_list)
        df['time'] = pd.to_datetime(df['time']) # time 컬럼 datetime으로 변환
        df = df.rename(columns={'wl': wl_name, 'fw':q_name}) # rf 컬럼명 관측소명으로 변경 (wl 수위 q 유량)
        dfs.append(df)
    
    # 데이터 null 예외처리
    if not dfs: 
        return pd.DataFrame(columns=['time', wl_name, q_name])
    
    #상반기 하반기 데이터 병합
    year_data = pd.concat(dfs, ignore_index=True) 

    #중복 시간 제거 및 시간 정렬
    year_data = year_data.drop_duplicates(subset=['time']).sort_values('time') 

    return year_data    


# --------------------------------- 2. 연도별 데이터셋 구성 ---------------------------------


# 해당 연도 10분단위 강우/수위/유량 데이터 취합 & csv 저장
def build_year_dataset(year:int, out_dir:str="./output"):
    
    os.makedirs(out_dir, exist_ok=True)

    base_df = None

    # ---- 2-1. 강우 관측소 전체 병합 ----
    for code, rf_name in zip(RF_OBSCD, RF_COL_NAME):
        df_rf = fetch_rf(code, rf_name, year)
        if df_rf.empty:
            continue
        
        if base_df is None:
            base_df=df_rf
        else:
            base_df = pd.merge(base_df, df_rf, on='time', how='outer')

        print(f"[{year}] 강우 수집  결과: {df_rf.shape} ")

    if base_df is None:
        print(f"[{year}] 강우 데이터가 없습니다.")
        return
    
    # ---- 2-2. 수위/유량 관측소 병합 ----
    for code, wl_name in zip(WL_OBSCD, WL_COL_NAME):
        df_wl = fetch_wl(code, wl_name, year)
        if df_wl.empty:
            continue

        if base_df is None:
            base_df=df_wl
        else:
            base_df = pd.merge(base_df, df_wl, on='time', how='outer')

        print(f"[{year}] 수위/유량 수집  결과: {df_wl.shape} ")

    # time 정렬
    base_df = base_df.sort_values('time').reset_index(drop=True)


    # ----------------------------- 3. 티센 유역 평균 강우 계산 -----------------------------
    # DG_THISSEN_AREA, GN_THISSEN_AREA 이 RF_COL_NAME 과 동일할 때


    # 대곡교 유역 강우
    if all(col in base_df.columns for col in DG_THISSEN_AREA):
        obs_DG = base_df[DG_THISSEN_AREA].to_numpy()
        base_df[DG_TI_COL[0]] = thissen_DG(obs_DG)
    else:
        print(f"[{year}] 경고 : DG_THISSEN_AREA중 컬럼 누락으로 대곡교_Ti 계산 생략")

    # 궁내교 유역 강우
    if all(col in base_df.columns for col in GN_THISSEN_AREA):
        obs_GN = base_df[GN_THISSEN_AREA].to_numpy()
        base_df[GN_TI_COL[0]] = thissen_GN(obs_GN)
    else:
        print(f"[{year}] 경고 : GN_THISSEN_AREA 컬럼 누락으로 궁내교_Ti 계산 생략")


    # ----------------------------- 4. 컬럼 순서 정리 -----------------------------


    # 수위/유량 컬럼명 정리
    wl_q_cols = []
    for wl_name in WL_COL_NAME:
        q_name = wl_name.replace('_WL','_Q')
        wl_q_cols.append(wl_name)
        wl_q_cols.append(q_name)

    # 최종 컬럼 순서 (존재하는 것만 남김)
    column_order = (
        ['time'] +
        RF_COL_NAME +           # 강우 관측소별 강우
        wl_q_cols  +            # 수위/유량
        DG_TI_COL +             # 대곡교_Ti
        GN_TI_COL               # 궁내교_Ti
    )

    existing_cols = [col for col in column_order if col in base_df.columns]

    base_df = base_df[existing_cols]


    # ----------------------------- 5. CSV 저장 -----------------------------


    out_path = os.path.join(out_dir, f"{year}_dataset.csv")
    base_df.to_csv(out_path, index=False, encoding='utf-8-sig') # 엑셀에서 한글 깨지지 않게 인코딩

    print(f"[저장 완료] {out_path}")


# -------------------- 6. 실행 -------------------- #


if __name__ == "__main__":
    # 2014~2025년까지 연도별 생성
    for y in range(2014, 2026):
        build_year_dataset(y, out_dir="./yearly_dataset")






