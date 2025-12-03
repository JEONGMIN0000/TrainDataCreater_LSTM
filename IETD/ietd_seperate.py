import os
import glob
import chardet
import pandas as pd
import numpy as np
from datetime import timedelta


def infer_dt_minutes(index: pd.DatetimeIndex) -> float:
    """
    DatetimeIndex에서 대표 시간 간격(분)을 추정
    - 가장 많이 나타나는 시간 간격을 사용 (mode)
    """
    diffs = index.to_series().diff().dropna()
    dt = diffs.mode().iloc[0]
    return dt.total_seconds() / 60.0


def split_rain_events_ietd(
    df: pd.DataFrame,
    rain_col: str = "rf",  # 강우 컬럼명
    ietd_hours: float = 6.0,  # IETD (시간 단위)
    dt_minutes: float | None = 10.0,  # 자료 시간 간격(분), None이면 자동 추정
    rain_threshold: float = 0.0,  # 유효 강우 임계값 (mm/Δt)
    min_event_depth: float = 0.0,  # 최소 사상 누적강우(mm) 필터
    include_dry_tail: bool = True,  # IETD 이하 무강우 구간을 이벤트 꼬리로 포함할지 여부
) -> pd.DataFrame:
    """
    IETD 기법에 의한 강우사상 분리.
    - df: DatetimeIndex를 가진 DataFrame (시간 오름차순 가정, 아니면 이 함수 내부에서 정렬)
    - rain_col: 강우(mm) 컬럼명
    - ietd_hours: IETD (시간)
    - dt_minutes: 시간 간격(분). None이면 index로부터 추정
    - rain_threshold: 이 값 이상이면 '강우 있음'으로 간주
    - min_event_depth: 이 값 미만인 이벤트는 제거 (event_id를 NaN으로 처리)
    - include_dry_tail:
        True  -> IETD 이하의 무강우 구간도 이벤트에 포함 (수위 반응 고려할 때 유용)
        False -> 강우가 있는 시점에만 event_id 부여
    반환:
        event_id 컬럼이 추가된 DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index는 DatetimeIndex여야 합니다.")

    df = df.sort_index().copy()

    if dt_minutes is None:
        dt_minutes = infer_dt_minutes(df.index)

    # IETD를 time-step 개수로 변환 (예: 6시간, 10분자료 -> 36 step)
    ietd_minutes = ietd_hours * 60.0
    ietd_steps = int(np.ceil(ietd_minutes / dt_minutes))

    rain = df[rain_col].fillna(0.0).values
    n = len(df)

    event_ids = np.full(n, np.nan, dtype=float)

    event_id = 0
    dry_steps = ietd_steps  # 처음에는 "충분히 건조하다" 가정

    for i in range(n):
        r = rain[i]
        wet = r >= rain_threshold

        if wet:
            # IETD 이상 건기 후 처음 비가 내리면 새로운 이벤트 시작
            if dry_steps >= ietd_steps:
                event_id += 1
            event_ids[i] = event_id
            dry_steps = 0

        else:
            # 비가 안 올 때
            if include_dry_tail and event_id > 0 and dry_steps < ietd_steps:
                # 아직 IETD에 도달하지 않았다면 이전 이벤트 꼬리로 포함
                event_ids[i] = event_id
                dry_steps += 1
            else:
                # 완전히 건기 or 이벤트 없음
                dry_steps = min(dry_steps + 1, ietd_steps)

    df["event_id"] = event_ids

    # 최소 이벤트 강우량 필터링 (너무 작은 이벤트 제거)
    if min_event_depth > 0:
        valid_ids = []
        for eid, group in df.groupby("event_id", dropna=True):
            depth = group[rain_col].sum()
            if depth >= min_event_depth:
                valid_ids.append(eid)

        df.loc[~df["event_id"].isin(valid_ids), "event_id"] = np.nan

    return df


def filter_events_by_water_level(
    events, water_level_data, threshold_level=2.5, time_lag=12
):
    """
    관심단계 수위 이상 사상만 필터링

    Parameters:
    -----------
    events : DataFrame
        분리된 강우 사상
    water_level_data : DataFrame
        columns: ['datetime', 'water_level']
    threshold_level : float
        관심단계 수위 (m)
    time_lag : int
        수위 응답 시간 (시간)
    """

    wl_df = water_level_data.copy()
    wl_df["datetime"] = pd.to_datetime(wl_df["datetime"])

    valid_events = []

    for idx, event in events.iterrows():
        # 분석 기간 설정
        analysis_start = event["start_time"] - timedelta(hours=1)
        analysis_end = event["end_time"] + timedelta(hours=time_lag)

        # 해당 기간 수위 추출
        mask = (wl_df["datetime"] >= analysis_start) & (
            wl_df["datetime"] <= analysis_end
        )
        period_wl = wl_df.loc[mask, "water_level"]

        if len(period_wl) > 0:
            max_wl = period_wl.max()

            # 관심단계 이상인 경우만 선택
            if max_wl >= threshold_level:
                event_dict = event.to_dict()
                event_dict["max_water_level"] = max_wl
                event_dict["water_level_rise"] = max_wl - period_wl.iloc[0]
                valid_events.append(event_dict)

    return pd.DataFrame(valid_events)


def summarize_rain_events(
    df: pd.DataFrame, rain_col: str = "rf", event_col: str = "event_id"
) -> pd.DataFrame:
    """
    event_id가 붙은 DataFrame에서 이벤트별 요약 통계 계산
    - 시작시각, 종료시각, 지속시간(분/시간), 총강우(mm), 최대강우강도(mm/Δt) 등
    """
    out = []

    for eid, group in df.groupby(event_col, dropna=True):
        start = group.index[0]
        end = group.index[-1]
        duration_min = (end - start).total_seconds() / 60.0
        total_rf = group[rain_col].sum()
        max_rf = group[rain_col].max()
        steps = len(group)

        out.append(
            {
                "event_id": eid,
                "start": start,
                "end": end,
                "duration_min": duration_min,
                "duration_hr": duration_min / 60.0,
                "n_steps": steps,
                "total_rf_mm": total_rf,
                "max_rf_mm_per_step": max_rf,
            }
        )

    return pd.DataFrame(out).set_index("event_id").sort_index()


#------------------------------------------------------------------------------------------------------

def detect_encoding(file_path, n_lines=5000):
    with open(file_path, 'rb') as f:
        raw = f.read(n_lines)
    return chardet.detect(raw)['encoding']


def cut_interest_level_instants(df: pd.DataFrame, wl_col: str, threshold: float) -> pd.DataFrame:
    """
    단일 수위 컬럼(wl_col)에 대해
    관심수위(threshold) 이상인 시점(row)만 남기는 함수.
    문자열로 읽힌 수위도 숫자로 변환해서 비교.
    """

    # 1) 수위 컬럼을 문자열로 캐스팅 후, 쉼표 제거 등 전처리
    s = df[wl_col].astype(str).str.replace(',', '', regex=False).str.strip()

    # 2) 숫자로 변환 (변환 안 되는 값은 NaN으로 처리)
    wl_numeric = pd.to_numeric(s, errors='coerce')

    # (선택) NaN이 있으면 로그 찍어서 확인
    if wl_numeric.isna().any():
        print(f"[WARN] {wl_col} 컬럼에서 숫자로 변환 안 된 값이 있습니다. (NaN 개수: {wl_numeric.isna().sum()})")

    # 3) 조건 마스크 생성 (숫자 기준 비교)
    cond = wl_numeric >= float(threshold)

    # 4) 조건 만족 row만 필터링
    df_cut = df[cond].copy()

    # (선택) 필터링된 DF의 해당 수위 컬럼은 숫자형으로 덮어쓰기
    df_cut[wl_col] = wl_numeric[cond]

    return df_cut


def cut_interest_level_window6h(
    df: pd.DataFrame,
    wl_col: str,
    threshold: float,
    window_hours: float = 6.0,
    pre_hours: float = 2.0,
) -> pd.DataFrame | None:
    """
    1) wl_col이 threshold 이상인 구간이 있는지 확인
    2) 그 중 피크 수위 시각(peak)을 anchor로 잡고
    3) 앞 pre_hours, 뒤 (window_hours - pre_hours) 만큼 붙여서
       총 window_hours 시간 길이의 구간을 잘라 반환.

    - df: 한 MIET 강우사상 CSV (time 컬럼 포함)
    - wl_col: 수위 컬럼 이름
    - threshold: 관심수위
    """

    if "time" not in df.columns:
        raise ValueError("DataFrame에 'time' 컬럼이 필요합니다.")

    # 1) 시간 파싱 + 인덱스로 설정
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.set_index("time").sort_index()

    # 2) 수위 컬럼 숫자로 변환
    s = df[wl_col].astype(str).str.replace(",", "", regex=False).str.strip()
    wl_numeric = pd.to_numeric(s, errors="coerce")

    cond = wl_numeric >= float(threshold)

    if not cond.any():
        # 관심수위 도달 안 한 강우사상
        return None

    # 3) 관심수위 이상 구간에서 피크 찾기
    wl_exceed = wl_numeric[cond]
    peak_idx = wl_exceed.idxmax()      # DatetimeIndex (피크 시각)
    peak_time = peak_idx

    # 4) 6시간 윈도우 계산
    post_hours = window_hours - pre_hours
    ideal_start = peak_time - pd.Timedelta(hours=pre_hours)
    ideal_end   = peak_time + pd.Timedelta(hours=post_hours)

    # 5) 이벤트(이 파일) 전체 시간 범위
    event_start = df.index.min()
    event_end   = df.index.max()

    # 먼저 start를 이벤트 범위 안으로
    start = max(ideal_start, event_start)
    end   = start + pd.Timedelta(hours=window_hours)

    # end가 이벤트 끝을 넘으면 뒤에서 다시 맞춰줌
    if end > event_end:
        end = event_end
        start = end - pd.Timedelta(hours=window_hours)

    # 실제 길이가 여전히 부족하면 스킵
    actual_hours = (end - start).total_seconds() / 3600.0
    if actual_hours < window_hours - 1e-6:
        return None

    # 6) 최종 슬라이싱
    df_win = df.loc[start:end].copy()

    # 수위 컬럼은 숫자형으로 덮어쓰기 (선택)
    df_win[wl_col] = wl_numeric.loc[df_win.index]

    # time 컬럼 다시 넣어두면 CSV로 저장하기 편함
    df_win = df_win.reset_index().rename(columns={"time": "time"})

    return df_win


def process_miet_dir_to_ietd(
        miet_dir: str,
        out_dir: str,
        wl_col: str,
        threshold: float,
        skip_empty: bool = True
    ):
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(miet_dir):
        if not fname.lower().endswith(".csv"):
            continue

        in_path = os.path.join(miet_dir, fname)

        # 🔍 자동 인코딩 감지
        encoding = detect_encoding(in_path)
        print(f"[INFO] {fname} detected encoding = {encoding}")

        # CSV 로드
        df = pd.read_csv(in_path, encoding=encoding)

        # time 파싱
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # 필터링
        df_cut = cut_interest_level_instants(df, wl_col=wl_col, threshold=threshold)

        if df_cut.empty and skip_empty:
            print(f"[SKIP] {fname} : 관심수위 도달 없음")
            continue

        out_path = os.path.join(out_dir, fname)
        df_cut.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {out_path} rows={len(df_cut)}")

    
def process_miet_dir_to_ietd_window6h(
    miet_dir: str,
    out_dir: str,
    wl_col: str,
    threshold: float,
    window_hours: float = 6.0,
    pre_hours: float = 2.0,
    skip_empty: bool = True,
    ):
    """
    MIET 폴더(miet_dir) 안의 각 강우사상 CSV에 대해:
    - 관심수위(threshold)를 넘는 이벤트가 있는 경우
    - 피크 시각 기준 6시간(window_hours) 구간으로 잘라서 out_dir에 저장
    """

    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(miet_dir):
        if not fname.lower().endswith(".csv"):
            continue

        in_path = os.path.join(miet_dir, fname)

        # 🔍 자동 인코딩 감지
        encoding = detect_encoding(in_path)
        print(f"[INFO] {fname} detected encoding = {encoding}")

        # CSV 로드
        df = pd.read_csv(in_path, encoding=encoding)

        # 6시간 윈도우 자르기
        df_win = cut_interest_level_window6h(
            df=df,
            wl_col=wl_col,
            threshold=threshold,
            window_hours=window_hours,
            pre_hours=pre_hours,
        )

        if (df_win is None or df_win.empty) and skip_empty:
            print(f"[SKIP] {fname} : 관심수위 도달 없음 또는 6시간 윈도우 생성 실패")
            continue

        # 출력 파일 경로 (원래 이름 그대로 쓰거나, 접미어를 붙여도 됨)
        out_path = os.path.join(out_dir, fname)
        df_win.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {out_path} rows={len(df_win)}")


if __name__ == "__main__":

    # 궁내 = "gn", 대곡 = "dg"
    station_type = "gn"

    miet_data = [86, 61, 80, 96, 92, 74, 62, 98, 99, 69, 95, 97]
    years = range(2014, 2025 + 1)

    # 연도와 MIET를 1:1로 매핑
    for year, MIET in zip(years, miet_data):

        # 관측소 타입에 따라 설정 분기
        if station_type == "gn":   # 궁내
            name = "궁내교"
            wl_col = "성남시(궁내교)_WL"
            threshold = 2.0

        elif station_type == "dg":  # 대곡
            name = "대곡교"
            wl_col = "서울시(대곡교)_WL"
            threshold = 3.8

        else:
            # 기본값: 궁내
            name = "궁내교"
            wl_col = "성남시(궁내교)_WL"
            threshold = 2.0

        base_dir = ".."

        miet_gn_dir = os.path.join(base_dir, "MIET", f"{year} 학습데이터 강우사상({MIET})")

        ietd_gn_dir = os.path.join(base_dir, "IETD", f"{name} {year} 학습데이터 관심 강우사상({MIET})")

        # process_miet_dir_to_ietd(
        #     miet_dir=miet_gn_dir,
        #     out_dir=ietd_gn_dir,
        #     wl_col=wl_col,
        #     threshold=threshold
        # )

        process_miet_dir_to_ietd_window6h(
            miet_dir=miet_gn_dir,
            out_dir=ietd_gn_dir,
            wl_col=wl_col,
            threshold=threshold,
            window_hours=6.0,   # 전체 6시간
            pre_hours=2.0,      # 피크 이전 2h + 이후 4h
        )
