import os
import chardet
import pandas as pd
import numpy as np


# 간격 추정
def infer_dt_minutes(index: pd.DatetimeIndex) -> float:
    diffs = index.to_series().diff().dropna()
    dt = diffs.mode().iloc[0]
    return dt.total_seconds() / 60.0


# IETD 함수
def split_rain_events_ietd(
    df: pd.DataFrame,
    rain_col: pd.Series | None = None,   # tfsum 등
    ietd_hours: float = 6.0,                # IETD (시간 단위)
    dt_minutes: float | None = 10.0,        # 자료 시간 간격(분), None이면 자동 추정
    rain_threshold: float = 0.0,            # 유효 강우 임계값 (mm/Δt)
    min_event_depth: float = 0.0,           # 최소 사상 누적강우(mm) 필터
    include_dry_tail: bool = True,          # IETD 이하 무강우 구간을 이벤트 꼬리로 포함할지 여부
) -> pd.DataFrame:
    
    """
    관심단계 수위 이상 사상만 필터링

    Parameters:
    events : DataFrame
        분리된 강우 사상
    water_level_data : DataFrame
        columns: ['datetime', 'water_level']
    threshold_level : float
        관심단계 수위 (m)
    time_lag : int
        수위 응답 시간 (시간)

    IETD 기법에 의한 강우사상 분리.
    - df: DatetimeIndex를 가진 DataFrame (시간 오름차순 가정)
    - rain_col: 여러 관측소 합 tfsum
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index는 DatetimeIndex여야 합니다.")

    df = df.sort_index().copy()

    rain_s = rain_col.reindex(df.index).fillna(0.0)

    # IETD를 time-step 개수로 변환
    ietd_minutes = ietd_hours * 60.0
    ietd_steps = int(np.ceil(ietd_minutes / dt_minutes))

    rain = rain_s.values
    n = len(df)

    event_ids = np.full(n, np.nan, dtype=float)

    event_id = 0
    dry_steps = ietd_steps  # 처음에는 "충분히 건조하다" 가정

    for i in range(n):
        r = rain[i]
        wet = r > rain_threshold

        if wet:
            # 처음 비가 내리면 새로운 이벤트 시작
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
            depth = rain_s.loc[group.index].sum()  # tfsum 기준으로 깊이 계산
            if depth >= min_event_depth:
                valid_ids.append(eid)

        df.loc[~df["event_id"].isin(valid_ids), "event_id"] = np.nan

    return df


# 인코딩
def detect_encoding(file_path, n_bytes=5000):
    with open(file_path, "rb") as f:
        raw = f.read(n_bytes)
    return chardet.detect(raw)["encoding"]


# IETD + 수위
def process_year_with_ietd(
    year: int,
    station_type: str = "gn",      # "gn" = 궁내, "dg" = 대곡
    ietd_hours: float = 6.0,       # IETD (시간)
    dt_minutes: float = 10.0,      # 자료 시간간격 (분)
    rain_threshold: float = 0.0,   # rf > rain_threshold 
    min_event_depth: float = 0.0,  # 이벤트 최소 누적강우 (mm)
    add_hours_1: float = 6.0,        # 이벤트 앞 패딩
    add_hours_2: float = 6.0,        # 이벤트 뒤 패딩
):

    # 1) 관측소별 수위 컬럼 / 관심수위 설정
    if station_type == "gn":
        name = "궁내교"
        wl_col = "성남시(궁내교)_WL"
        wl_threshold = 2.0
    elif station_type == "dg":
        name = "대곡교"
        wl_col = "서울시(대곡교)_WL"
        wl_threshold = 3.8
    else:
        raise ValueError("station_type은 'gn' 또는 'dg'만 허용")

    # 2) 연도별 원본 데이터 로드
    in_path = f"../yearly_dataset/{year}_dataset.csv"
    encoding = detect_encoding(in_path)
    print(f"[INFO] Load {in_path} (encoding={encoding})")

    df = pd.read_csv(in_path, encoding=encoding)

    # 3) 시간 인덱스 설정
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.set_index("time").sort_index()

    # 4) 강우 합산 컬럼
    rf_cols = [
        "서울시(대곡교)",
        "성남시(성남북초교)",
        "광주시(남한산초교)",
        "성남시(대장동)",
        "성남시(구미초교)",
        "성남시(한국학중앙연구원)",
    ]
    tfsum = df[rf_cols].sum(axis=1)

    # 5) 수위 컬럼 문자열로 캐스팅 후, 쉼표 제거 등 전처리
    wl_series = (df[wl_col].astype(str).str.replace(",", "", regex=False).str.strip())

    # 5-1) 숫자로 변환 (변환 안 되는 값은 NaN으로 처리)
    df[wl_col] = pd.to_numeric(wl_series, errors="coerce")

    # 6) IETD로 강우사상 분리
    df_evt = split_rain_events_ietd(
        df,
        rain_col=tfsum,
        ietd_hours=ietd_hours,
        dt_minutes=dt_minutes,
        rain_threshold=rain_threshold,
        min_event_depth=min_event_depth,
        include_dry_tail=True,
    )

    # 7) 이벤트 중, 관심수위 이상 도달한 사상만 골라서 ±6시간 후 저장
    # out_dir = f"./{year}_학습데이터_{name}_IETD"
    out_dir = f"./{year}_학습데이터"
    os.makedirs(out_dir, exist_ok=True)

    total_events = 0
    saved_events = 0

    for id, group in df_evt.groupby("event_id", dropna=True):
        total_events += 1

        # (1) 이벤트 시간 범위
        event_start = group.index.min()
        event_end = group.index.max()

        # (2) 이벤트 내에서 관심수위 이상 도달 여부 확인
        over = group[wl_col] >= wl_threshold
        if not over.any():
            # 관심수위에 도달하지 않은 이벤트는 스킵
            continue

        # (3) 관심수위 포함된 이벤트면 원본 df 기준으로 ±6시간 확장
        start_pad = max(df.index.min(), event_start - pd.Timedelta(hours=add_hours_1))
        end_pad = min(df.index.max(), event_end + pd.Timedelta(hours=add_hours_2))

        print(f"[start_pad] {start_pad} ")
        print(f"[end_pad] {end_pad} ")

        event_df = df.loc[start_pad:end_pad].reset_index()

        # (4) 파일로 저장
        out_name = f"{year}_event_{int(id):03d}.csv"
        out_path = os.path.join(out_dir, out_name)
        event_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        saved_events += 1

        print(f"[SAVE] {out_path} rows={len(event_df)}")

    print(f"[SUMMARY] {year}년 총 이벤트 {total_events}개 중 관심수위 이상 이벤트 {saved_events}개 저장 (관측소: {name})")


if __name__ == "__main__":

    for year in range(2014, 2025 + 1):
        process_year_with_ietd(
            year=year,
            station_type="gn",      # "gn" or "dg"
            ietd_hours=6.0,         # IETD (시간)
            dt_minutes=10.0,        # 자료 시간간격 (분)
            rain_threshold=0.0,     # rf > rain_threshold 
            min_event_depth=0.0,    # 이벤트 최소 누적강우 (mm) -> 너무 작은 비 제외하려면 예: 3.0 같은 값으로
            add_hours_1=6.0,        # 이벤트 앞 패딩
            add_hours_2=6.0,        # 이벤트 뒤 패딩  
        )