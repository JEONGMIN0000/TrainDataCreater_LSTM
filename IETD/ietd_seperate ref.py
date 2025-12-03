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