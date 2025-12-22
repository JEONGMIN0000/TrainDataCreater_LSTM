import os
import glob
import chardet
import pandas as pd
import numpy as np
from datetime import timedelta


def detect_encoding(file_path, n_lines=5000):
    with open(file_path, "rb") as f:
        raw = f.read(n_lines)
    return chardet.detect(raw)["encoding"]


def cut_interest_level_instants(
    df: pd.DataFrame, wl_col: str, threshold: float
) -> pd.DataFrame:
    """
    단일 수위 컬럼(wl_col)에 대해
    관심수위(threshold) 이상인 시점(row)만 남기는 함수.
    문자열로 읽힌 수위도 숫자로 변환해서 비교.
    """

    # 1) 수위 컬럼을 문자열로 캐스팅 후, 쉼표 제거 등 전처리
    s = df[wl_col].astype(str).str.replace(",", "", regex=False).str.strip()

    # 2) 숫자로 변환 (변환 안 되는 값은 NaN으로 처리)
    wl_numeric = pd.to_numeric(s, errors="coerce")

    # (선택) NaN이 있으면 로그 찍어서 확인
    if wl_numeric.isna().any():
        print(
            f"[WARN] {wl_col} 컬럼에서 숫자로 변환 안 된 값이 있습니다. (NaN 개수: {wl_numeric.isna().sum()})"
        )

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
    peak_idx = wl_exceed.idxmax()  # DatetimeIndex (피크 시각)
    peak_time = peak_idx

    # 4) 6시간 윈도우 계산
    post_hours = window_hours - pre_hours
    ideal_start = peak_time - pd.Timedelta(hours=pre_hours)
    ideal_end = peak_time + pd.Timedelta(hours=post_hours)

    # 5) 이벤트(이 파일) 전체 시간 범위
    event_start = df.index.min()
    event_end = df.index.max()

    # 먼저 start를 이벤트 범위 안으로
    start = max(ideal_start, event_start)
    end = start + pd.Timedelta(hours=window_hours)

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


# 관심 수위 이상인 데이터 추출
def process_miet_dir_to_ietd(
    miet_dir: str, out_dir: str, wl_col: str, threshold: float, skip_empty: bool = True
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


# 관심 수위 이상인 데이터 앞뒤 2.4시간씩 데이터 추출
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


# 관심 수위를 넘은 데이터가 포함된 파일 추출
def process_miet_dir_to_ietd_wholefile_if_exceed(
    miet_dir: str,
    out_dir: str,
    wl_col: str,
    threshold: float,
    skip_empty: bool = True,
):
    """
    MIET 폴더(miet_dir) 안의 각 강우사상 CSV에 대해:
    - wl_col이 threshold 이상인 row가 하나라도 있으면
        => 그 파일 전체를 out_dir에 저장
    - 없으면 (skip_empty=True인 경우) 스킵
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

        if wl_col not in df.columns:
            print(f"[WARN] {fname} : '{wl_col}' 컬럼이 없습니다. 스킵합니다.")
            continue

        # 수위 컬럼 숫자로 변환 (쉼표 제거 등 포함)
        s = df[wl_col].astype(str).str.replace(",", "", regex=False).str.strip()
        wl_numeric = pd.to_numeric(s, errors="coerce")

        cond = wl_numeric >= float(threshold)

        if not cond.any():
            if skip_empty:
                print(f"[SKIP] {fname} : 관심수위({threshold}) 도달 없음")
                continue
            else:
                # 관심수위 도달 안 해도 빈 DF라도 저장하고 싶다면 여기서 처리
                pass

        # 관심수위를 한 번이라도 넘었으면 => 파일 전체 저장
        out_path = os.path.join(out_dir, fname)

        # (선택) 수위 컬럼을 숫자로 덮어쓰고 싶으면:
        df[wl_col] = wl_numeric

        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {out_path} rows={len(df)}")


if __name__ == "__main__":

    # 궁내 = "gn", 대곡 = "dg"
    station_type = "gn"

    miet_data = [86, 61, 80, 96, 92, 74, 62, 98, 99, 69, 95, 97]
    years = range(2014, 2025 + 1)

    # 연도와 MIET를 1:1로 매핑
    for year, MIET in zip(years, miet_data):

        # 관측소 타입에 따라 설정 분기
        if station_type == "gn":  # 궁내
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

        miet_gn_dir = os.path.join(base_dir, "MIET", f"{year} 강우사상({MIET})")

        # ietd_gn_dir = os.path.join(base_dir, "IETD", f"{year} 관심 강우사상({MIET})")
        gn_dir = os.path.join(base_dir, "MIET", f"{year} 관심 강우사상({MIET})")

        # 관심 수위 이상인 데이터 추출
        # process_miet_dir_to_ietd(
        #     miet_dir=miet_gn_dir,
        #     out_dir=ietd_gn_dir,
        #     wl_col=wl_col,
        #     threshold=threshold
        # )

        # 관심 수위 이상인 데이터 앞뒤 2.4시간씩 데이터 추출
        # process_miet_dir_to_ietd_window6h(
        #     miet_dir=miet_gn_dir,
        #     out_dir=ietd_gn_dir,
        #     wl_col=wl_col,
        #     threshold=threshold,
        #     window_hours=6.0,  # 전체 6시간
        #     pre_hours=2.0,  # 피크 이전 2h + 이후 4h
        # )

        # 관심 수위를 넘은 데이터가 포함된 파일 추출
        process_miet_dir_to_ietd_wholefile_if_exceed(
            miet_dir=miet_gn_dir,
            out_dir=gn_dir,
            wl_col=wl_col,
            threshold=threshold,
            skip_empty=True,  # 관심수위 도달 못한 이벤트는 스킵
        )
