import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# ---------- 1. 기본 설정 ----------

# 궁내 = "gn", 대곡 = "dg"
station_type = ["gn","dg"]

# IETD 폴더 경로
file_dir = r"./IETD"   

# 한글 폰트 (윈도우 기준: 맑은 고딕)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------- 2. IETD 폴더 안 모든 CSV 찾기 ----------

file_pattern = os.path.join(file_dir, "**", "*.csv")
csv_file = glob.glob(file_pattern,recursive=True)

for csv_path in csv_file:

    # ---------- 3. CSV 읽기 ----------

    df = pd.read_csv(csv_path)

    # 필요한 컬럼 이름 
    time_col = "time"
    gn_wl_col = "성남시(궁내교)_WL"
    gn_ti_col = "궁내교_Ti"
    dg_wl_col = "서울시(대곡교)_WL"
    dg_ti_col = "대곡교_Ti"

    #CSV 하나당 station_type 두 번 반복 ("gn", "dg")
    for station in station_type:

        if station == "gn" :
            wl_col = gn_wl_col
            ti_col = gn_ti_col
            station_name = "궁내교"
        elif station == "dg" :
            wl_col = dg_wl_col
            ti_col = dg_ti_col
            station_name = "대곡교"

        col_list = [time_col, wl_col, ti_col]

        # 시간 컬럼을 datetime 으로 변환
        time = pd.to_datetime(df[time_col])

        # ---------- 4. 데이터 숫자 변환 ----------

        wl = pd.to_numeric(df[wl_col], errors="coerce")
        rf = pd.to_numeric(df[ti_col], errors="coerce")

        # ---------- 5. 그래프 생성 ----------

        fig, ax1 = plt.subplots(figsize =(12,6))

        # 수위 선 그래프
        ax1.plot(time, wl, color="blue", linewidth=2)
        ax1.set_xlabel("시간")
        ax1.set_ylabel("수위 (m)")
        ax1.grid(alpha=0.3)

        # 강우 막대 그래프
        ax2 = ax1.twinx()
        ax2.bar(time, rf, width=0.005, alpha=0.6)
        ax2.set_ylabel("강우 (Ti)")
        ax2.set_ylim(rf.max() * 1.2, 0) 

        title = os.path.basename(csv_path) + f" - {station_name} 수위 / 강우"
        ax1.set_title(title)

        plt.tight_layout()

        base, ext = os.path.splitext(csv_path)

        save_path = base + f"_{station_name}.png"

        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f" [저장] {station_name} 그래프 : ", save_path)



