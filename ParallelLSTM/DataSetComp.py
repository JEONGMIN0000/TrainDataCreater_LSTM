import pandas as pd
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Data Set Feature

# Raw Data Column Info
rain_cols = ['R_한국학중앙연구원', 'R_대장동', 'R_구미초교', 'R_대곡교', 'R_성남북초교', 'R_남한산초교', 'R_궁내교_Ti', 'R_대곡교_Ti']
flow_cols = ['F_궁내교', 'F_대곡교']

# 환경 설정에서 x_cols / f_cols / y_cols 값 설정함.
num_x = len(x_cols)
num_f = len(f_cols)
num_y = len(y_cols)

config['num_of_feature'] = num_x
config['num_of_forecast']= num_f


# 기본 함수 정의
def getShape(v_list):
    if isinstance(v_list, list):
        shape = []
        while isinstance(v_list, list):
            shape.append(len(v_list))
            v_list = v_list[0] if len(v_list) > 0 else []
        return tuple(shape)
    return None

def checkDF(df):
    print(f'[ Dataframe Info -Shape:{df.shape}, -index:{df.index.name},{df.index.dtype} ]')
    summary = pd.DataFrame(df.dtypes, columns=['데이터 타입'])
    summary = summary.reset_index()
    summary = summary.rename(columns={'index': 'Feature'})
    summary['결측값 개수'] = df.isnull().sum().values
    summary['고유값 개수'] = df.nunique().values
    summary['첫 번째 값'] = df.iloc[0].values
    summary['두 번째 값'] = df.iloc[1].values
    summary['세 번째 값'] = df.iloc[2].values

    return summary

# 1차원 Dataframe 리스트의 각 df의 Shape 보기 함수 
def showShape1D(df_lst, width=10, title=' 1D Data Frame List ') :
    i = 0  
    print(f'[ {title} : {len(df_lst)} ]')
    for df in df_lst:
        print(f'{df.shape[0]},{df.shape[1]}', end='\t')
        i += 1
        if i%width == 0 :
            print('')
    print('')

# 시간 format을 yyyymmddhh24 형태로 변경하는 함수 ( 각 자료 처리에 사용 )
# ex, 2003091101
# dataframe의 각 행에 대해 각각 처리하기 위해서 apply(함수 or lamda 함수 ) 를 사용해야 함.

def formatYYYYMMDDHH(val) :
    datestr = str(val)
    if len(datestr) == 15 :
        return datestr[0:4] + datestr[5:7] + datestr[8:10] + '0' + datestr[11:12] 
    else :
        return datestr[0:4] + datestr[5:7] + datestr[8:10] + datestr[11:13] 

def formatYYYYMMDDHHMI(val) :
    datestr = str(val)
    if len(datestr) == 15 :
        return datestr[0:4] + datestr[5:7] + datestr[8:10] + '0' + datestr[11:12] + datestr[13:15]
    else :
        return datestr[0:4] + datestr[5:7] + datestr[8:10] + datestr[11:13] + datestr[14:16]

def getDurationStr(start_time, end_time) :
    diff_time = end_time - start_time
    hours, remainder = divmod(diff_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def getDurationFromNowStr(start_time) : 
    diff_time = datetime.now() - start_time
    hours, remainder = divmod(diff_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


# Data Loading
def read_excel_files(file_pattern):
    file_list = sorted(glob.glob(file_pattern), reverse=True)
    
    result = []
    
    # shape : [2 파일, 6 (filename, sheetnames, daataframes]
    for file in file_list:
        print(f'reading... {file}')

        sheets = pd.read_excel(file, sheet_name=None)
        
        sheet_data = [(file, sheet_name, df) for sheet_name, df in sheets.items()]
        result.append(sheet_data)
    
    return result

def makeDataFrame(file_pattern='./data/sungnam_*.xlsx'):

    data = read_excel_files(file_pattern)

    # shape : [2 파일, 6 (filename, sheetnames, daataframes]
    for file_sheets in data:
        print('[ File ]')
        # shape : [6 (filename, sheetnames, daataframes]
        for file, sheet_name, df in file_sheets:
            print(f"File: {file}, Sheet: {sheet_name}, DataFrame shape: {df.shape}")

    result = []

    #data[0]  : [ 6 (filename, sheetnames, daataframes) ]
    for i in range(len(data[0])):
        print(f"Two files's Sheet{i+1} is concatenated.")
        # concat : [ 6 (daataframes )  + 6 (daataframes ) ]
        event_df = pd.concat([data[0][i][2], data[1][i][2]], axis=1)
        result.append(event_df)
    
    return result

# excel 파일 처리를 위해서는 openpyxl 패키지 필요
# !pip install openpyxl

def cleanReform(df_list) :
    col_names  = ['DateTime', 'F_궁내교','R_한국학중앙연구원','R_대장동','R_구미초교','R_궁내교_Ti',
                    'DateTime2','F_대곡교', 'R_대곡교', 'R_성남북초교', 'R_남한산초교', 'R_대장동2', 'R_구미초교2', 'R_한국학중앙연구원2', 'R_대곡교_Ti' ]
    col_names2 = ['DateTime', 'R_한국학중앙연구원','R_대장동','R_구미초교', 'R_대곡교', 'R_성남북초교', 'R_남한산초교', 'R_대장동2', 'R_구미초교2', 'R_한국학중앙연구원2', 
                    'R_궁내교_Ti', 'R_대곡교_Ti' , 'F_궁내교', 'F_대곡교' ]

    cols_float64 = ['F_궁내교', 'F_대곡교'] 

    # index 0 은 표지
    for df in df_list:
        df.columns = col_names

    for df in df_list:
        diff_result = df[df.DateTime != df.DateTime2]
        print('- Different Time Count:', diff_result.DateTime.count())

    for i in range(len(df_list)):
        df_list[i] = df_list[i][1:]
        df_list[i].drop(columns=['DateTime2'], inplace=True)
        df_list[i].DateTime = df_list[i].DateTime.astype(str).str[0:16]
        df_list[i][cols_float64] = df_list[i][cols_float64].astype('float64')
        df_list[i] = df_list[i].reindex(columns=col_names2)


raw_dfs = makeDataFrame('./data/sungnam_*.xlsx')
raw_dfs = raw_dfs[1:]
cleanReform(raw_dfs)

# 최종 Raw Data 확인

for i, df in enumerate(raw_dfs):
    print(f'Event Number : {i+1}')
    display(checkDF(df))
    print('\n')

if is_save_data :
    os.makedirs('./model_stage', exist_ok=True)
    datafile = './model_stage/rawdata_all_event_df.pkl'
    with open(datafile, 'wb') as file:
        pickle.dump(raw_dfs, file)

    print(f'[{len(raw_dfs)} count] Event Data Sets List are saved in {datafile}')

    with open(datafile, 'rb') as file:
        temp = pickle.load(file)
        print(f'checking event count : {len(temp)}')

    for event in temp:
        print(f'checking df : {event.shape}')


# Data 검토 및 Visualization
# 그래프 관련 함수
# sub-graph 배열
# 횡축 : Set 종류
# 종축 : Input 유형

def drawSmallChart(v_set, isCompare, min_val, max_val, v_ax, v_width, y_label, title ) :
  
    v_set.plot.line(ax=v_ax)

    if isCompare :
        # Real 컬럼은 실선으로 설정
        for i in range(v_set.shape[1]//2):
            v_ax.lines[i].set_linestyle('-')
        
        # Simulation 컬럼은 점선으로 설정
        for i in range(v_set.shape[1]//2, v_set.shape[1]):
            v_ax.lines[i].set_linestyle('--')

    v_ax.set_title(title)
    v_ax.set_xlabel(' ', fontsize=12)
    v_ax.set_ylabel(y_label, fontsize=12)
    v_ax.grid(True, axis='y', linestyle='--')
    v_ax.tick_params(axis='x', rotation=45)
    v_ax.set_ylim([min_val, max_val])
    v_ax.legend(loc='upper left')
    
def drawRowChart(all_set, col_list, isCompare, min_val, max_val, lst_ax, y_label, grid_col, grid_row):
    for i in range(0,len(all_set)) :
        flow_only = all_set[i][col_list]
        # drawSmallChart(flow_only, isCompare, min_val, max_val, lst_ax[i], 0.8, y_label, f'DataSet {grid_row*grid_col + i}')
        drawSmallChart(flow_only, isCompare, min_val, max_val, lst_ax[i], 0.8, y_label, f' ')

def drawGridChart(df_list, col_list, isCompare, col_num, title):
    maxvalue = 0      
    for df in df_list:
        pd_max = df[col_list].max().max()
        if pd_max > maxvalue :
            maxvalue = pd_max
    # chartmax = (maxvalue//100 + 1)*100
    chartmax = maxvalue*1.05

    row_num = len(df_list) // col_num
    if len(df_list) % col_num > 0 :
        row_num = row_num + 1

    fig_W = 6
    fig_H = 6
    totalW= fig_W * col_num
    totalH =fig_H * row_num

    fig, ax = plt.subplots(row_num, col_num, figsize=(totalW, totalH), constrained_layout=True)

    if row_num > 1 :
        for i in range(0, row_num):
            startI = col_num*i
            endI   = col_num*(i+1)
            if endI > len(df_list) :
                endI = len(df_list)
            drawRowChart(df_list[startI:endI], col_list, isCompare, 0, chartmax, ax[i], '강수량/유량', col_num, i)
    else :
        drawRowChart(df_list[:col_num], col_list, isCompare, 0, chartmax, ax, '강수량/유량', col_num, 0)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    plt.show()

def drawDataSets(v_dataSets, v_columns):
    max_graph = 8
    drawGridChart(v_dataSets[0][:max_graph], v_columns, False, 2, f'< 유입 유출량 - Train >')
    drawGridChart(v_dataSets[1][:4]        , v_columns, False, 2, f'< 유입 유출량 - Validation >')
    drawGridChart(v_dataSets[2][:4]        , v_columns, False, 2, f'< 유입 유출량 - Test >')

%%skip isSkipDataChart Raw Data Chart is skipped.

max_graph = 8
# v_columns  = ['DateTime', 'R_한국학중앙연구원', 'R_대장동', 'R_구미초교', 'R_대곡교', 'R_성남북초교', 'R_남한산초교', 'R_궁내교_Ti', 'R_대곡교_Ti', 'F_궁내교', 'F_대곡교']
# v_columns = dataSets[0][0].columns
# v_columns  = ['R_대장동', 'R_구미초교']
v_columns  = ['R_한국학중앙연구원', 'R_대장동', 'R_구미초교', 'R_대곡교', 'R_성남북초교', 'R_남한산초교', 'R_궁내교_Ti', 'R_대곡교_Ti', 'F_궁내교', 'F_대곡교']
print('   ')
drawGridChart(raw_dfs[4:6], v_columns, False, 2, f'[ Feature Data ]')
print('   ')

def profiling_data(v_df, v_cols, is_detail=False):
    v_data = v_df[v_cols]

    if is_detail :
    display(pd.DataFrame(corr_matrix))

    corr_matrix = v_data.corr()
    # Plot the correlation heatmap
    plt.figure(figsize=(len(v_cols)//0.6, len(v_cols)//2))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    if is_detail :
    sns.pairplot(v_data)
    plt.show()


for i, df in enumerate(raw_dfs):
    print(f'[ {i+1} Data ]')
    profiling_data(df, x_cols, False)

def drawGraph(pd_lst, graph_sidx, graph_eidx, rain_cols, flow_cols, title):
    df_lst = pd_lst[graph_sidx:graph_eidx]
    graph_idx = graph_sidx

    base_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r',]
    # CSS4 색상
    css4_colors = list(mcolors.CSS4_COLORS.keys())

    # Tableau 색상
    tableau_colors = list(mcolors.TABLEAU_COLORS.keys())

    # Xkcd 색상 (상위 50개 사용)
    xkcd_colors = list(mcolors.XKCD_COLORS.keys())[:50]

    # 전체 색상 배열 결합
    colors = base_colors + css4_colors + tableau_colors + xkcd_colors
    
    plt.rcParams["figure.figsize"] = (20,6)

    for view_df in df_lst:
        # 조회 데이터 추출
        graph_idx += 1 
        # 그래프 크기 설정

        fig, ax1 = plt.subplots()  

        # 강수량
        lines = []
        for col in rain_cols:
            line, = ax1.plot(view_df[col], color=colors[len(lines)], linestyle='--', label=col)
            lines.append(line)
            
        ax1.set_ylabel('강수량', color='b')  
        ax1.tick_params(axis='y', labelcolor='b')      
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_xticks(view_df.index[::12])
        ax1.grid(True, axis='x', linestyle='--')

        # 유입유출량 
        ax2 = ax1.twinx()
        for col in flow_cols:
            line, = ax2.plot(view_df[col], color=colors[len(lines)], linewidth=2, label=col)
            lines.append(line)
            
        ax2.set_ylabel('유입/유출량', color='g')  
        ax2.tick_params(axis='y', labelcolor='g') 
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_xticks(view_df.index[::12])
        ax2.grid(True, axis='x', linestyle='--')

        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper right')  # 범례를 오른쪽 상단에 위치
        
        plt.title(f'[ {title} Data Set Number : {graph_idx} ]')
        plt.show()


def drawRainFlowDataSetsChart(v_dataSets, v_rain_cols, v_flow_cols):
    drawGraph(dataSets[0], 0, 10, v_rain_cols, v_flow_cols, 'Train')
    drawGraph(dataSets[1], 0, 10, v_rain_cols, v_flow_cols, 'Validation')
    drawGraph(dataSets[2], 0, 10, v_rain_cols, v_flow_cols, 'Test')

%%skip isSkipDataChart Raw Data Chart is skipped.

v_rain_cols = ['R_한국학중앙연구원', 'R_대장동', 'R_구미초교', 'R_대곡교', 'R_성남북초교', 'R_남한산초교', 'R_궁내교_Ti', 'R_대곡교_Ti']
v_flow_cols = ['F_궁내교', 'F_대곡교']
graph_start = 0
graph_end   = 10
drawGraph(raw_dfs, graph_start, graph_end, v_rain_cols, v_flow_cols, '[ Raw Data ]')
