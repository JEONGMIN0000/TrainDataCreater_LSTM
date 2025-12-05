from tabulate import tabulate


#Data Set별 Event 선택
# 전체 데이터 중 특정 이벤트의 자료를 학습에서 제외 시키고 Validation/Test Set로 분할

def getEventList(df_lst, eventIdxs, isShuffle):
    event_list = []
    for i in range(0, len(df_lst)):
        if i in eventIdxs:
            event_list.append(df_lst[i])
    
    if isShuffle :
        random.shuffle(event_list)

    return event_list

# 각 세트의 From To Date 보기
def printFromToDate(df_1d_lst, title='Data Frame'):
    print(f'< {title} Count : {len(df_1d_lst)} >')
    print(' [ No.    Start             End     ]')
    for i, df in enumerate(df_1d_lst, 1):
        print(f' {i:>3}  {df.head(1).DateTime.values} ~ {df.tail(1).DateTime.values}')

def printDateDataSets(v_dataSets) :
    printFromToDate(v_dataSets[0], 'Train Set')
    printFromToDate(v_dataSets[1], 'Valid Set')
    printFromToDate(v_dataSets[2],  'Test Set')


def print_list_shape(nested_list, indent=0):
    if isinstance(nested_list, list):
        print('    '*indent + f"List - {len(nested_list)}")
        for item in nested_list:
            print_list_shape(item, indent + 1)
    else:
        print('    '*indent + f"Data Set - shape {nested_list.shape}")

# Train Set 선정
is_shuffle = False

# Train / Validation /Test Set 선정 
train_event_lst = getEventList(raw_dfs, config['train_events'], is_shuffle)
valid_event_lst = getEventList(raw_dfs, config['valid_events'], is_shuffle)
test_event_lst = getEventList(raw_dfs,  config['test_events'], is_shuffle)

dataSets = [train_event_lst, valid_event_lst, test_event_lst]

# 최종 입력 자료 Shape 보기 ( Shuffle 한 경우 순서 변경됨 )

showShape1D(dataSets[0], 10, 'Train Set')
showShape1D(dataSets[1], 10, 'Validation Set')
showShape1D(dataSets[2], 10, 'Test Set')

print('')
printDateDataSets(dataSets)

print('[ Data Set - 1 ]')
display(checkDF(dataSets[0][0]))

view_cols=2

# Data Set 별 Chart

# view_cols =  ['R_한국학중앙연구원', 'R_대장동', 'R_구미초교', 'R_대곡교', 'R_성남북초교', 'R_남한산초교', 'R_궁내교_Ti', 'R_대곡교_Ti', 'F_궁내교', 'F_대곡교']
view_cols = ['R_궁내교_Ti', 'R_대곡교_Ti', 'F_궁내교', 'F_대곡교']

drawDataSets(dataSets, view_cols)


#Data Scaling

# 스케일링시 Train Set 통계량을 활용해 모든 Data Set을 스케일링.
# 1. Test set의 정보는 학습시 입력하면 인됨.
# 2. 모델이 훈련용 데이터의 통계값으로 스케일링 된 데이터로 학습됨.
# 3. 결국 모든 Data Set을 훈련용 데이터의 통계량으로 스케일링해야 함.

# x : feature ( 'datetime/Y_유입량'을 제외한 전체 컬럼 )
# y : 예측값  ( Y_유입량 )

# 예측값의 시작 컬럼 index ( 맨마지막 컬럼은 -1 도 가능 )
# 예측값이 1개 이상인 경우는 -1/-2/-3 ... 이런 방식으로 지점
# scaled 된 dataframe list
def getScalers(train_dfs, x_cols, f_cols, y_cols):
    scaler_x = MinMaxScaler()
    scaler_f = MinMaxScaler()
    scaler_y = MinMaxScaler()

    for df in train_dfs:
        scaler_x.partial_fit(df.iloc[:][x_cols].values)
        scaler_f.partial_fit(df.iloc[:][f_cols].values)
        scaler_y.partial_fit(df.iloc[:][y_cols].values)

    return [scaler_x, scaler_f, scaler_y]

def scale_data(dataSets, scaler, cols):
    scaled_dataSets = []
    for dataSet in dataSets:
        scaled_list = []
        for df in dataSet:
            # 지정된 컬럼에 대해서만 transform 적용
            scaled_numeric = pd.DataFrame(scaler.transform(df[cols]), columns=cols)
            # 지정되지 않은 컬럼은 원래 값 유지
            non_scaled_cols = df.drop(cols, axis=1)
            # 원래의 컬럼 순서 유지
            scaled_df = pd.concat([non_scaled_cols.reset_index(drop=True), scaled_numeric.reset_index(drop=True)], axis=1)
            scaled_df = scaled_df[df.columns]  # 원래의 컬럼 순서로 정렬
            scaled_list.append(scaled_df)
        scaled_dataSets.append(scaled_list)
    return scaled_dataSets

# Train Data Set를 기준으로 3개의 scaler를 생성  
scalers = getScalers(dataSets[0], x_cols, f_cols, y_cols)

# 스케일러를 사용하여 데이터 세트를 스케일링처리하여 return
# Train / Valid / Test 각각에 Data Set가 있음  
scaled_dataSets = scale_data(dataSets, scalers[0], x_cols)

def printScalersInfo(scalers):
    for i, scaler in enumerate(scalers) :
        row_titles = ['Min', 'Max', 'Range', 'Scale']
        col_titles = [ f'col {i}' for i in range(scaler.n_features_in_) ]
        print('\n')
        print(f'[ Scaler {i} - {scaler.n_features_in_} Columns ]')
        print(tabulate([scaler.data_min_, scaler.data_max_,scaler.data_range_,scaler.scale_],  headers=col_titles, showindex=row_titles, tablefmt='grid'))

printScalersInfo(scalers)

# Scaled_DataSets 형태 

print('[ Scaled_DataSets Shape ]')
print_list_shape(scaled_dataSets)

# Train Set 첫번째 Data

display(checkDF(scaled_dataSets[0][0]))


# 시계열 데이터 생성 (DataX, DataF, DataY, Datal)
def printTSDataSetShape(dataset):
    print(f"trainX shape: {dataset[0][0].shape}")
    print(f"trainF shape: {dataset[0][1].shape}")
    print(f"trainY shape: {dataset[0][2].shape}")
    print(f"trainI shape: {dataset[0][3].shape}")
    print('')
    print(f"validX shape: {dataset[1][0].shape}")
    print(f"validF shape: {dataset[1][1].shape}")
    print(f"validI shape: {dataset[1][2].shape}")
    print(f"validY shape: {dataset[1][3].shape}")
    print('')
    print(f"testX shape: {dataset[2][0].shape}")
    print(f"testF shape: {dataset[2][1].shape}")
    print(f"testY shape: {dataset[2][2].shape}")
    print(f"testI shape: {dataset[2][3].shape}")
    print('')


def make_sequence_data(df_list, x_cols, f_cols, y_cols, seq_length, lead_time, predict_hour):
    dataX, dataF, dataY, dataI = [], [], [], []
    for df in df_list:
        for i in range(len(df) - seq_length - lead_time - predict_hour + 1):
            x = df.iloc[i:(i + seq_length)][x_cols].values
            f = df.iloc[(i + seq_length + lead_time):(i + seq_length + lead_time + predict_hour)][f_cols].values
            y = df.iloc[(i + seq_length + lead_time):(i + seq_length + lead_time + predict_hour)][y_cols].values
            i = df.iloc[(i + seq_length + lead_time)][['DateTime']].values
            dataX.append(x)
            dataF.append(f)
            dataY.append(y)
            dataI.append(i)
    return np.array(dataX), np.array(dataF), np.array(dataY), np.array(dataI)

def make_ts_dataset(df_lists, x_cols, f_cols, y_cols, seq_length, lead_time, predict_hour):
    data_set_ts = []
    for df_list in df_lists:
        x, f, y, i = make_sequence_data(df_list, x_cols, f_cols, y_cols, seq_length, lead_time, predict_hour)
        data_set_ts.append([x, f, y, i])
    return data_set_ts


# Data Set을 Time Series 형태로 변환 
seq_length = config['seq_length']  
lead_time  = config['lead_time']   
predict    = config['predict'] 

dataSets_org_ts = make_ts_dataset(dataSets, x_cols, f_cols, y_cols, seq_length, lead_time, predict) 
dataSets_ts = make_ts_dataset(scaled_dataSets, x_cols, f_cols, y_cols, seq_length, lead_time, predict)

# Time Series 형태의 Data Shape
print(' [ Time Series Data Shape ( before scaled ) ] ')
printTSDataSetShape(dataSets_org_ts)

print(' [ Time Series Data Shape ( after scaled )] ')
printTSDataSetShape(dataSets_ts)