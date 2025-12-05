
## nbconverter로 외부 실행시 default 값을 True로 변경하여 사용하기 위헤
# notebook의 tag 기능을 사용 ( cell의 tag에 parameters 삽입함 - 셀 삭제나 복사에 유의 ) 

# GPU 관련 설정 (가능한 비결정론적 알고리즘을 방지)
GPU_FIX = False

# 수동 실행 or 외부 설정 파일로 자동 실행 여부 ( True/False )
isSkipDataChart = True
isSkipPredictChart = True
isSkipPredictDetailChart = True

is_save_result = True
is_save_predict = True

is_save_model = False
is_read_config = False
is_save_data = False

use_sample_weight = False

# config name
config_name = 'sungnam_dg72'

# hyper parameter ( s:36, l:24-test_r2:0.9720897466704637)
predict = 72
seq_length = 36
num_of_cells = [64, 64]
num_of_batch = 16
learning_rate = 0.001

# MultiStep / multi_LSTM  : 72-1 / 36-2 / 24-3 / 18-4 / 12-6
num_of_layer = 24
feedback_win_size = 1

# DG Best 0.973456 : V2-T3, Layer 24, Seq Length 36, smoothing_type=2
# GN Best 0.952333 : V1-T3, Layer 24, Seq Length 36, smoothing_type=1, moving_avg_win_size=4

smoothing_type = 2
# moving average window size
moving_avg_win_size = 4
# exponential smoothing alpha
es_alpha = 0.65
# Kalman filter
proc_v_init = 1e-3
meas_v_init = 1e-3

# Available Loss Function 
# 'mean_squared_error'
# 'mean_absolute_error'
# 'huber_loss'
# 'mean_squared_logarithmic_error'
# 'mean_absolute_percentage_error'
loss_func = 'mean_squared_error'

# Activation Function : LeakyReLU / ReLU / tanh / linear
activation_func='linear'

# Data Set 설정
# 입력 : [x,f]
# 출력 : y
x_cols = ['R_한국학중앙연구원', 'R_대장동', 'R_구미초교', 'R_대곡교', 'R_성남북초교', 'R_남한산초교', 'R_궁내교_Ti', 'R_대곡교_Ti', 'F_궁내교', 'F_대곡교']
f_cols = ['R_한국학중앙연구원', 'R_대장동', 'R_구미초교', 'R_대곡교', 'R_성남북초교', 'R_남한산초교', 'R_궁내교_Ti', 'R_대곡교_Ti']

# x_cols = ['R_한국학중앙연구원', 'R_대장동', 'R_구미초교', 'F_궁내교']
# f_cols = ['R_한국학중앙연구원', 'R_대장동', 'R_구미초교']

# y_cols  = ['F_궁내교','F_대곡교']
y_cols = ['F_대곡교']

train_events = [0, 1, 4, 5]
valid_events = [2]
test_events  = [3]

# 모델 설정 및 결과를 저장할 파일 ( 파일이 없는 경우 생성후 head 정보 추가됨. ) 
model_result_file = 'model_result.csv'

# 예측시 건별 예측 여부
isBatch = True

# Papermill는 parameters block 다음 block에 삽입되어 대체 되므로 같은 block에 변수 변경 내용이 있으면 이전값이 적용됨.
# 반드시 대체된 내용으로 적용시 다음 block에서 처리

num_p_layer = predict//num_of_layer

# 수동 실행 or 외부 설정 파일로 자동 실행 여부 ( True/False )
# 통합 예보 사용 여부  

# multi-step의 Base 모델 ( LSTM4Multi)
base_model_path = './model_stage/multistep_base_model'
base_scaler_path = './model_stage/multistep_base_scaler.pkl'

# multi-step 모델 ( MultiStep Model )
multistep_model_path  = './model_stage/multistep_model'
multistep_scaler_path = './model_stage/multistep_scaler.pkl'


# Multi-Step의 Base 모델이 2개인 경우 
base_model_path1  = './model_stage/multistep_base_model1'
base_model_path2  = './model_stage/multistep_base_model2'
base_scaler_path2 = './model_stage/multistep_base_scaler1.pkl'
base_scaler_path2 = './model_stage/multistep_base_scaler2.pkl'

# 설정 정보를 저장할 dictionary
config = {
    "name"         : config_name,
    "use_model"    : "MultiLSTM",
    "seq_length"   : seq_length,         # 한번에 입력되는 연속된 데이터 시간
    "lead_time"    : 0,                  # label로 사용할 Y값을 몇 시간 뒤로 보낼지 결정
    "predict"      : predict,            # 예측 시간
    "forecast"     : predict,            # 강수 예보 사용 시간
    "y_idxs"       : [-1],               # 목표값 컬럼들의 index 리스트
    "train_events" : train_events,       # Train할 Event index 리스트
    "valid_events" : valid_events,       # Validation Event index 리스트
    "test_events"  : test_events,        # Test Event index 리스트
    "isExecTuning" : False,              # Tuning 실행 여부

    "num_of_feature"  : 0 ,              # input feature 갯수 ( data set 설정에 따라 수정됨 )
    "num_of_forecast" : 0 ,              # forecast 컬럼 갯수  ( data set 설정에 따라 수정됨 )

    "num_of_cells"  : num_of_cells,      # LSTM Layer의 Cell 수
    "num_of_layer"  : num_of_layer,      # 병렬 LSTM Layer 수
    "smoothing_type": smoothing_type,    # smoothing 유형 
    "ma_win_size"   : moving_avg_win_size,# moving average window size 
    "fb_win_size"   : feedback_win_size, # moving average window size 
    
    
    "num_of_epochs" : 200,               # 최대 Epoch 수
    "num_of_batch"  : num_of_batch,      # batch size
    "learning_rate" : learning_rate,     # learning rate
    "dropout_rate"  : 0.0,               # drop out 비율
    "isEarlystop"   : True,              # Early Stop 여부
    "patience_count": 10,                # Early Stop stop 조건
    "isCheckpoint"  : False              # check point 저장 여부 
}

# 설정 정보 출력
if is_read_config :
    config = read_config_file('config.conf')
print_config(config)