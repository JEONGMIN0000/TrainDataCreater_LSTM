from IPython.display import display
from os import path
from datetime import datetime
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint
import tensorflow as tf
import keras

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.client import device_lib
from IPython.core.interactiveshell import InteractiveShell

RANDOM_SEED = 42

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# tf.set_random_seed(RANDOM_SEED) --> deprecated 
tf.random.set_seed(RANDOM_SEED)

# GPU 관련 설정 (가능한 경우 CUDA의 비결정론적 알고리즘을 방지)
if GPU_FIX :
    # Tensorflow 비결정론적 동작 방지
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # cuDNN 비결정론적 동작 방지
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  
    # 사용할 GPU를 고정
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 


# 2.16 이상의 경우 keras.utils.set_random_seed 로 전체 random seed 고정 지원함.( 아래 링크 참조 )
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed
# keras.utils.set_random_seed(RANDOM_SEED)

# jedi 자동 완성 사용하지 않음 
%config Completer.use_jedi = False

# 출력테스트 
print(keras.__version__)
print(tf.__version__)
print(device_lib.list_local_devices())

#자료 검토를 위해 컬럼과 행의 생략없이 출력하도록 옵션 변경

# row 생략 없이 100 행까지 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)

#모든 결과를 출력하도록 설정 [ all / last_expr ]
InteractiveShell.ast_node_interactivity = "last_expr"

def displayAll(isAll):
    if isAll==True :
        InteractiveShell.ast_node_interactivity = "all"
    else :
        InteractiveShell.ast_node_interactivity = "last_expr"

# 자동 실행시 tuning 실행 중단을 위한 함수 

class StopExecution(Exception):
    def _render_traceback_(self):
        # nbconvert error 시 return [] 사용
        return []
        # pass

def exit():
    raise StopExecution
    
# 해당 셀의 실행 중지 여부에 따라 중단하는 함수
def isStop(v_stop):
    if v_stop == True :
        print('isStop()에 의한 셀 실행 중지')
        exit()

# 설정 정보를 출력하는 함수
def print_result(v_result):
    print('[ Model Train and Predict Result ]')
    for key, value in v_result.items():
        print(f'{key}: {value}')

def print_config(v_config):
    print('# Execution Configuration')
    for key, value in v_config.items():
        print(f'{key}: {value}')

# 설정 정보를 파일에서 읽어오는 함수
def read_config_file(filename):
    config = {}

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # 줄의 앞뒤 공백 제거
            if not line or line.startswith('#'):  # 비어있거나 주석인 경우 무시
                continue

            key, value = line.split(':', 1)  # 첫 번째 ':' 기준으로 분리
            config[key.strip()] = eval(value.strip()) if key.strip() in ["input_data_types", "y_idxs", "train_events", "valid_events", "test_events"] else value.strip()  # 리스트로 변환
    return config
