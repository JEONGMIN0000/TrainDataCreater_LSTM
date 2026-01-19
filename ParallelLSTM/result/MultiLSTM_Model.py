import numpy as np
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Input, concatenate, Reshape
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

predict = 36 # 예측기간
seq_length = 36 # 선행시간
num_of_cells = [64, 64]
num_of_batch = 64
learning_rate = 0.0001

num_of_layer = 18
feedback_win_size = 1

# DG Best 0.973456 : V2-T3, Layer 24, Seq Length 36, smoothing_type=2
# GN Best 0.952333 : V1-T3, Layer 24, Seq Length 36, smoothing_type=1, moving_avg_win_size=4

# GN = 1 , DG = 2
smoothing_type = 2
moving_avg_win_size = 4
es_alpha = 0.65
proc_v_init = 1e-3
meas_v_init = 1e-3

loss_func = 'mean_squared_error'
activation_func='linear'

# config name
config_name = 'sungnam_dg_ver3'
config = {
    "name"         : config_name,
    "use_model"    : "MultiLSTM",
    "seq_length"   : seq_length,         # 한번에 입력되는 연속된 데이터 시간
    "lead_time"    : 0,                  # label로 사용할 Y값을 몇 시간 뒤로 보낼지 결정
    "predict"      : predict,            # 예측 시간
    "forecast"     : predict,            # 강수 예보 사용 시간
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



# v_num_y : 예측 feature 수 
# v_step : 예측 시간 (step)
# num_layers : layer 수
# num_p_layer : layer 당 예측수 
class MultiLSTM_Model(Model):
    def __init__(self, rf_units, fc_units, num_layers, num_p_layer, v_num_x, v_num_f, v_num_y, v_step,  v_fc_step=None, prev_steps=feedback_win_size):
        super().__init__()
        self.input1 = Input(shape=(None, v_num_x), name='input_rainflow')
        self.input2 = Input(shape=(None, v_num_f), name='input_forecast')

        self.rf_units = rf_units
        self.fc_units = fc_units
        self.step = v_step
        self.fc_horizon  = v_fc_step or v_step  # 예보 horizon (없으면 기본은 v_step)
        self.num_y = v_num_y
        self.num_layers = num_layers
        self.num_p_layer = num_p_layer
        self.prev_steps = prev_steps  # 새로 추가된 이전 스텝(layer)의 수

        # Create LSTM layers
        self.rf_lstm_layers1 = [LSTM(rf_units, return_state=True, return_sequences=True) for _ in range(num_layers)]
        self.fc_lstm_layers1 = [LSTM(fc_units, return_state=True, return_sequences=True) for _ in range(num_layers)]

        # Dense layers
        self.dense_current = [Dense(num_p_layer * 8, activation='LeakyReLU') for _ in range(num_layers)]
        self.dense_prev    = [Dense(num_p_layer * 8, activation='tanh') for _ in range(num_layers)]
        self.dense_result  = [Dense(num_p_layer, activation='LeakyReLU') for _ in range(num_layers)]

        # Reshape the final output (Ensure total size matches)
        self.reshape_result = Reshape((v_step, v_num_y))

        # smoothing type
        self.smoothing_type = smoothing_type

        # Kalman Filter Variables: Q (Process variance), R (Measurement variance)
        self.process_variance = self.add_weight(shape=(),
                                                initializer=initializers.Constant(proc_v_init),
                                                # initializer="random_normal", 
                                                trainable=True, 
                                                name='process_variance', 
                                                constraint=tf.keras.constraints.NonNeg())  # Q: Process variance
        self.measurement_variance = self.add_weight(shape=(),
                                                    initializer=initializers.Constant(meas_v_init),
                                                    # initializer="random_normal", 
                                                    trainable=True, 
                                                    name='measurement_variance', 
                                                    constraint=tf.keras.constraints.NonNeg())  # R: Measurement variance
        # exponential smoothing weight
        self.alpha = self.add_weight( shape=(), 
                                      initializer=tf.keras.initializers.Constant(1.0),
                                      constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, axis=None),
                                      trainable=True, 
                                      name='smoothing_alpha' )

        # WMA 가중치 초기화
        self.window_size = moving_avg_win_size
        self.wma_weights = self.add_weight(
          shape=(self.window_size,),  # WMA 가중치의 개수는 window_size에 맞춰 설정
          initializer=tf.keras.initializers.Constant(np.arange(1, self.window_size + 1) / np.sum(np.arange(1, self.window_size + 1))),  # 초기화
          trainable=True,
          name='wma_weights')

    def call(self, inputs):
        input1, input2 = inputs
        
        outputs = []
        outputs_fc = []
        # fc_step = self.step // self.num_layers
        fc_step = self.fc_horizon // self.num_layers   # 예보 전체 72step 기준으로 layer별 분배

        # Initial states for LSTM
        h_rf, c_rf = None, None
        h_fc, c_fc = None, None
        prev_steps_buffer = []  # 이전 스텝들을 저장할 리스트
      
        # for i in range(self.num_layers):
        #     if i == 0:
        #         x1, h_rf, c_rf = self.rf_lstm_layers1[i](input1)
        #         x2, h_fc, c_fc = self.fc_lstm_layers1[i](input2[:, :fc_step, :])
        #     else:
        #         x1, h_rf, c_rf = self.rf_lstm_layers1[i](input1, initial_state=[h_rf, c_rf])
        #         x2, h_fc, c_fc = self.fc_lstm_layers1[i](input2[:, :fc_step * (i + 1), :], initial_state=[h_fc, c_fc])

         # 실제 예보 길이 (동적)
        T_f = tf.shape(input2)[1]
    
        for i in range(self.num_layers):
            # 관측 LSTM
            if i == 0:
                x1, h_rf, c_rf = self.rf_lstm_layers1[i](input1)
            else:
                x1, h_rf, c_rf = self.rf_lstm_layers1[i](
                    input1, initial_state=[h_rf, c_rf]
                )
    
            # 이 layer가 사용할 예보 길이 (72step 내에서 점점 늘어나도록)
            this_fc_len = tf.minimum(fc_step * (i + 1), T_f)
    
            if i == 0:
                x2, h_fc, c_fc = self.fc_lstm_layers1[i](
                    input2[:, :this_fc_len, :]
                )
            else:
                x2, h_fc, c_fc = self.fc_lstm_layers1[i](
                    input2[:, :this_fc_len, :],
                    initial_state=[h_fc, c_fc]
                )

            x = concatenate([x1[:,-1], x2[:,-1]], axis=1)
            # Dense layer
            x = self.dense_current[i](x)

            # 이전 스텝의 정보들을 가져오기
            if len(prev_steps_buffer) > 0:
                # 필요한 만큼의 이전 스텝을 concatenate
                prev_steps_to_use = concatenate(prev_steps_buffer, axis=1)
                prev_steps_to_use = self.dense_prev[i](prev_steps_to_use)
                prev_steps_buffer.append(x)
                x = concatenate([x, prev_steps_to_use], axis=1)
            else:
                prev_steps_buffer.append(x)

            x = self.dense_result[i](x)
            outputs.append(x)

            # prev_step 리스트에 현재 스텝 추가
            # 버퍼가 self.prev_steps 이상이면 오래된 값을 제거
            if len(prev_steps_buffer) > self.prev_steps:
                prev_steps_buffer.pop(0)
            
        # Concatenate all outputs
        result = concatenate(outputs, axis=1)
        
        # Modify reshape to ensure correct dimensions
        result = self.reshape_result(result)

        # 1. Moving Smoothing Start 
        # 이동평균으로 step간 Data 평활화 
        # 맨 앞의 이동평균이 없는 경우는 원래 Data 그대로 사용하도록 처리. 
        if self.smoothing_type == 1:
            normalized_weights = self.wma_weights / tf.reduce_sum(self.wma_weights)
            wma_results = []
            for t in range(self.window_size - 1, result.shape[1]):
                window = result[:, t - self.window_size + 1:t + 1, :]  # Current window
                wma = tf.reduce_sum(window * normalized_weights[:, tf.newaxis], axis=1)  # WMA 계산
                wma_results.append(wma)
    
            # WMA 결과를 tensor로 변환
            wma_results = tf.stack(wma_results, axis=1)
          
            # 원래 데이터와 WMA 결합 (앞의 window_size - 1 만큼은 원본 데이터 사용)
            result = tf.concat([result[:, :self.window_size - 1, :], wma_results], axis=1)

        # 2. Exponential Smoothing 
        elif smoothing_type == 2 :
            smoothed = []
            last = result[:, 0, :]
            smoothed.append(last)
      
            for t in range(1, result.shape[1]):
                last = self.alpha * result[:, t, :] + (1 - self.alpha) * last 
                smoothed.append(last)
      
            result = tf.stack(smoothed, axis=1)

        # 3. Kalman filtering ( fixed )
        elif smoothing_type == 3 :
            # Q: Process variance : 시스템 자체의 잡음 또는 모델이 설명하지 못하는 불확실성 표현 
            # range : 0 ~ 1 이상 
            # 안정된 시스템 or 데이터가 매우 매끄러운 경우 : 작은 값 (예: 1e-5에서 1e-2) 사용
            # 예측불헌 시스템 or 데이터가 급격한 변화가 잦은 경우 : 상대적으로 큰 값 (예: 1e-1에서 1) 사용
            # R: Measurement variance : 모델이 예측값을 얼마나 신뢰할지 조정하는 역할
            # Range : 1e-5 ~ 1e-1 
            # 관측값(이전 예측) 이 매우 정확 : 작은 값 (예: 1e-4 이하) 사용
            # 관측값(이전 예측) 에 잡음이 많이 포함된 경우 : 상대적으로 큰 값 (예: 1e-2에서 1e-1) 사용
            process_variance = 1e-4
            measurement_variance = 1e-3 
            estimated_variance = tf.ones_like(result[:, 0, :])
            kalman_gain = tf.zeros_like(result[:, 0, :])
            state_estimate = result[:, 0, :] 
    
            smoothed = [state_estimate]
    
            for t in range(1, result.shape[1]):
                # Predict step
                predicted_state = state_estimate 
                predicted_variance = estimated_variance + process_variance  
    
                # Update step
                kalman_gain = predicted_variance / (predicted_variance + measurement_variance) 
                state_estimate = predicted_state + kalman_gain * (result[:, t, :] - predicted_state)
                estimated_variance = (1 - kalman_gain) * predicted_variance 
    
                smoothed.append(state_estimate)
    
            result = tf.stack(smoothed, axis=1)
        # 4. Kalman filtering ( trainable )
        elif smoothing_type == 4 :
            estimated_variance = tf.ones_like(result[:, 0, :])  # P: Estimated error in prediction
            kalman_gain = tf.zeros_like(result[:, 0, :])  # K: Kalman gain
            state_estimate = result[:, 0, :]  # Initial state estimate
    
            smoothed = [state_estimate]
          
            # Iterate through the sequence and apply the Kalman Filter
            for t in range(1, result.shape[1]):
                # Predict step
                predicted_state = state_estimate  # A * state_estimate (A=1 in simple case)
                predicted_variance = estimated_variance + self.process_variance  # P = P + Q
    
                # Update step
                kalman_gain = predicted_variance / (predicted_variance + self.measurement_variance)  # K = P / (P + R)
                state_estimate = predicted_state + kalman_gain * (result[:, t, :] - predicted_state)  # x = x' + K * (z - x')
                estimated_variance = (1 - kalman_gain) * predicted_variance  # P = (1 - K) * P
    
                smoothed.append(state_estimate)
    
            # Stack the results back into a single tensor
            result = tf.stack(smoothed, axis=1)
          
        return result


def build_MultiLSTM(rf_units, fc_units, num_layers, num_p_layer, v_num_x, v_num_f, v_num_y, v_step, v_fc_step) :
    model = MultiLSTM_Model(rf_units, fc_units, num_layers, num_p_layer, v_num_x, v_num_f, v_num_y, v_step=v_step, v_fc_step=v_fc_step)
    
    input1 = Input(shape=(None, v_num_x), name='input_1')
    input2 = Input(shape=(None, v_num_f), name='input_2')
    output = model([input1, input2])
    model = Model(inputs=[input1, input2], outputs=output)
    return model
