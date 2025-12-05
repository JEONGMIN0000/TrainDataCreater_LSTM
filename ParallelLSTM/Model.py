# keras 필요 패키지 import
from keras import Model, layers
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import json
import keras
import pickle
from keras.layers import RepeatVector, TimeDistributed, Dense, Input, Concatenate, Lambda, LSTM, Flatten, ConvLSTM1D, BatchNormalization, Dropout, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D, MaxPooling2D, Reshape
from keras.models import Model, Sequential, load_model
from tensorflow.keras import initializers
from tensorflow.keras.layers import Attention, LeakyReLU, ReLU
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

# 모델 학습중 성능 향상이 진행되지 않는 경우 중단 여부 설정
# 모델 학습중 model 저장 여부 설정
# 저장된 모델을 Google Drive 에 저장 여부
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ModelCheckpoint, ReduceLROnPlateau, CSVLogger



# 모델 생성

# v_num_y : 예측 feature 수 
# v_step : 예측 시간 (step)
# num_layers : layer 수
# num_p_layer : layer 당 예측수 
class MultiLSTM_Model(Model):
    def __init__(self, rf_units, fc_units, num_layers, num_p_layer, v_num_x, v_num_f, v_num_y, v_step, prev_steps=feedback_win_size):
        super().__init__()
        self.input1 = Input(shape=(None, v_num_x), name='input_rainflow')
        self.input2 = Input(shape=(None, v_num_f), name='input_forecast')

        self.rf_units = rf_units
        self.fc_units = fc_units
        self.step = v_step
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
        fc_step = self.step // self.num_layers

        # Initial states for LSTM
        h_rf, c_rf = None, None
        h_fc, c_fc = None, None
        prev_steps_buffer = []  # 이전 스텝들을 저장할 리스트
    
        for i in range(self.num_layers):
            if i == 0:
                x1, h_rf, c_rf = self.rf_lstm_layers1[i](input1)
                x2, h_fc, c_fc = self.fc_lstm_layers1[i](input2[:, :fc_step, :])
            else:
                x1, h_rf, c_rf = self.rf_lstm_layers1[i](input1, initial_state=[h_rf, c_rf])
                x2, h_fc, c_fc = self.fc_lstm_layers1[i](input2[:, :fc_step * (i + 1), :], initial_state=[h_fc, c_fc])

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


def build_MultiLSTM(rf_units, fc_units, num_layers, num_p_layer, v_num_x, v_num_f, v_num_y, v_step) :
    model = MultiLSTM_Model(rf_units, fc_units, num_layers, num_p_layer, v_num_x, v_num_f, v_num_y, v_step)
    
    input1 = Input(shape=(None, v_num_x), name='input_1')
    input2 = Input(shape=(None, v_num_f), name='input_2')
    output = model([input1, input2])
    model = Model(inputs=[input1, input2], outputs=output)
    return model

#%%script echo Self Attention Model Skip....
print("Self Attention Model Skip....")

class MultiLSTM_Attention_Model(Model):
    def __init__(self, rf_units, fc_units, num_layers, num_p_layer, v_num_x, v_num_f, v_num_y, v_step, prev_steps=feedback_win_size):
        super().__init__()
        self.input1 = Input(shape=(None, v_num_x), name='input_rainflow')
        self.input2 = Input(shape=(None, v_num_f), name='input_forecast')

        self.rf_units = rf_units
        self.fc_units = fc_units
        self.step = v_step
        self.num_layers = num_layers
        self.num_p_layer = num_p_layer
        self.v_num_y = v_num_y

        # LSTM layers
        self.rf_lstm_layers1 = [LSTM(rf_units, return_state=True, return_sequences=True) for _ in range(num_layers)]
        self.fc_lstm_layers1 = [LSTM(fc_units, return_state=True, return_sequences=True) for _ in range(num_layers)]
        
        # Attention layers
        self.rf_attn_layers = [Attention() for _ in range(num_layers)]
        self.fc_attn_layers = [Attention() for _ in range(num_layers)]
        
        # Dense layers ( in self attention ) 
        self.rf_dense      = [Dense(num_p_layer * 16, activation='tanh') for _ in range(num_layers)]
        self.fc_dense      = [Dense(num_p_layer * 16, activation='tanh') for _ in range(num_layers)]
        self.concat_dense  = [Dense(num_p_layer * 8, activation='LeakyReLU') for _ in range(num_layers)]

        # For Previous Step Sub-Result Connections
        self.dense_prev    = [Dense(num_p_layer * 8, activation='tanh') for _ in range(num_layers)]

        # final result Dense
        self.result_dense  = [Dense(num_p_layer, activation=activation_func) for _ in range(num_layers)]

        # Reshape layer
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

        # Initial states for LSTM
        h_rf, c_rf = None, None
        h_fc, c_fc = None, None
        prev_steps_buffer = []  # 이전 스텝들을 저장할 리스트
    
        fc_step = self.step // self.num_layers
        for i in range(self.num_layers):
            if i == 0:
                x1, h_rf, c_rf = self.rf_lstm_layers1[i](input1)
                x2, h_fc, c_fc = self.fc_lstm_layers1[i](input2[:, :fc_step, :])
            else:
                x1, h_rf, c_rf = self.rf_lstm_layers1[i](input1, initial_state=[h_rf, c_rf])
                x2, h_fc, c_fc = self.fc_lstm_layers1[i](input2[:, :fc_step * (i + 1), :], initial_state=[h_fc, c_fc])

            # Apply Attention between LSTM outputs (rainflow and forecast)
            rf_attn_out = self.rf_attn_layers[i]([x1, x1])
            rf_attn_out = Lambda(lambda x: K.mean(x, axis=1))(rf_attn_out)            

            fc_attn_out = self.fc_attn_layers[i]([x2, x2])
            fc_attn_out = Lambda(lambda x: K.mean(x, axis=1))(fc_attn_out)

            # Concatenate LSTM and Attention outputs
            x1 = concatenate([x1[:,-1], rf_attn_out], axis=1)
            x2 = concatenate([x2[:,-1], fc_attn_out], axis=1)
            
            x1 = self.rf_dense[i](x1)
            x2 = self.fc_dense[i](x2)

            # Apply Dense layers
            x = concatenate([x1, x2], axis=1)
            x = self.concat_dense[i](x)

            # 이전 스텝의 정보들을 가져오기
            if len(prev_steps_buffer) > 0:
                # 필요한 만큼의 이전 스텝을 concatenate
                prev_steps_to_use = concatenate(prev_steps_buffer, axis=1)
                prev_steps_to_use = self.dense_prev[i](prev_steps_to_use)
                prev_steps_buffer.append(x)
                x = concatenate([x, prev_steps_to_use], axis=1)
            else:
                prev_steps_buffer.append(x)

            x = self.result_dense[i](x)
            outputs.append(x)

        # Concatenate the outputs of all layers
        result = concatenate(outputs, axis=1)
        result = self.reshape_result(result)

        if self.smoothing_type == 1:
            # 1. start Moving Smoothing Start 
            # 가중이동평균으로 step간 Data 평활화 ( Traininig )
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
        elif smoothing_type == 2 :
        # Exponetial Smoothing으로 step간 Data 평활화 ( Traininig )
            # 2. start Exponential Smoothing 
            smoothed = []
            last = result[:, 0, :]
            smoothed.append(last)
    
            for t in range(1, result.shape[1]):
                last = self.alpha * result[:, t, :] + (1 - self.alpha) * last 
                smoothed.append(last)
    
            result = tf.stack(smoothed, axis=1)
        elif smoothing_type == 3 :
            # 3. Start Kalman filtering
            # Q: Process variance : 시스템 자체의 잡음 또는 모델이 설명하지 못하는 불확실성 표현 
            # range : 0 ~ 1 이상 
            # 안정된 시스템 or 데이터가 매우 매끄러운 경우 : 작은 값 (예: 1e-5에서 1e-2) 사용
            # 예측불헌 시스템 or 데이터가 급격한 변화가 잦은 경우 : 상대적으로 큰 값 (예: 1e-1에서 1) 사용
            process_variance = proc_v_init
            # R: Measurement variance : 모델이 예측값을 얼마나 신뢰할지 조정하는 역할
            # Range : 1e-5 ~ 1e-1 
            # 관측값(이전 예측) 이 매우 정확 : 작은 값 (예: 1e-4 이하) 사용
            # 관측값(이전 예측) 에 잡음이 많이 포함된 경우 : 상대적으로 큰 값 (예: 1e-2에서 1e-1) 사용
            measurement_variance = meas_v_init
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
    
            # Stack the results back into a single tensor
            result = tf.stack(smoothed, axis=1)
        elif smoothing_type == 4 :
            # Kalman Filter Application
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


def build_MultiLSTM_Attention(rf_units, fc_units, num_layers, num_p_layer, v_num_x, v_num_f, v_num_y, v_step) :
    model = MultiLSTM_Attention_Model(rf_units, fc_units, num_layers, num_p_layer, v_num_x, v_num_f, v_num_y, v_step)
    
    input1 = Input(shape=(None, v_num_x), name='input_1')
    input2 = Input(shape=(None, v_num_f), name='input_2')
    output = model([input1, input2])
    model = Model(inputs=[input1, input2], outputs=output)
    return model

def build_LSTM(rf_units, fc_units, v_num_x, v_num_f, v_num_y, v_step, v_layer, v_dropout=0.0) :

    rf_cells  = rf_units
    fc_cells  = fc_units
    out_feature  = v_step*v_num_y

    in_rainflow = Input(shape=(None, v_num_x), name='input_rainflow')
    in_forecast = Input(shape=(None, v_num_f), name='input_forecast')

    
    stepN = v_step//v_layer
    stepL = v_step - stepN*(v_layer-1)
    
    # 첫 행부터 stepN행까지 (shape: (n, stepN, n))
    part_fc_list = []
    for i in range(1,v_layer+1):
        if i < v_layer:
            part_fc = Lambda(lambda x: x[:, :stepN*i, :])(in_forecast)
        else:
            part_fc = Lambda(lambda x: x[:, :       , :])(in_forecast)

        part_fc_list.append(part_fc)

    part_result_list = []
    for i in range(v_layer):
        rainflow = LSTM(rf_cells, dropout=v_dropout)(in_rainflow)
        forecast = LSTM(fc_cells, dropout=v_dropout)(part_fc_list[i])
        d = Concatenate()([rainflow, forecast])

        # d = Dense(rf_cells//4, activation='relu')(d)
        # d = Dense(rf_cells, activation='linear')(d)
        # if i < v_layer-1:
        #     part_result = Dense(v_num_y*stepN, activation='relu')(d)
        # else:
        #     part_result = Dense(v_num_y*stepL, activation='relu')(d)

        d = Dense(v_num_y*stepN*16, activation='LeakyReLU')(d)
        # d = Dense(v_num_y*stepN*4, activation='relu')(d)
        d = Dense(v_num_y*stepN, activation='LeakyReLU')(d)
        part_result_list.append(d)
    
    result = Concatenate()([p_result for p_result in part_result_list])
    result = Dense(v_step*v_num_y*16, activation='LeakyReLU')(result)
    # result = Dense(v_step*v_num_y*4, activation='relu')(result)
    result = Dense(v_step*v_num_y, activation='LeakyReLU')(result)
    result = Reshape((v_step, v_num_y))(result)  
    
    model = Model(inputs=[in_rainflow, in_forecast], outputs=[result])

    return model

def build_LSTM4(rf_units, fc_units, v_num_x, v_num_f, v_num_y, v_step, v_dropout=0.0) :

    rf_cells  = rf_units
    fc_cells  = fc_units
    out_feature  = v_step*v_num_y

    in_rainflow = Input(shape=(None, v_num_x), name='input_rainflow')
    in_forecast = Input(shape=(None, v_num_f), name='input_forecast')

    stepN = v_step//5
    
    stepL = v_step - stepN*4
    
    # 첫 행부터 stepN행까지 (shape: (n, stepN, n))
    in_forecast1 = Lambda(lambda x: x[:, :stepN  , :])(in_forecast)
    in_forecast2 = Lambda(lambda x: x[:, :stepN*2, :])(in_forecast)
    in_forecast3 = Lambda(lambda x: x[:, :stepN*3, :])(in_forecast)
    in_forecast4 = Lambda(lambda x: x[:, :stepN*4, :])(in_forecast)
    in_forecast5 = Lambda(lambda x: x[:, :       , :])(in_forecast)

    rainflow1 = LSTM(rf_cells, dropout=v_dropout)(in_rainflow)
    forecast1 = LSTM(fc_cells, dropout=v_dropout)(in_forecast1)
    d1 = Concatenate()([rainflow1, forecast1])
    d1 = Dense(out_feature*8, activation='relu')(d1)
    d1 = Dense(out_feature*2, activation='relu')(d1)
    result1 = Dense(v_num_y*stepN)(d1)
    
    rainflow2 = LSTM(rf_cells, dropout=v_dropout)(in_rainflow)
    forecast2 = LSTM(fc_cells, dropout=v_dropout)(in_forecast2)
    d2 = Concatenate()([rainflow2, forecast2])
    d2 = Dense(out_feature*8, activation='relu')(d2)
    d2 = Dense(out_feature*2, activation='relu')(d2)
    result2 = Dense(v_num_y*stepN)(d2)

    rainflow3 = LSTM(rf_cells, dropout=v_dropout)(in_rainflow)
    forecast3 = LSTM(fc_cells, dropout=v_dropout)(in_forecast3)
    d3 = Concatenate()([rainflow3, forecast3])
    d3 = Dense(out_feature*8, activation='relu')(d3)
    d3 = Dense(out_feature*2, activation='relu')(d3)
    result3 = Dense(v_num_y*stepN)(d3)

    rainflow4 = LSTM(rf_cells, dropout=v_dropout)(in_rainflow)
    forecast4 = LSTM(fc_cells, dropout=v_dropout)(in_forecast4)
    d4 = Concatenate()([rainflow4, forecast4])
    d4 = Dense(out_feature*8, activation='relu')(d4)
    d4 = Dense(out_feature*2, activation='relu')(d4)
    result4 = Dense(v_num_y*stepN)(d4)

    rainflow5 = LSTM(rf_cells, dropout=v_dropout)(in_rainflow)
    forecast5 = LSTM(fc_cells, dropout=v_dropout)(in_forecast5)
    d5 = Concatenate()([rainflow5, forecast5])
    d5 = Dense(out_feature*8, activation='relu')(d5)
    d5 = Dense(out_feature*2, activation='relu')(d5)
    result5 = Dense(v_num_y*stepL)(d5)

    result = Concatenate()([result1, result2, result3, result4, result5])
    result = Reshape((v_step, v_num_y))(result)  
    
    model = Model(inputs=[in_rainflow, in_forecast], outputs=[result])

    return model

def build_LSTM3(rf_units, fc_units, v_num_x, v_num_f, v_num_y, v_step, v_dropout=0.0) :

    rf_cells  = rf_units
    fc_cells  = fc_units
    out_feature  = v_step*v_num_y

    in_rainflow = Input(shape=(None, v_num_x), name='input_rainflow')
    in_forecast = Input(shape=(None, v_num_f), name='input_forecast')

    rainflow1 = LSTM(rf_cells, dropout=v_dropout)(in_rainflow)
    forecast1 = LSTM(fc_cells, dropout=v_dropout)(in_forecast)
    d1 = Concatenate()([rainflow1, forecast1])
    d1 = Dense(out_feature*8, activation='relu')(d1)
    d1 = Dense(out_feature*2, activation='relu')(d1)
    result1 = Dense(out_feature)(d1)

    rainflow2 = LSTM(rf_cells, dropout=v_dropout)(in_rainflow)
    forecast2 = LSTM(fc_cells, dropout=v_dropout)(in_forecast)
    d2 = Concatenate()([rainflow2, forecast2])
    d2 = Dense(out_feature*8, activation='relu')(d2)
    d2 = Dense(out_feature*2, activation='relu')(d2)
    result2 = Dense(out_feature)(d2)

    result = Concatenate()([result1, result2])
    result = Dense(out_feature)(result)
    result = Reshape((v_step, v_num_y))(result)  
    
    model = Model(inputs=[in_rainflow, in_forecast], outputs=[result])

    return model

def build_LSTM2(rf_units, fc_units, v_num_x, v_num_f, v_num_y, v_step, v_dropout=0.0) :

    rf_cells  = rf_units
    fc_cells  = fc_units
    out_feature  = v_step*v_num_y

    rainflow = Input(shape=(None, v_num_x), name='input_rainflow')
    forecast = Input(shape=(None, v_num_f), name='input_forecast')
    # forecast = Input(shape=(v_step, v_num_f), name='input_forecast')

    # activation default : tanh 
    # None -> linear 
    rainflow1 = LSTM(rf_cells, dropout=v_dropout)(rainflow)
    forecast1 = LSTM(fc_cells, dropout=v_dropout)(forecast)

    # forecast1 = Flatten()(forecast)
    # forecast1 = Dense(fc_cells, activation='relu')(forecast1)

    d1 = Concatenate()([rainflow1, forecast1])
    # d1 = Dense(rf_cells//4, activation='relu')(d1)
    # d1 = Dense(rf_cells//16, activation='relu')(d1)

    d1 = Dense(out_feature*8, activation='relu')(d1)
    d1 = Dense(out_feature*2, activation='relu')(d1)

    result = Dense(out_feature)(d1)
    result = Reshape((v_step, v_num_y))(result)  

    model = Model(inputs=[rainflow, forecast], outputs=[result])

    return model

def build_LSTM4Multi(num_of_units, v_num_x, v_num_f, v_num_y, v_step, v_dropout=0.0) :

    rf_cells  = num_of_units
    fc_cells  = num_of_units//4

    rainflow = Input(shape=(None, v_num_x), name='input_rainflow')
    forecast = Input(shape=(None, v_num_f), name='input_forecast')

    # activation default : tanh 
    # None -> linear 
    rainflow1 = LSTM(rf_cells, input_shape=(None, v_num_x), dropout=v_dropout, activation='relu')(rainflow)
    forecast1 = LSTM(fc_cells, input_shape=(None, v_num_f), dropout=v_dropout, activation='relu')(forecast)
    # forecast1 = Dense(fc_cells, activation='relu')(forecast)
    d1 = Concatenate()([rainflow1, forecast1])
    result1 = Dense(1)(d1)

    rainflow2 = LSTM(rf_cells, input_shape=(None, v_num_x), dropout=v_dropout, activation='relu')(rainflow)
    forecast2 = LSTM(fc_cells, input_shape=(None, v_num_f), dropout=v_dropout, activation='relu')(forecast)
    # forecast2 = Dense(fc_cells, activation='relu')(forecast)
    d2 = Concatenate()([rainflow2, forecast2])
    result2 = Dense(1)(d2)

    rainflow3 = LSTM(rf_cells, input_shape=(None, v_num_x), dropout=v_dropout, activation='relu')(rainflow)
    forecast3 = LSTM(fc_cells, input_shape=(None, v_num_f), dropout=v_dropout, activation='relu')(forecast)
    # forecast3 = Dense(fc_cells, activation='relu')(forecast)
    d3 = Concatenate()([rainflow3, forecast3])
    result3 = Dense(v_num_y)(d3)
    
    result = Concatenate(axis=1)([result1, result2, result3]) 
    model = Model(inputs=[rainflow, forecast], outputs=[result])

    return model

# 모델을 반복 호출하는 모델
class MultiStepModel(keras.Model):
    def __init__(self, single_model, hours=3):
        super().__init__()
        self.single_model = single_model
        self.hours = hours
        
    def call(self, inputs) :
        # base_feature = inputs['Base_Feature']
        # fc_feature = inputs['Forecast_Feature']
        base_feature, fc_feature = inputs

        predictions = []
        single_data = base_feature

        # single모델을 재귀적으로 호출
        for i in range(self.hours):
            # 예보 데이터중 1시간 부분만 추출
            now_fc = fc_feature[:,i,:]
            now_fc = tf.reshape(now_fc, (-1, fc_feature.shape[-1]))
            # 1시간 예측
            output = self.single_model([single_data, now_fc])
            
            # 예측 결과를 리스트에 저장
            predictions.append(output[:,-1])
            
            # 예측 결과를 예보 데이터와 결합후 입력 데이터로 통합 
            next_feature = tf.concat([now_fc, output], axis=1) 
            next_feature = tf.reshape(next_feature, (-1,1,next_feature.shape[1]))
            single_data = tf.concat([single_data[:,1:,:], next_feature], axis=1)
        
        # 리스트로부터 예측 결과 반환
        result = tf.stack(predictions, axis=1)
        # result = result[:,:,-1]
        
        # result = result.numpy() 
        # result = result.reshape(-1,result.shape[1],1)
        # result = tf.reshape(result, (-1,result.shape[1],1))

        return result
    
# Multi Step Model 

def build_MultiStepModel(base_model_path, pred_hours) :
    
    print('[ Multi-Step Model ]')
    print('- Base Model Path : ', base_model_path)
    base_model = keras.models.load_model(base_model_path)

    # 모델 인스턴스 생성
    mstep_model = MultiStepModel(base_model, hours=pred_hours)
    mstep_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
    mstep_model.build(input_shape=[(None, config['seq_length'], config['num_of_feature']), (None, pred_hours, config['forecast'])])

    return mstep_model 

# use_model에 따라 모델 생성
# Convolution Layer 사용시 사용할 subsequence (영상에서 frame에 해당) 기본 길이 ( 모델에 따라 다름 )
def build_model(model_type, rf_cells, fc_cells, num_of_predict, v_layer) :
    if model_type == 'LSTM' :
        v_model = build_LSTM(rf_cells, fc_cells, num_x, num_f, num_y, num_of_predict, v_layer, v_dropout=0.0)

    if model_type == 'MultiLSTM' :
        v_model = build_MultiLSTM(rf_cells, fc_cells,  v_layer, num_p_layer, num_x, num_f, num_y, num_of_predict)

    return v_model

# Model 생성
if config['use_model'] == 'MultiStepModel':
    model = build_MultiStepModel(config['base_model_path'], config['predict'])
else :
    model = build_model(config['use_model'], config['num_of_cells'][0], config['num_of_cells'][1], config['predict'], config['num_of_layer'])

# Model 및 Hyper Parameter, 각종 configration 설정 내용 확인 
print(f"\n[ {config['use_model']} Model / Hper Parameter / Configuration ]\n")
model.summary()
print('')
print_config(config) 


# 모델 학습 설정

# callback 함수 정의
callback_list = []

# loss 또는 val_loss의 개선이 patience_count회 연속 없을 경우 학습을 중단 
if config['isEarlystop'] :
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=config['patience_count'], restore_best_weights=True, mode='auto')
    callback_list.append(earlystop)


# 학습 단계마다 학습 결과(모델 데이터)를 저장할지 여부
if config['isCheckpoint'] :
    # 체크포인트 활성시 모델을 저장할 폴더명
    chkpt_dir = 'model_checkpoint' 

    # 각 epochs마다 저장될 파일명 ( 개별 Epoch 별로 지정하기 위해서는 epoch 변수 사용  ) 
    chkpt_filename = config['name'] + '_' + config['use_model'] + '_S' + str(config['seq_length']) + '_best'

    chkpt_filename = path.join(chkpt_dir, chkpt_filename)

    print('- Model will be saved in [', chkpt_filename, ']')
    checkpoint = ModelCheckpoint(chkpt_filename,      # file명
                                monitor='val_loss',  # loss/val_loss 값이 개선되었을때만 저장
                                verbose=1,           # 로그 출력 레벨
                                save_best_only=True, # 이전에 저장한 모델보다 향상시 저장 여부
                                mode='auto'          # auto/min/max  loss, acc 값의 해석
                                )
    callback_list.append(checkpoint)

class PrintParamterCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\r---------- Epoch {epoch+1} -------------                                                  ")
        # Weighted Moving Average
        wma_weights = [var for var in model.trainable_variables if 'wma_weights' in var.name][0].numpy()
        normalized_wma_weights = wma_weights / np.sum(wma_weights)
        print("Weighted MA weights  :", ', '.join(f"{weight:.5f}" for weight in normalized_wma_weights))

        # Exponential Smoothing
        # smoothing_alpha_value = [var for var in self.model.trainable_variables if 'smoothing_alpha' in var.name][0].numpy()
        # print(f"Exponential Smoothing Alpha  : {smoothing_alpha_value:.5f}")

        # Kalman Filter
        # process_variance_value = [var for var in model.trainable_variables if 'process_variance' in var.name][0].numpy()
        # print(f"process_variance     : {process_variance_value:.5f}")
        # measurement_variance_value = [var for var in model.trainable_variables if 'measurement_variance' in var.name][0].numpy()
        # print(f"measurement_variance : {measurement_variance_value:.5f}")
        logs = logs or {}
        for metric, value in logs.items():
            print(f"{metric}: {value:.8f}", end='\t')  # 소수점 이하 8자리까지 출력
        print("\n-----------------------------------------")

# callback_list.append( PrintParamterCallback() )


# TensorBorad 실행 방법 ( 모델 학습 중-모니터링 또는 실행후 실행-검토 ) 
# Shell 실행 방법
# ~$ tensorboard --bind_all --logdir ./tensor_board_log
# Notebook 실행 방법
# %load_ext tensorboard
# %tensorboard --bind_all --logdir ./tensor_board_log 

isTensor_board = False
if isTensor_board == True :
    tb_callback = TensorBoard(log_dir='./tensor_board_log')
    callback_list.append(tb_callback)

# 개선이 없을 경우 Learning rate 축소
# Test 결과 adam의 learning rate의 관리에 부정적 영향을 주어 좋지 않은 결과가 나옴.
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
# callback_list.append(reduce_lr)

# Train 과정을 log에 남기는 callback
now = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('./logs', exist_ok=True)
csv_logger = CSVLogger(f"./logs/training_log_{now}_{config['use_model']}.csv")
callback_list.append(csv_logger)

print('\n< Added Callback List >') 
display(callback_list)

# 학습 수행 준비 (shape 등)
# Convolution층이 있는 모델은 Data의 shape를 변경

printTSDataSetShape(dataSets_ts)

def save_dictionary(filename, results_dict):
    # 결과가 리스트인 경우 문자열로 변환
    for key, value in results_dict.items():
        if isinstance(value, list):
            results_dict[key] = json.dumps(value)  # 리스트를 JSON 문자열로 변환

    # 파일이 존재하는지 확인
    # 파일이 존재하는지 확인
    if os.path.exists(filename):
        # 파일이 존재하면 새로운 데이터 추가
        new_data = pd.DataFrame([results_dict])
        new_data.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig')  # append 모드로 저장
    else:
        # 파일이 존재하지 않으면 새로운 파일 생성
        new_data = pd.DataFrame([results_dict])
        new_data.to_csv(filename, index=False, encoding='utf-8-sig')

def load_dictinary(filename):
    if os.path.exists(filename):
        data = pd.read_csv(filename)
        # 리스트 형태로 변환
        for column in data.columns:
            data[column] = data[column].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x)
        return data
    else:
        print(f"{filename} 파일이 존재하지 않습니다.")
        return None

# 모델 실행 결과    
model_result = {
    "train_sdate"     : '',
    "train_edate"     : '',
    "chkpt_file"      : '',
    "best_epochs"     : 0,
    "train_loss"      : 0.0,
    "valid_loss"      : 0.0,
    "test_loss"       : 0.0,
    "train_rows"      : 0,
    "valid_rows"      : 0,
    "test_rows"       : 0,
    "train_rmse"      : 0.0,
    "valid_rmse"      : 0.0,
    "test_rmse"       : 0.0,
    "train_mae"       : 0.0,
    "valid_mae"       : 0.0,
    "test_mae"        : 0.0,
    "train_mape"      : 0.0,
    "valid_mape"      : 0.0,
    "test_mape"       : 0.0,
    "train_r2"        : 0.0,
    "valid_r2"        : 0.0,
    "test_r2"         : 0.0,
    "train_cr2"       : 0.0,
    "valid_cr2"       : 0.0,
    "test_cr2"        : 0.0,
    "train_qer"       : 0.0,
    "valid_qer"       : 0.0,
    "test_qer"        : 0.0,
    "train_rmse_h"    : [0.0],
    "valid_rmse_h"    : [0.0],
    "test_rmse_h"     : [0.0],
    "train_mae_h"     : [0.0],
    "valid_mae_h"     : [0.0],
    "test_mae_h"      : [0.0],
    "train_mape_h"    : [0.0],
    "valid_mape_h"    : [0.0],
    "test_mape_h"     : [0.0],
    "train_r2_h"      : [0.0],
    "valid_r2_h"      : [0.0],
    "test_r2_h"       : [0.0],
    "train_cr2_h"     : [0.0],
    "valid_cr2_h"     : [0.0],
    "test_cr2_h"      : [0.0],
    "train_qer_h"     : [0.0],
    "valid_qer_h"     : [0.0],
    "test_qer_h"      : [0.0]
}

# save_result(model_result, 'model_result_tmp.csv')
# 결과 읽기

# loaded_result = load_result('model_result_tmp.csv')
# loaded_result.T

# 모델 Compile (Loss/Opti 지정)

# Optimizer를 Adam으로 설정
#
# 보통 Learning Rate를 제외하고 default 값을 사용
# lr: 0보다 크거나 같은 float 값. 학습률.
# beta_1: 0보다 크고 1보다 작은 float 값. 일반적으로 1에 가깝게 설정
# beta_2: 0보다 크고 1보다 작은 float 값. 일반적으로 1에 가깝게 설정
# epsilon: 0보다 크거나 같은 float형 fuzz factor. None인 경우 K.epsilon()이 사용
# decay: 0보다 크거나 같은 float 값. 업데이트마다 적용되는 학습률의 감소율
# amsgrad: 불리언. Adam의 변형인 AMSGrad의 적용 여부를 설정

# 주의 : epsilon/decay/amsgrad 지정시 오류 발생하는 경우가 있음. 오류 발생시 learning_rate 값만 지정하여 사용
# default : Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
#adam = Adam(learning_rate=config['learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# adam = Adam(learning_rate=config['learning_rate)
adam = Adam(learning_rate=0.01)

# 'mean_squared_error'
# 'mean_absolute_error'
# 'huber_loss'
# 'mean_squared_logarithmic_error'
# 'mean_absolute_percentage_error'
# WeightedMSE(v_std=0.1)


# 모델 학습 설정
model.compile(loss=loss_func, optimizer=adam)
#model.compile(loss=WeightedMSE(v_std=1.5), optimizer=adam)
#model.compile(loss=HourWeightedMSE(w_list=[1, 1.2, 1.5]), optimizer=adam)

# Test Data Set 저장 (필요시)
# Data Set 저장 및 로드 관련 함수
def save_dataset(v_dataset_list, filepath, description) : 
    cnt_list = len(v_dataset_list)
    with open(filepath, 'wb') as file:
        pickle.dump(description, file)
        pickle.dump(cnt_list, file)
        for v_list in v_dataset_list:
            pickle.dump(v_list, file)

    print(f'[{cnt_list} count] Data Sets List are saved in {filepath}')

def load_dataset(datafile):
    data_list = []
    with open(datafile, 'rb') as file:
        description = pickle.load(file)
        print(description)
        cnt_list = pickle.load(file)
        for i in range(cnt_list):
            load_data = pickle.load(file)
            print(f'{i+1} data({load_data.__class__}) has loaded.')
            data_list.append(load_data)

    return data_list

# is_save_data 인 경우 Test Data Set 저장 

if is_save_data :
    # 파일 저장 관련 정보 
    os.makedirs('./model_stage', exist_ok=True)
    datafile = './model_stage/data_' + config['name'] + '.pkl'
    dataset_list = [dataSets[2],dataSets_org_ts[2],dataSets_ts[2]]
    data_desc = """
    - data : 성남 data
        Type : Test Data Set
        count : 3
        order : dataset(df list), dataSets_org_ts(ndarray list), dataSets_ts(ndarray list)
    """
    save_dataset(dataset_list, datafile, data_desc)

    # File을 load 하여 이상 유무 check 
    print('\nChecking saved datafile...')
    loaded_data = load_dataset(datafile)


# 학습 수행
# 학습 및 예측 등 모델의 각종 결과를 저장하기 위한 객체 생성
model_result['train_rows'] = dataSets_ts[0][0].shape[0]
model_result['valid_rows'] = dataSets_ts[1][0].shape[0]
model_result['test_rows']  = dataSets_ts[2][0].shape[0]

# 모델 학습 실행
# 데이터 일부에 NaN 이 있는 경우 전체 loss/val_lod 가 NaN 이 됨.
# epochs : 학습 수행 횟수

if config['use_model'] == 'MultiStepModel' :
    isSkip = True
else :
    isSkip = False

# sample weight 함수
# Y 의 값이 threshhold 이상인 경우 샘플 가중치를 높여 좀더 집중하여 처리 

# v_TrainY : 학습 데이터의 Y
# v_threshhold : 가중치를 높이는 기준
# v_weight : 지정할 가중치값
def make_sample_weight(v_TrainY, v_threshold, v_weight, v_min_cnt=0):
    original_shape = v_TrainY.shape
    v_TrainY_reshaped = v_TrainY.reshape(original_shape[0], -1)
    
    # 기준을 초과하는 data 선정 (1건이라도 넘는 경우)
    exceed_count = np.sum(v_TrainY_reshaped > v_threshold, axis=1)
    focus_data = exceed_count > v_min_cnt
    # max_exceed = np.max(exceed_count)  # 최대 초과 개수

    # 기본값 1을 가진 weight array 생성
    data_weights = np.ones(original_shape[0])
    # 기준에 부합하는 sample에 대해 weight 값을 지정
    data_weights[focus_data] = v_weight

    nofocus_data = ~focus_data

    print(f'- Train Data : {original_shape[0]}')
    print(f'- Weighted Data : {np.sum(focus_data)}')
    print(f'- No weighted Data : {np.sum(nofocus_data)}')
    print(f'- Exceed Count\n{exceed_count}')
    
    data_weights /= np.mean(data_weights)
    
    return data_weights

def shuffle_dataset(x, f, y, i):
    index = np.arange(x.shape[0]) 
    np.random.shuffle(index)  

    # 인덱스를 기준으로 trainX, trainF, trainY 셔플
    return x[index], f[index], y[index], i[index]

%%skip isSkip skip... : MultiStepModel은 학습하지 않음.
# 학습 수행 ( MultiStepModel의 경우 학습하지 않음. )

print('[ Training is starting. ] ') 
train_start = datetime.now()
model_result['train_sdate'] = train_start.strftime('%Y%m%d_%H%M%S')

trainX = dataSets_ts[0][0]
trainF = dataSets_ts[0][1]
trainY = dataSets_ts[0][2]
trainI = dataSets_ts[0][3]

trainset_shuffle = True
if trainset_shuffle :
    trainX, trainF, trainY, trainI = shuffle_dataset(trainX, trainF, trainY, trainI)

validX = dataSets_ts[1][0]
validF = dataSets_ts[1][1]
validY = dataSets_ts[1][2]
validI = dataSets_ts[1][3]

testX = dataSets_ts[2][0]
testF = dataSets_ts[2][1]
testY = dataSets_ts[2][2]
testI = dataSets_ts[2][3]

if use_sample_weight :
    # make_sample_weight ( data_set, threshhold, weight, min_count=0 )
    sample_weights = make_sample_weight(trainY, 0.75, 3.0)
    # print(f'Sample Weight\n{sample_weights}')
    train_hist = model.fit([trainX, trainF], [trainY], epochs=config['num_of_epochs'], batch_size=config['num_of_batch'], 
                            validation_data=([validX, validF] , [validY] ), callbacks=callback_list, sample_weight=sample_weights )
else:
    train_hist = model.fit([trainX, trainF], [trainY], epochs=config['num_of_epochs'], batch_size=config['num_of_batch'], 
                        validation_data=([validX, validF] , [validY] ), callbacks=callback_list)

train_end = datetime.now()
model_result['train_edate'] = train_end.strftime('%Y%m%d_%H%M%S')
print('[ Training is finished. ] ') 

model_result['best_epochs'] = np.argmin(train_hist.history['val_loss'])+1
print('best_epochs :', np.argmin(train_hist.history['val_loss'])+1 ) 

print('Training Time :', getDurationStr(train_start, train_end))

#%%script echo specific layer training
print("specific layer training")

def set_layers_trainable(model, target_layer_names):
    # Helper function to recursively set trainable layers
    def traverse_and_set_trainable(layer, trainable_layers):
        if layer.name in trainable_layers:
            return
        trainable_layers.add(layer.name)
        if hasattr(layer, '_inbound_nodes'):
            for node in layer._inbound_nodes:
                if hasattr(node, 'inbound_layers'):
                    inbound_layers = node.inbound_layers
                    if not isinstance(inbound_layers, list):
                        inbound_layers = [inbound_layers]
                    for inbound_layer in inbound_layers:
                        traverse_and_set_trainable(inbound_layer, trainable_layers)

    # Set of layers that should be trainable
    trainable_layers = set()

    # Traverse the model to find all layers connected to each target layer
    for target_layer_name in target_layer_names:
        target_layer = model.get_layer(target_layer_name)
        traverse_and_set_trainable(target_layer, trainable_layers)

    # Set layers' trainable property based on their presence in the trainable_layers set
    for layer in model.layers:
        if layer.name in trainable_layers:
            layer.trainable = True
        else:
            layer.trainable = False



set_layers_trainable(model, ['dense_17', 'dense_18'])

%%skip isSkip skip... : MultiStepModel은 학습하지 않음.

# 학습 과정 Plotting
plt.figure(figsize=(25,8))
plt.plot(train_hist.history['loss'])
plt.plot(train_hist.history['val_loss'])
# 모델간 loss 비교를 위해서는 Y축 scale을 고정하는 것이 좋음. 
# 비교가 아닌 경우 코멘트 처리.
# loss_max = max(max(train_hist.history['loss']), max(train_hist.history['val_loss']))

# if loss_max > 1 :
#   plt.ylim(0.0, loss_max)
# elif loss_max > 0.5 :
#   plt.ylim(0.0, 1)
# elif loss_max > 0.1 :
#   plt.ylim(0.0, 0.5)
# elif loss_max > 0.01 :
#   plt.ylim(0.0, 0.1)
# else :
#   plt.ylim(0.0, loss_max*1.5)

print('[ Train Chart ]')
loss_min = min(train_hist.history['val_loss'])
plt.ylim(0.0, loss_min*10)

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# %%script echo skip
print("skip")
# 학습된 중요 Parameter 값 확인 

smoothing_alpha = [var for var in model.trainable_variables if 'smoothing_alpha' in var.name][0].numpy()
print('Exponential smoothing')
print(f"Learned smoothing_alpha (a): {smoothing_alpha}")

process_variance_value = [var for var in model.trainable_variables if 'process_variance' in var.name][0].numpy()
measurement_variance_value = [var for var in model.trainable_variables if 'measurement_variance' in var.name][0].numpy()
print('Kalman filter')
print(f"Learned process_variance (Q): {process_variance_value}")
print(f"Learned measurement_variance (R): {measurement_variance_value}")

%%skip isSkip skip... : MultiStepModel은 학습하지 않으므로 skip
if config['use_model'] == 'LSTM_4Multi' :
    model.save(config['base_model_path'])
    print('The Multi-Step Base Model is saved in ', config['base_model_path'])

    with open(config['base_scaler_path'], 'wb') as file:
        pickle.dump(scaler_x, file)
        pickle.dump(scaler_f, file)
        pickle.dump(scaler_y_step, file)

    print('The Multi-Step Base Scalers are saved in ', config['base_scaler_path'])

%%skip isSkip skip... : MultiStepModel은 Evalution 수행하지 않음.
# 모델 평가

print('------------------------------------')
model.reset_states()
model_result['train_loss'] = model.evaluate([trainX, trainF], trainY, verbose=0)
print(f"Train loss Score : {model_result['train_loss']}")

model.reset_states()
model_result['valid_loss'] = model.evaluate([validX,validF], validY, verbose=0)
print(f"Valid loss Score : {model_result['valid_loss']}")

model.reset_states()
model_result['test_loss'] = model.evaluate([testX, testF], testY, verbose=0)
print(f"Test  loss Score : {model_result['test_loss']}")
print('------------------------------------')


#%%script echo skipping... Tensor Board
print("skipping... Tensor Board")
# Tensor Board = True 설정시에만 실행

if isTensor_board == True :
    %load_ext tensorboard
    %tensorboard --bind_all --logdir ./tensor_board_log