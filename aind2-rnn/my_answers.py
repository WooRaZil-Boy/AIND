import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
#from keras.layers import Dense, LSTM, Activation로 쓸 수도 있다.

import keras

import string

# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    #window_size만큼 잘라서 시계열 데이터 인풋과 정답레이블을 만든다.
    #예를 들어 series가 [1, 3, 5, 7, 9, 11, 13]이고 window_size가 2라면 #x와 y는 각각
    #[1, 3], [5]
    #[3, 5], [7]
    #[5, 7], [9]
    #[7, 9], [11]
    #[9, 11], [13] 이 된다.

    # containers for input/output pairs
    X = []
    y = []

    for i in range(0, len(series) - window_size, 1):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential([
        LSTM(5, input_shape=(window_size, 1)), #LSTM 모델 5 #drop out등을 추가 시킬 수도 있다.
		Dense(1) #FC로 마지막을 연결시킨다.
    ]) #선형적인 stack의 모델

    #Sequential() 생성 후, add로 연결시켜 줄 수도 있다.
    # model = Sequential()
    # model.add(LSTM(5, input_shape=(window_size, 1)))
    # model.add(Dense(1))

    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    characters = list(string.ascii_lowercase) #string.ascii_lowercase로 ASCII에 쓰는 문자열을 가져올 수 있다.
    dic = punctuation + characters

    text = text.lower()
    new_text = "".join(ch if ch in dic else " " for ch in text)

    return new_text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    #window_transform_series와 유사하다. step_size만 따로 생각해서 계산해 주면 된다.

    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    #build_part1_RNN과 유사하다.

    model = Sequential([
        LSTM(200, input_shape=(window_size, num_chars)), #LSTM 200개
        Dense(num_chars), #마지막엔 FC를 연결해 준다.
        Activation('softmax') #활성화 함수 소프트맥스. #전체 num_chars 중 하나를 예측하는 것이기 때문에 분류.
    ]) #선형적인 stack의 모델

    #Sequential() 생성 후, add로 연결시켜 줄 수도 있다.
    # model.add(LSTM(200, input_shape=(window_size, num_chars)))
    # model.add(Dense(num_chars))
    # model.add(Activation('softmax'))

    return model
