import re
import nltk

import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 데이터 로드
train_data = pd.read_csv('train_classify.csv', sep=',')
test_data = pd.read_csv('test_classify.csv', sep=',')

# 텍스트 전처리
def preprocessing1(sentence):
    result = []

    sentence = removeShortWord(sentence)
    sentence = replaceURL(sentence)
    sentence = removeNumbers(sentence)
    sentence = removeEmoticons(sentence)
    sentence = removePunctuation(sentence)
    sentence = replaceLower(sentence)

    WordTypes = ['J', 'R', 'V', 'N']  # J는 형용사, R은 부사, V는 동사, N은 명사 > 필요한 단어 구분
    s = open('en_stopword.txt', 'r', encoding='utf-8')  # 불용어 list, 추가
    stop_words = stopwords.words('english')
    for word in s:
        if word.strip() not in stop_words:
            stop_words.append(word.strip())

    L = WordNetLemmatizer()
    S = PorterStemmer()

    tokens = nltk.word_tokenize(sentence)  # 토큰화
    tagged = nltk.pos_tag(tokens)  # pos 정보 입력

    for w in tagged:
        if (w[1][0] in WordTypes and w[0] not in stop_words):
            new_word = L.lemmatize(w[0])  # 표제어 추출
            new_word = S.stem(new_word)  # 어간 추출
            result.append(new_word)

    return result
# URL 변환
def replaceURL(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'url', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text
# 짧은 단어(1, 2단어) 제거
def removeShortWord(text):
    text = re.sub(r'\W*\b\w{1,2}\b', r'', text)
    return text
# 숫자 제거
def removeNumbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text
# 이모티콘 제거
def removeEmoticons(text):
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text
# 소문자 변환
def replaceLower(text):
    text = text.lower()
    return text
# 특수문자 제거
def removePunctuation(text):
    text = re.sub(r'\!|\"|\#|\$|\%|\&|\\|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\?|\@|\[|\\|\\|\]|\^|\_|\{|\||\}|\~|\`', '', text)
    return text

# 텍스트 전처리 실행
token_list = []
print('훈련 데이터 전처리 중...')
for i in train_data['reviews']:
    token_list.append(preprocessing1(i))
train_data['tokens'] = token_list

token_list = []
print('테스트 데이터 전처리 중...')
for i in test_data['reviews']:
    token_list.append(preprocessing1(i))
test_data['tokens'] = token_list

# 훈련 데이터, 검증 데이터 구분
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

print('훈련용 리뷰 개수 : ', len(train_data))
print('테스트용 리뷰 개수 : ', len(test_data))
print('검증용 리뷰 개수 : ', len(val_data))
print('\n')

# 데이터 셋
X_train = train_data['tokens'].values
Y_train = train_data['sentiment'].values
X_test = test_data['tokens'].values
Y_test = test_data['sentiment'].values
X_val = val_data['tokens'].values
Y_val = val_data['sentiment'].values

tokenizer = Tokenizer()

# 정수 인코딩
tokenizer.fit_on_texts(X_train)

def vocab_size():
    threshold = 2                               # 최소 등장 빈도 수
    total_cnt = len(tokenizer.word_index)       # 단어의 수
    rare_cnt = 0                                # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0                              # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0                               # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value
        # 단어의 등장 빈도수가 threshold보다 작으면
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    # 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거
    # 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
    vocab_size = total_cnt - rare_cnt + 2
    return vocab_size

# 인덱스가 1이면 빈도수가 2이하인 단어(OOV)
tokenizer = Tokenizer(vocab_size(), oov_token='OOV')

tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_val = tokenizer.texts_to_sequences(X_val)

# 문장 최대 길이 패딩
max_len = 60

# max_len 이하의 길이를 가지도록 패딩
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
X_val = pad_sequences(X_val, maxlen=max_len)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

batch_size = Y_train.shape[0]
input_dim = 1
Y_train = encoder.fit_transform((Y_train))
Y_train = np.reshape(Y_train, (batch_size, input_dim))

batch_size = Y_test.shape[0]
Y_test = encoder.transform((Y_test))
Y_test = np.reshape(Y_test, (batch_size, input_dim))

print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
print('Y_train shape', Y_train.shape)
print('Y_test shape', Y_test.shape)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

early_stopping = EarlyStopping()
model = Sequential()
model.add(Embedding(vocab_size(), 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=32, epochs=300, callbacks=[early_stopping])

loss, acc = model.evaluate(X_test, Y_test, batch_size=32)
print('Test loss:', loss)
print('Test accuracy:', acc)

model_name = 'model2.h5'
model.save(model_name)
