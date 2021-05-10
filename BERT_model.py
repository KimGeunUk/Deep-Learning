import re
import time
import random
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

train = pd.read_csv('train_classify.csv', sep=',')
test = pd.read_csv('test_classify.csv', sep=',')

sentences = train['reviews'].values
labels = train['sentiment'].values

# 텍스트 전처리하기
def text_preprocessing(sentence):
    sentence = re.sub(r"(\!)\1+", '!', sentence)
    sentence = re.sub(r"(\?)\1+", '?', sentence)
    sentence = re.sub(r"(\.)\1+", '.', sentence)
    sentence = re.sub(r"(\*)\1+", '*', sentence)
    sentence = re.sub(r':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', sentence)
    sentence = re.sub(r'\!|\"|\#|\$|\%|\&|\\|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\?|\@|\[|\\|\\|\]|\^|\_|\{|\||\}|\~|\`', '', sentence)
    return sentence

# 훈련 데이터, 검증 데이터 분류
train_X, val_X, train_Y, val_Y = train_test_split(sentences, labels, test_size=0.1, random_state=42)

# GPU 잡기
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# 버트 토크나이저 불러오기
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 100

# 버트를 위해 input_ids, attention_masks 생성
def preprocessing_bert(sentence):
    input_ids = []
    attention_masks = []

    for sent in sentence:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # 텍스트를 토큰으로 분할
            add_special_tokens=True,        # 특수 [CLS] 및 [SEP] 토큰 추가
            max_length=MAX_LEN,             # 최대 길이로 채우거나 자르기
            pad_to_max_length=True,         # 최대 길이로 문장 채우기
            return_attention_mask=True      # 어텐션 마스크 추가
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

train_inputs, train_masks = preprocessing_bert(train_X)
val_inputs, val_masks = preprocessing_bert(val_X)

# 라벨을 torch의 tensor로 변환
train_Y = torch.tensor(train_Y)
val_Y = torch.tensor(val_Y)

# 파인튜닝을 위해 배치 사이즈 조정
batch_size = 10

#DataLoader 클래스를 사용하여 데이터 Set에 대한 반복기 생성 (메모리 절약, 훈련 속도 향상)
train_data = TensorDataset(train_inputs, train_masks, train_Y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_Y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        # Bert 파라미터 설정
        D_in, H, D_out = 768, 50, 2
        # Bert 모델
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # 피드 포워드 분류
        self.classifier = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out))
        # BERT 모델 고정
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits

# 버트 최적화 시키기 ( batch = 16 or 32, 학습률(Adam) 5e-5, 3e-5 or 2e-5, epochs = 2 or 3 or 4 )
def initialize_model(epochs):
    bert_classifier = BertClassifier(freeze_bert=False)
    # GPU 사용
    bert_classifier.to(device)
    # 최적화
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,                          # 기본 학습률
                      eps=1e-8                          # 기본 엡실론
                      )

    total_steps = len(train_dataloader) * epochs
    # 학습률 스케쥴러 설정
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    return bert_classifier, optimizer, scheduler

# 오차 함수 지정
loss_fn = nn.CrossEntropyLoss()

# 랜덤 시드 고정
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    print("Start training...\n")

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # 결과 틀
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        # 경과 시간 측정
        t0_epoch, t0_batch = time.time(), time.time()

        # Epoch 시작시 변수 초기화
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # 모델 훈련 모드
        model.train()

        # 훈련 데이터의 배치 설정
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # GPU에 batch 로드
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # 계산된 가중치 초기화
            model.zero_grad()

            # forward pass
            logits = model(b_input_ids, b_attn_mask)

            # 오차를 구하고 오차 값 누적
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # 기울기를 위해 역전파
            loss.backward()

            # 기울지 폭주를 방지하기 위해 표준을 1.0으로 자르기
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 파라미터와 학습률 업데이트
            optimizer.step()
            scheduler.step()

            # 5개 배치마다 오차 값과 경과 시간 출력
            if (step % 5 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # 경과 시간
                time_elapsed = time.time() - t0_batch

                # 훈련 결과
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # 배치 변수 초기화
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # 전체 훈련 데이터에 대한 평균 오차값 계산
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # 모델 성능 측정
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            time_elapsed = time.time() - t0_epoch

            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    print("Training complete!")

    model_name = 'bert_model.h5'
    model.save(model_name)

def evaluate(model, val_dataloader):
    # 모델 평가 모드
    model.eval()

    # 검증 추적
    val_accuracy = []
    val_loss = []

    # 검증 데이터를 각 배치에 적용
    for batch in val_dataloader:
        # GPU 사용
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # 정확도 비율 계산
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # 검증 데이터 Set에 대한 정확도와 오차를 계산
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

set_seed(42)    # 랜덤 seed 지정
bert_classifier, optimizer, scheduler = initialize_model(epochs=4)
train(bert_classifier, train_dataloader, val_dataloader, epochs=4, evaluation=True)