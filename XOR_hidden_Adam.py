import random
import numpy as np

random.seed(90025)

# 입력 값 및 정답 값
data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

# 실행 횟수, 학습률, 모멘텀 설정
iterations = 5000
lr = 0.15
beta1 = 0.9
beta2 = 0.999
eps = 3e-8



# 활성화 함수. -1. 시그모이드
# 미분 할 떄와 아닐 떄의 값

def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# 활성화 함수. -2. 하이퍼탄젠트
# 미분 할 떄와 아닐 때

def tanh(x, derivative=False):
    if (derivative == True):
        return 1 - x ** 2
    return np.tanh(x)


def makeMatrix(i, j, fill=0.0):
    mat = []
    for i in range(i):
        mat.append([fill] * j)
    return mat


class NeuralNetwork:
    # 초깃값 지정
    def __init__(self, num_x, num_yh, num_yo, bias=1):
        # 입력 값(num_x), 은닉층의 초깃값(num_yh), 출력층의 초깃값(num_yo), 바이어스
        self.num_x = num_x + bias  # 3
        self.num_yh = num_yh  # [3, 2, 1, 3]
        self.num_yo = num_yo  # 1
        self.m = 0
        self.v = 0

        self.activation_input = [1.0] * self.num_x
        self.activation_hidden = []
        for i in range(len(num_yh)):
            self.activation_hidden.append([1.0] * self.num_yh[i])
        self.activation_out = [1.0] * self.num_yo

        # 가중치 입력 초깃값
        self.weight_in = makeMatrix(self.num_x, self.num_yh[0])
        for i in range(self.num_x):
            for j in range(self.num_yh[0]):
                self.weight_in[i][j] = random.random()

        # 가중치 은닉층 초깃값
        self.weight_hidden = []
        for i in range(len(num_yh) - 1):
            self.weight_hidden.append(makeMatrix(self.num_yh[i], self.num_yh[i + 1]))
            for j in range(self.num_yh[i]):
                for k in range(self.num_yh[i + 1]):
                    self.weight_hidden[i][j][k] = random.random()

        # 가중치 출력 초깃값
        self.weight_out = makeMatrix(self.num_yh[len(num_yh) - 1], self.num_yo)
        for j in range(self.num_yh[len(num_yh) - 1]):
            for k in range(self.num_yo):
                self.weight_out[j][k] = random.random()

        # 모멘텀 SGD를 위한 이전 가중치 초깃값
        self.gradient_in = makeMatrix(self.num_x, self.num_yh[0])
        self.gradient_hidden = []
        for i in range(len(num_yh) - 1):
            self.gradient_hidden.append(makeMatrix(self.num_yh[i], self.num_yh[i + 1]))
        self.gradient_out = makeMatrix(self.num_yh[len(num_yh) - 1], self.num_yo)

    # 업데이트 함수
    def update(self, inputs):
        # 입력층의 활성화 함수
        for i in range(self.num_x - 1):
            self.activation_input[i] = inputs[i]

        # 은닉층의 활성화 함수
        for k in range(len(self.num_yh)):
            for j in range(self.num_yh[k]):
                sum = 0.0
                if k == 0:
                    for i in range(self.num_x):
                        sum = sum + self.activation_input[i] * self.weight_in[i][j]
                else:
                    for i in range(self.num_yh[k - 1]):
                        sum = sum + self.activation_hidden[k - 1][i] * self.weight_hidden[k - 1][i][j]
                # 시그모이드와 tanh 중에서 활성화 함수 선택
                self.activation_hidden[k][j] = tanh(sum, False)

        # 출력층의 활성화 함수
        for k in range(self.num_yo):
            sum = 0.0
            for j in range(self.num_yh[len(self.num_yh) - 1]):
                sum = sum + self.activation_hidden[len(self.num_yh) - 1][j] * self.weight_out[j][k]
            # 시그모이드와 tanh 중에서 활성화 함수 선택
            self.activation_out[k] = tanh(sum, False)

        return self.activation_out[:]

    # 역전파 실행
    def backPropagate(self, targets):
        # 델타 출력 계산
        output_deltas = [0.0] * self.num_yo
        for k in range(self.num_yo):
            error = targets[k] - self.activation_out[k]
            # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
            output_deltas[k] = tanh(self.activation_out[k], True) * error

        # 은닉 노드의 오차 함수
        hidden_deltas = []
        for i in range(len(self.num_yh)):
            hidden_deltas.append([0.0] * self.num_yh[i])

        for i in range(len(self.num_yh) - 1, -1, -1):
            for j in range(self.num_yh[i]):
                error = 0.0
                if i == len(self.num_yh) - 1:
                    for k in range(self.num_yo):
                        error = error + output_deltas[k] * self.weight_out[j][k]
                else:
                    for k in range(self.num_yh[i + 1]):
                        error = error + hidden_deltas[i + 1][k] * self.weight_hidden[i][j][k]
                # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
                hidden_deltas[i][j] = tanh(self.activation_hidden[i][j], True) * error

        # 출력 가중치 업데이트
        for j in range(self.num_yh[len(self.num_yh) - 1]):  # 3
            for k in range(self.num_yo):  # 1
                gradient = output_deltas[k] * self.activation_hidden[len(self.num_yh) - 1][j]
                self.m = beta1 * self.m + (1 - beta1) * gradient
                self.v = beta2 * self.v + (1 - beta2) * (gradient**2)
                m_hat = self.m / (1 - beta1 ** (1 + iterations))
                v_hat = self.v / (1 - beta2 ** (1 + iterations))

                self.weight_out[j][k] += lr * m_hat / (np.sqrt(v_hat) + eps)
                self.gradient_out[j][k] = gradient


        # 은닉 가중치 업데이트
        for i in range(len(self.num_yh) - 2, -1, -1):  # 2 1 0
            for j in range(self.num_yh[i]):  # 1 2 3
                for k in range(self.num_yh[i + 1]):  # 3 1 2
                    gradient = hidden_deltas[i + 1][k] * self.activation_hidden[i][j]
                    self.m = beta1 * self.m + (1 - beta1) * gradient
                    self.v = beta2 * self.v + (1 - beta2) * (gradient**2)
                    m_hat = self.m / (1 - beta1 ** (1 + iterations))
                    v_hat = self.v / (1 - beta2 ** (1 + iterations))

                    self.weight_hidden[i][j][k] += lr * m_hat / (np.sqrt(v_hat) + eps)
                    self.gradient_hidden[i][j][k] = gradient

        # 입력 가중치 업데이트
        for i in range(self.num_x):  # 3
            for j in range(self.num_yh[0]):  # 3
                gradient = hidden_deltas[0][j] * self.activation_input[i]
                self.m = beta1 * self.m + (1 - beta1) * gradient
                self.v = beta2 * self.v + (1 - beta2) * (gradient**2)
                m_hat = self.m / (1 - beta1 ** (1 + iterations))
                v_hat = self.v / (1 - beta2 ** (1 + iterations))

                self.weight_in[i][j] += lr * m_hat / (np.sqrt(v_hat) + eps)
                # self.weight_in[i][j] += (lr / (np.sqrt(v_hat) + eps)) * m_hat
                self.gradient_in[i][j] = gradient

        # 오차의 계산(최소 제곱법)
        error = 0.0
        for k in range(len(targets)):
            delta = 1e-7
            # MSE
            error = error + 0.5 * (targets[k] - self.activation_out[k]) ** 2
            # 크로스엔트로피
            #error = error - np.sum(targets[k] * np.log(self.activation_out[k] + delta))
        return error

    # 훈련
    def train(self, patterns):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets)
            if i % 500 == 0:
                print('error : %-.5f' % error)

    def result(self, patterns):
        for p in patterns:
            print('Input : %s, Predict : %s' % (p[0], self.update(p[0])))


if __name__ == '__main__':
    n = NeuralNetwork(2, [3, 2, 4], 1)
    #n.train([data[0]])
    n.train(data)
    n.result(data)
