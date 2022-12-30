from random import random
import numpy as np
import math

def getmMtrixWeight(hn,ls):
    W1 = []
    for neuron in range(hn):
        arr = []
        for digit in range(ls):
            arr.append(random() * 2 - 1)
        W1.append(arr)
    return np.array(W1)

def activation(x):
    return math.log10(x + math.pow((math.pow(x,2)+1),0.5))

def der_activation(x):
    return(2*x*math.pow((math.pow(x,2)+1),0.5)+2*math.pow(x,2)+1)/(2*math.pow(x,3)+math.pow((math.pow(x,2)+1),0.5)*(2*math.pow(x,2)+1)+2*x)

def W_1_learning(W_1):
    W_1 = W_1 - alpha * np.matmul((er * np.array([Y_1]).transpose()), np.array([X]))
    return W_1

def W_2_learning(W_2):
    W_2 = W_2 - alpha * (res - s) * np.array([Y]).transpose()
    return W_2

def W_3_learning(W_3):
    W_3 = W_3 - alpha * np.array([Y_1]) * er * np.array(Context)
    return W_3

if __name__ == '__main__':
    # 1 3 5 7 9 11 13
    sequence = np.array([[1, 2, 3],[2, 3, 4],[3, 4, 5],[4, 5, 6]])
    sample = np.array([4, 5, 6, 7])
    Context = np.array([0,0,0])
    T=np.array([[0,0,0]])
    T_1 = 0
    p = 3
    e = 1
    alpha = 0.01
    N = 3
    m = len(sequence[0])
    W_1 = getmMtrixWeight(p, m)  # """W_I"""
    W_2 = getmMtrixWeight(p, 1) #"""W_O"""
    W_3 = getmMtrixWeight(p, p) #"""W_C"""
    steps = 0
    while True:
        E = 0
        for q in range(len(sequence)):
            X = np.array(sequence[q])
            s = sample[q]
            H = np.matmul(X,W_1)
            C = np.matmul(Context,W_3)
            S = np.array([])
            for i in range(len(X)):
                S = np.append(S, X[i] + H[i]-T[0, i])
            Y=np.array([])
            for i in range(len(S)):
                Y = np.append(Y, activation(S[i]))
            Context = np.array(Y)
            Z = np.matmul(Y,W_2) - T_1
            res = np.array([])
            for i in range(len(Z)):
                res = np.append(res, activation(Z[i]))
            W_2 = W_2_learning(W_2)
            T_1 = T_1 + alpha * (res[0] - s)
            er = (res - s) * W_2
            Y_1 = np.array([])
            for i in range(len(S)):
                Y_1 = np.append(Y_1, der_activation(S[i]))
            W_1 = W_1_learning(W_1)
            W_3 = W_3_learning(W_3)
            T = T + alpha * er.transpose() * np.array([Y_1])
            E_i = (res[0] - s) ** 2
            E += E_i
            print("res = ", res, "; s = ", sample[q])
            print(E)
        if E <= e:
            break

    H = np.matmul([1, 4, 7], W_1)
    C = np.matmul(Context, W_3)
    S = np.array([])
    for i in range(len(X)):
        S = np.append(S, X[i] + H[i] - T[0, i])
    Y = np.array([])
    for i in range(len(S)):
        Y = np.append(Y, activation(S[i]))
    Context = np.array(Y)
    Z = np.matmul(Y, W_2) - T_1
    res = np.array([])
    for i in range(len(Z)):
        res = np.append(res, activation(Z[i]))
    print(round(res[0]))