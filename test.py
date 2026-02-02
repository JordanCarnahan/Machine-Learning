# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
arr = np.arange(10).reshape(2, 5)
text = "hello"
value = 42

# %%
torch.sin(torch.tensor(math.pi)) 

import torch
import torch.nn as nn
import math
import numpy as np

def calculate_softmax(attention_scores):
    scores = attention_scores - np.max(attention_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return weights

def calculate_attention_scores(matrix_k_8x4, matrix_v_8x4, d_k):
    raw_scores = np.dot(matrix_k_8x4, matrix_v_8x4.T)
    attention_scores = raw_scores / np.sqrt(d_k)
    #print(attention_scores)
    weights = (calculate_softmax(attention_scores))
    average = np.mean(weights, axis=1)
    print(average)
    min_per_query = np.min(weights, axis=1)
    print("Min per query:", min_per_query)
    max_per_query = np.max(weights, axis=1)
    print("Max per query:", max_per_query)


d_model = 32
h = 4
d_k = d_model // h

array_q = np.array(['if','it','is','serving','then','serve','if','it','is','teaching','then','teach','if','it','is','to','encourage','then','give','encouragement','if','it','is','giving','then','give','generously','if','it','is','to','lead']) # %%
array_k = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
array_v = np.array([0.15625,0.15625,0.15625,0.03125,0.125,0.03125,0.15625,0.15625,0.15625,0.03125,0.125,0.03125,0.15625,0.15625,0.15625,0.0625,0.03125,0.125,0.0625,0.03125,0.15625,0.15625,0.15625,0.03125,0.125,0.0625,0.03125,0.15625,0.15625,0.15625,0.0625,0.03125])

K = array_k.reshape(8, 4)
V = array_v.reshape(8, 4)
calculate_attention_scores(K, V, d_k)

