# 写入
import os
import pandas as pd
import torch
import numpy as np

os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 创建目录
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NaN,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NaN,106000\n')
    f.write('4,NaN,178100\n')
    f.write('NaN,NaN,140000\n')

# 读取文件
data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=1))
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
# 将布尔值转为整数
inputs = inputs.astype(int)
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(Y)

data = {
    'NumRooms': [3.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0],
    'Alley': ['Pave', 'Unknown', np.nan, 'Pave', np.nan, 'Unknown', 'Pave'],
    'Garage': [np.nan, 'Yes', 'No', np.nan, 'Yes', 'No', 'No'],
    'Pool': [np.nan, np.nan, 'Yes', 'No', 'Yes', np.nan, np.nan],
    'Price': [127500, 106000, 178100, 140000, 250000, 310000, np.nan]
}
df = pd.DataFrame(data)

print(df)

miss_counts = df.isnull().sum()
column_drop = miss_counts.idxmax()
df = df.drop(columns=[column_drop])
print(df)

inputs = df.iloc[:, 0:3]
outputs = df.iloc[:, 3]
inputs = inputs.fillna(inputs.mean(numeric_only=1))
inputs = pd.get_dummies(inputs, dummy_na=True)
inputs = inputs.astype(int)
outputs = outputs.fillna(outputs.mean(numeric_only=1))
print(inputs)
print(outputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(Y)