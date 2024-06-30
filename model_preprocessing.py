import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path, scaler=None):
    data = pd.read_csv(file_path)
    
    if scaler is None:
        scaler = StandardScaler()
        data[['Temperature']] = scaler.fit_transform(data[['Temperature']])
    else:
        data[['Temperature']] = scaler.transform(data[['Temperature']])
    
    return data, scaler
scaler = None
for filename in os.listdir('train'):
    file_path = os.path.join('train', filename)
    processed_data, scaler = load_and_preprocess(file_path, scaler)
    processed_data.to_csv(file_path, index=False)

for filename in os.listdir('test'):
    file_path = os.path.join('test', filename)
    processed_data, _ = load_and_preprocess(file_path, scaler)
    processed_data.to_csv(file_path, index=False)

print("Предобработка данных завершена.")
