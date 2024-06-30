import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

def load_data_from_folder(folder_path):
    data_list = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path)
        data_list.append(data)
    
    combined_data = pd.concat(data_list, ignore_index=True)
    return combined_data

train_data = load_data_from_folder('train')

X_train = train_data[['Day']]
y_train = train_data['Temperature']

model = RandomForestRegressor()
model.fit(X_train, y_train)

with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Модель успешно обучена.")