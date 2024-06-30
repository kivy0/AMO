import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle

def load_data_from_folder(folder_path):
    data_list = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path)
        data_list.append(data)
    
    combined_data = pd.concat(data_list, ignore_index=True)
    return combined_data

test_data = load_data_from_folder('test')

X_test = test_data[['Day']]
y_test = test_data['Temperature']

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"MSE на тестовых данных: {mse}")
