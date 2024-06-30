import os
import numpy as np
import pandas as pd

os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

def generate_temperature_data(days, noise_level=0.0, anomaly=False):
    np.random.seed(42)
    base_temperature = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, days))
    noise = np.random.normal(0, noise_level, days)
    temperatures = base_temperature + noise
    
    if anomaly:
        anomaly_days = np.random.choice(days, size=5, replace=False)
        temperatures[anomaly_days] += np.random.uniform(-10, 10, size=5)

    data = pd.DataFrame({
        'Day': np.arange(1, days + 1),
        'Temperature': temperatures
    })
    
    return data

for i in range(5):
    train_data = generate_temperature_data(365, noise_level=2.0, anomaly=(i % 2 == 0))
    train_data.to_csv(f'train/train_data_{i}.csv', index=False)
    
    test_data = generate_temperature_data(365, noise_level=2.0, anomaly=(i % 2 != 0))
    test_data.to_csv(f'test/test_data_{i}.csv', index=False)

print("Данные успешно созданы")
