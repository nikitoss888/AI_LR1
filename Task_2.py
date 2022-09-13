import numpy as np
from sklearn import preprocessing

# Дані до обробки (11й варіант)
input_data = np.array([
    [-5.3, -8.9, 3.0],
    [2.9, 5.1, -3.3],
    [3.1, -2.8, -3.2],
    [2.2, -1.4, 5.1]
])

# Бінаризація даних
data_binarized = preprocessing.Binarizer(threshold=2).transform(input_data)
print(f"Binarized data:\n{data_binarized}")

# Виключення середнього
print("\nBefore:")
print("Mean = ", input_data.mean(axis=0))
print("Std deviation = ", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nAfter:")
print("Mean = ", data_scaled.mean(axis=0))
print("Std deviation = ", data_scaled.std(axis=0))

# Масштабування
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

# Нормалізація
data_normalized_l1 = preprocessing.normalize(input_data, norm="l1")
data_normalized_l2 = preprocessing.normalize(input_data, norm="l2")
print("\nl1 normalized data:\n", data_normalized_l1)
print("l2 normalized data:\n", data_normalized_l2)
