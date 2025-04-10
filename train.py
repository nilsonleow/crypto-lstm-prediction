import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('data/btc_data.csv')
features = data[['close', 'volume', 'high', 'low']].values

# Нормализация
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Создание последовательностей
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Предсказываем только close
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(scaled_features, seq_length)

# Разделение на train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Построение модели
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 4)))
model.add(Dropout(0.3))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Прогноз
predicted = model.predict(X_test)

# Обратная нормализация только для close
predicted = scaler.inverse_transform(
    np.concatenate((predicted, np.zeros((len(predicted), 3))), axis=1)
)[:, 0]
y_test_real = scaler.inverse_transform(
    np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), 3))), axis=1)
)[:, 0]

# Визуализация
plt.plot(y_test_real, label='Real Price')
plt.plot(predicted, label='Predicted Price')
plt.title('BTC Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price (USDT)')
plt.legend()
plt.savefig('prediction_updated.png')
plt.show()

# Ошибка
mse = np.mean((y_test_real - predicted) ** 2)
print(f"Mean Squared Error: {mse}")

model.save('btc_lstm_model.h5')
