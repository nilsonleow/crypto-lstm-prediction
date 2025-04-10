import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Функция для вычисления RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Функция для вычисления MACD
def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Загрузка данных
data = pd.read_csv('data/btc_data.csv')
data['rsi'] = calculate_rsi(data['close'], periods=14)
data['macd'], data['macd_signal'] = calculate_macd(data['close'])
data = data.dropna()  # Удаляем NaN после вычисления индикаторов

# Подготовка признаков
features = data[['close', 'rsi', 'macd', 'macd_signal']].values

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
model.add(Input(shape=(seq_length, 4)))  # 4 признака: close, rsi, macd, macd_signal
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Обучение
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

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
plt.title('BTC Price Prediction with RSI and MACD')
plt.xlabel('Days')
plt.ylabel('Price (USDT)')
plt.legend()
plt.savefig('prediction_rsi_macd.png')
plt.show()

# Ошибка
mse = np.mean((y_test_real - predicted) ** 2)
print(f"Mean Squared Error: {mse}")

# Сохранение модели
model.save('btc_lstm_model.keras')
