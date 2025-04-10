import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Функция для вычисления RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Функция для загрузки и подготовки данных
def load_and_prepare_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Загружен файл {file_path}, размер: {data.shape}")
        data['rsi'] = calculate_rsi(data['close'], periods=14)
        data = data.dropna()  # Удаляем NaN после вычисления RSI
        features = data[['close', 'rsi']].values
        return features
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None

# Создание последовательностей
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Предсказываем только close
    return np.array(X), np.array(y)

# Загружаем данные всех криптовалют
data_dir = 'data/'
all_X, all_y = [], []
seq_length = 30
scalers = {}  # Для сохранения скейлеров для каждой криптовалюты

for file_name in os.listdir(data_dir):
    if file_name.endswith('_data.csv'):
        print(f"Обрабатываем {file_name}...")
        coin_id = file_name.replace('_data.csv', '')
        features = load_and_prepare_data(os.path.join(data_dir, file_name))
        
        if features is None or len(features) < seq_length + 1:
            print(f"Пропускаем {file_name}: недостаточно данных")
            continue
        
        # Нормализация для каждой криптовалюты отдельно
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        scalers[coin_id] = scaler  # Сохраняем скейлер для дальнейшего использования
        
        # Создаем последовательности
        X, y = create_sequences(scaled_features, seq_length)
        print(f"Создано {len(X)} последовательностей для {coin_id}")
        all_X.append(X)
        all_y.append(y)

# Проверяем, есть ли данные для обучения
if not all_X:
    print("Ошибка: не удалось создать последовательности для обучения. Проверьте данные.")
    exit(1)

# Объединяем данные
X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
print(f"Общий размер данных: X={X.shape}, y={y.shape}")

# Разделение на train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"Размер тренировочного набора: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Размер тестового набора: X_test={X_test.shape}, y_test={y_test.shape}")

# Построение модели
model = Sequential()
model.add(Input(shape=(seq_length, 2)))  # 2 признака: close, rsi
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
print("Модель скомпилирована")

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Обучение
try:
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
except Exception as e:
    print(f"Ошибка при обучении модели: {e}")
    exit(1)

# Сохранение модели
try:
    model.save('multi_crypto_lstm_model.keras')
    print("Модель сохранена как 'multi_crypto_lstm_model.keras'")
    # Проверяем, что файл действительно создался
    if os.path.exists('multi_crypto_lstm_model.keras'):
        print(f"Файл модели успешно сохранен, размер: {os.path.getsize('multi_crypto_lstm_model.keras')} байт")
    else:
        print("Ошибка: файл модели не был создан")
except Exception as e:
    print(f"Ошибка при сохранении модели: {e}")
    exit(1)

# Прогноз на тестовом наборе (для оценки)
predicted = model.predict(X_test)

# Для оценки MSE на нормализованных данных
mse = np.mean((y_test - predicted) ** 2)
print(f"Mean Squared Error (на нормализованных данных): {mse}")

# Визуализация истории обучения
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')
plt.show()
