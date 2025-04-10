import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# Функция для вычисления RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Функция для подготовки данных
def prepare_data(file_path, seq_length):
    try:
        data = pd.read_csv(file_path)
        print(f"Загружен файл {file_path}, размер: {data.shape}")
        data['rsi'] = calculate_rsi(data['close'], periods=14)
        data = data.dropna()
        features = data[['close', 'rsi']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        return scaled_features, scaler
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None, None

# Создание последовательностей
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

# Функция для получения текущей цены с CoinGecko
def get_current_price(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        price = data[coin_id]['usd']
        print(f"Текущая цена {coin_id.upper()} на {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {price} USD")
        return price
    except Exception as e:
        print(f"Ошибка при получении текущей цены для {coin_id}: {e}")
        return None

# Загрузка модели
try:
    model = load_model('multi_crypto_lstm_model_powerful.keras')
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    exit(1)

seq_length = 30

# Список криптовалют для предсказания
coins = ['BTCUSDT', 'ETHUSDT']
coin_ids = {'BTCUSDT': 'bitcoin', 'ETHUSDT': 'ethereum'}  # Соответствие символов CoinGecko

for coin in coins:
    print(f"\nПредсказание для {coin}...")
    # Подготовка данных
    scaled_features, scaler = prepare_data(f'data/{coin}_data.csv', seq_length)
    if scaled_features is None:
        print(f"Пропускаем {coin} из-за ошибки в данных")
        continue
    
    X, y = create_sequences(scaled_features, seq_length)
    
    # Разделение на train/test
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Прогноз на тестовом наборе
    predicted = model.predict(X_test)
    
    # Обратная нормализация
    predicted = scaler.inverse_transform(
        np.concatenate((predicted, np.zeros((len(predicted), 1))), axis=1)
    )[:, 0]
    y_test_real = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), 1))), axis=1)
    )[:, 0]
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_real, label='Real Price')
    plt.plot(predicted, label='Predicted Price')
    plt.title(f'{coin} Price Prediction (Powerful Model)')
    plt.xlabel('Days')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.savefig(f'prediction_{coin}_powerful.png')
    plt.show()
    
    # Ошибка
    mse = np.mean((y_test_real - predicted) ** 2)
    print(f"Mean Squared Error для {coin}: {mse}")
    print(f"RMSE для {coin}: {np.sqrt(mse):.2f} USDT")
    
    # Прогноз на следующий день
    last_sequence = scaled_features[-seq_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    next_day_pred = model.predict(last_sequence)
    next_day_pred = scaler.inverse_transform(
        np.concatenate((next_day_pred, np.zeros((len(next_day_pred), 1))), axis=1)
    )[:, 0]
    predicted_price = next_day_pred[0]
    print(f"Предсказанная цена {coin} на 11 апреля 2025: {predicted_price:.2f} USDT")
    
    # Получение текущей цены
    coin_id = coin_ids[coin]
    current_price = get_current_price(coin_id)
    if current_price is not None:
        # Сравнение предсказанной и текущей цены
        error = abs(predicted_price - current_price)
        relative_error = (error / current_price) * 100 if current_price != 0 else float('inf')
        print(f"Разница между предсказанной и текущей ценой: {error:.2f} USDT")
        print(f"Относительная ошибка: {relative_error:.2f}%")
    else:
        print("Не удалось получить текущую цену для сравнения")
