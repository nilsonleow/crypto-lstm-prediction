import os
import subprocess
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import random

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Создаем каталоги для моделей и результатов
if not os.path.exists('models'):
    os.makedirs('models')
    logger.info("Создана папка models/")
if not os.path.exists('models/results'):
    os.makedirs('models/results')
    logger.info("Создана папка models/results/")

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
        logger.info(f"Загружен файл {file_path}, размер: {data.shape}")
        data['rsi'] = calculate_rsi(data['close'], periods=14)
        data = data.dropna()
        features = data[['close', 'rsi']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        return scaled_features, scaler
    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file_path}: {e}")
        return None, None

# Создание последовательностей
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Предсказываем только close
    return np.array(X), np.array(y)

# Функция для получения списка файлов данных
def get_data_files(model_type):
    data_dir = 'data/'
    if not os.path.exists(data_dir):
        logger.error(f"Папка {data_dir} не существует. Убедитесь, что данные собраны.")
        exit(1)
    
    all_files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv')]
    
    if model_type == 'mini':
        # Только BTC
        btc_file = 'BTCUSDT_data.csv'
        if btc_file in all_files:
            return [os.path.join(data_dir, btc_file)]
        else:
            logger.error("Файл BTCUSDT_data.csv не найден. Убедитесь, что данные собраны.")
            exit(1)
    elif model_type == 'standard':
        # Топ-20 криптовалют
        return [os.path.join(data_dir, f) for f in all_files[:20]]
    else:  # powerful
        # Все доступные данные
        return [os.path.join(data_dir, f) for f in all_files]

# Функция для определения версии модели
def get_model_version(model_type):
    model_dir = 'models/'
    existing_models = [f for f in os.listdir(model_dir) if f.startswith(f"{model_type}_v") and f.endswith('.keras')]
    if not existing_models:
        return 1
    versions = [int(f.split('_v')[1].split('.keras')[0]) for f in existing_models]
    return max(versions) + 1

# Функция для создания модели
def create_model(model_type, seq_length, feature_count):
    model = Sequential()
    if model_type in ['mini', 'standard']:
        # Архитектура для mini и standard (как в старом train.py)
        model.add(Input(shape=(seq_length, feature_count)))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100))
        model.add(Dropout(0.2))
        model.add(Dense(units=50))
        model.add(Dense(units=1))
    else:  # powerful
        # Архитектура для powerful (как в train_v2.py)
        model.add(Input(shape=(seq_length, feature_count)))
        model.add(LSTM(units=150, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=150, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=150, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=150))
        model.add(Dropout(0.2))
        model.add(Dense(units=75))
        model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Train LSTM models for cryptocurrency price prediction.')
parser.add_argument('--model', type=str, default='mini', choices=['mini', 'standard', 'powerful'],
                    help='Model type to train: mini (BTC only), standard (top 20 cryptos), powerful (all data)')
args = parser.parse_args()

model_type = args.model
logger.info(f"Обучаем модель типа: {model_type}")

# Шаг 1: Сбор данных
logger.info("Запускаем сбор данных...")
try:
    subprocess.run(['python3', 'collect_data.py', '--limit', '100', '--start-date', '1 Jan, 2020', '--end-date', datetime.now().strftime('%d %b, %Y')], check=True)
    logger.info("Сбор данных завершен")
except subprocess.CalledProcessError as e:
    logger.error(f"Ошибка при сборе данных: {e}")
    exit(1)

# Шаг 2: Подготовка данных
seq_length = 30
data_files = get_data_files(model_type)
logger.info(f"Используем данные из файлов: {data_files}")

all_X, all_y = [], []
scalers = {}

for file_path in data_files:
    scaled_features, scaler = prepare_data(file_path, seq_length)
    if scaled_features is None:
        continue
    
    symbol = os.path.basename(file_path).replace('_data.csv', '')
    scalers[symbol] = scaler
    X, y = create_sequences(scaled_features, seq_length)
    all_X.append(X)
    all_y.append(y)

# Объединяем данные
if not all_X or not all_y:
    logger.error("Не удалось подготовить данные для обучения")
    exit(1)

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
logger.info(f"Общий размер данных: X={X.shape}, y={y.shape}")

# Разделение на train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]  # Исправлено

# Шаг 3: Создание и обучение модели
model = create_model(model_type, seq_length, X.shape[2])
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Шаг 4: Сохранение модели
version = get_model_version(model_type)
model_name = f"{model_type}_v{version}"
model_path = f'models/{model_name}.keras'
model.save(model_path)
logger.info(f"Модель сохранена как '{model_path}'")
logger.info(f"Размер файла модели: {os.path.getsize(model_path)} байт")

# Шаг 5: Оценка модели
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
logger.info(f"Mean Squared Error (на нормализованных данных): {mse}")

# Шаг 6: Прогноз на следующий день для BTC, ETH и случайной криптовалюты
symbols_to_predict = ['BTCUSDT', 'ETHUSDT']
# Выбираем случайную криптовалюту из списка, на которых обучалась модель
available_symbols = [os.path.basename(f).replace('_data.csv', '') for f in data_files]
# Исключаем BTC и ETH из случайного выбора
random_candidates = [s for s in available_symbols if s not in symbols_to_predict]
if random_candidates:
    random_symbol = random.choice(random_candidates)
    symbols_to_predict.append(random_symbol)
    logger.info(f"Случайная криптовалюта для прогноза: {random_symbol}")
else:
    logger.warning("Нет дополнительных криптовалют для случайного прогноза")

pred_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
for symbol in symbols_to_predict:
    file_path = f'data/{symbol}_data.csv'
    if os.path.exists(file_path):
        scaled_features, scaler = prepare_data(file_path, seq_length)
        if scaled_features is not None:
            # Прогноз на следующий день
            last_sequence = scaled_features[-seq_length:]
            last_sequence = np.expand_dims(last_sequence, axis=0)
            next_day_pred = model.predict(last_sequence)
            next_day_pred = scaler.inverse_transform(
                np.concatenate((next_day_pred, np.zeros((len(next_day_pred), 1))), axis=1)
            )[:, 0]
            logger.info(f"Предсказанная цена {symbol} на {pred_date}: {next_day_pred[0]:.2f} USDT")
            
            # Создание графика
            plt.figure(figsize=(10, 6))
            # Реальные данные (последние 30 дней)
            data = pd.read_csv(file_path)
            real_prices = data['close'].values[-30:]
            plt.plot(range(30), real_prices, label='Real Price', color='blue')
            # Прогноз
            plt.plot([30], next_day_pred, 'ro', label='Predicted Price (Next Day)')
            plt.title(f'{symbol} Price Prediction ({model_type.capitalize()} Model v{version})')
            plt.xlabel('Days')
            plt.ylabel('Price (USDT)')
            plt.legend()
            plt.grid()
            result_path = f'models/results/{model_name}_{symbol}_prediction.png'
            plt.savefig(result_path)
            plt.close()
            logger.info(f"График для {symbol} сохранен в {result_path}")
        else:
            logger.warning(f"Не удалось сделать прогноз для {symbol}: ошибка в данных")
    else:
        logger.warning(f"Файл {symbol}_data.csv не найден, прогноз не выполнен")

# Шаг 7: График потерь
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Training and Validation Loss ({model_type.capitalize()} Model v{version})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
loss_plot_path = f'models/results/{model_name}_loss.png'
plt.savefig(loss_plot_path)
plt.close()
logger.info(f"График потерь сохранен в {loss_plot_path}")
