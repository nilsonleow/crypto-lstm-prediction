from binance import Client
import pandas as pd
import requests
import os
from datetime import datetime

# Создаем папку data, если ее нет
if not os.path.exists('data'):
    os.makedirs('data')

# Список известных стейблкоинов (в нижнем регистре для сравнения)
stablecoins = ['usdt', 'usdc', 'busd', 'dai', 'tusd', 'usdp', 'gusd', 'pax']

# Функция для получения списка 25 крупнейших криптовалют по рыночной капитализации
def get_top_25_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 50,  # Берем больше, чтобы после фильтрации осталось достаточно
        'page': 1,
        'sparkline': False
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Фильтруем стейблкоины и преобразуем в формат Binance (например, BTCUSDT)
    symbols = []
    for coin in data:
        symbol = coin['symbol'].lower()
        if symbol not in stablecoins:  # Исключаем стейблкоины
            binance_symbol = symbol.upper() + 'USDT'
            symbols.append(binance_symbol)
    
    # Ограничиваем список до 25 символов (после исключения стейблкоинов)
    return symbols[:25]

# Функция для проверки, торгуется ли пара на Binance
def is_valid_symbol(client, symbol):
    try:
        client.get_symbol_info(symbol)
        return True
    except:
        return False

# Функция для получения исторических данных с Binance
def get_historical_data(client, symbol, interval, start_str, end_str):
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                             'close_time', 'quote_asset_volume', 'trades', 
                                             'taker_buy_base', 'taker_buy_quote', 'ignored'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return data
    except Exception as e:
        print(f"Ошибка при сборе данных для {symbol}: {e}")
        return None

# Инициализация клиента Binance
client = Client()

# Получаем список 25 крупнейших криптовалют (без стейблкоинов)
top_25_symbols = get_top_25_coins()
print(f"Список символов (до проверки на Binance): {top_25_symbols}")

# Проверяем, какие символы действительно торгуются на Binance
valid_symbols = []
for symbol in top_25_symbols:
    if is_valid_symbol(client, symbol):
        valid_symbols.append(symbol)
    else:
        print(f"Пара {symbol} не торгуется на Binance, пропускаем.")

print(f"Собираем данные для: {valid_symbols}")

# Параметры для сбора данных
interval = Client.KLINE_INTERVAL_1DAY
start_str = '1 Jan, 2020'
end_str = '10 Apr, 2025'  # Сегодняшняя дата

# Собираем данные для каждой криптовалюты
for symbol in valid_symbols:
    print(f"Собираем данные для {symbol}...")
    data = get_historical_data(client, symbol, interval, start_str, end_str)
    if data is not None and not data.empty:
        # Сохраняем в CSV
        data.to_csv(f'data/{symbol}_data.csv', index=False)
        print(f"Данные для {symbol} сохранены в data/{symbol}_data.csv")
    else:
        print(f"Не удалось собрать данные для {symbol}")

print("Сбор данных завершен!")
