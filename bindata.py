from binance import Client
import pandas as pd

# Инициализация клиента Binance (без API-ключа для публичных данных)
client = Client()

# Функция для получения исторических данных
def get_historical_data(symbol, interval, start_str, end_str):
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_asset_volume', 'trades', 
                                         'taker_buy_base', 'taker_buy_quote', 'ignored'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

# Параметры
symbol_btc = 'BTCUSDT'  # Пара BTC к USDT
symbol_eth = 'ETHUSDT'  # Пара ETH к USDT
interval = Client.KLINE_INTERVAL_1DAY  # Дневной интервал
start_str = '1 Jan, 2020'  # Начало периода
end_str = '9 Apr, 2025'   # Сегодняшняя дата

# Получение данных
btc_data = get_historical_data(symbol_btc, interval, start_str, end_str)
eth_data = get_historical_data(symbol_eth, interval, start_str, end_str)

# Сохранение в CSV
btc_data.to_csv('btc_data.csv', index=False)
eth_data.to_csv('eth_data.csv', index=False)

print("Данные по BTC:")
print(btc_data.tail())
print("\nДанные по ETH:")
print(eth_data.tail())
