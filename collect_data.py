import argparse
import pandas as pd
import requests
import os
from binance import Client
from datetime import datetime, timedelta
import time
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collect_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Создаем папку data, если ее нет
if not os.path.exists('data'):
    os.makedirs('data')
    logger.info("Создана папка data/")

# Список известных стейблкоинов (в нижнем регистре для сравнения)
stablecoins = ['usdt', 'usdc', 'busd', 'dai', 'tusd', 'usdp', 'gusd', 'pax', 'lusd', 'husd']
logger.info(f"Список стейблкоинов для исключения: {stablecoins}")

# Функция для получения списка топ-N криптовалют по рыночной капитализации
def get_top_coins(limit=100):
    logger.info(f"Запрашиваем топ-{limit} криптовалют с CoinGecko...")
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': limit,
        'page': 1,
        'sparkline': False
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Получено {len(data)} записей с CoinGecko")
    except Exception as e:
        logger.error(f"Ошибка при запросе к CoinGecko: {e}")
        return []
    
    # Фильтруем стейблкоины и преобразуем в формат Binance (например, BTCUSDT)
    symbols = []
    for coin in data:
        symbol = coin['symbol'].lower()
        if symbol not in stablecoins:
            binance_symbol = symbol.upper() + 'USDT'
            symbols.append(binance_symbol)
    
    return symbols

# Функция для проверки, торгуется ли пара на Binance
def is_valid_symbol(client, symbol):
    try:
        client.get_symbol_info(symbol)
        logger.info(f"Пара {symbol} торгуется на Binance")
        return True
    except Exception as e:
        logger.warning(f"Пара {symbol} не торгуется на Binance: {e}")
        return False

# Функция для получения исторических данных с Binance
def get_historical_data(client, symbol, interval, start_str, end_str):
    try:
        logger.info(f"Запрашиваем данные для {symbol} с {start_str} по {end_str}...")
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        if not klines:
            logger.warning(f"Нет данных для {symbol}")
            return None
        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                             'close_time', 'quote_asset_volume', 'trades', 
                                             'taker_buy_base', 'taker_buy_quote', 'ignored'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
        logger.info(f"Получено {len(data)} записей для {symbol}")
        return data
    except Exception as e:
        logger.error(f"Ошибка при сборе данных для {symbol}: {e}")
        return None

# Функция для проверки актуальности данных
def check_data_freshness(symbol, end_date_str):
    file_path = f'data/{symbol}_data.csv'
    if not os.path.exists(file_path):
        logger.info(f"Данные для {symbol} отсутствуют")
        return False
    
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            logger.warning(f"Файл для {symbol} пустой")
            return False
        
        # Получаем последнюю дату в данных
        last_date = pd.to_datetime(data['timestamp']).max()
        # Преобразуем end_date в datetime
        end_date = datetime.strptime(end_date_str, '%d %b, %Y')
        
        # Сравниваем даты (без учета времени)
        last_date = last_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if last_date >= end_date:
            logger.info(f"Данные для {symbol} актуальны (последняя дата: {last_date.date()})")
            return True
        else:
            logger.info(f"Данные для {symbol} устарели (последняя дата: {last_date.date()}, требуется: {end_date.date()})")
            return False
    except Exception as e:
        logger.error(f"Ошибка при проверке актуальности данных для {symbol}: {e}")
        return False

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Collect historical data for cryptocurrencies from Binance.')
parser.add_argument('--limit', type=int, default=100, help='Number of top cryptocurrencies to collect (default: 100)')
parser.add_argument('--interval', type=str, default=Client.KLINE_INTERVAL_1DAY, help='Binance kline interval (default: 1DAY)')
parser.add_argument('--start-date', type=str, default='1 Jan, 2020', help='Start date for data collection (default: 1 Jan, 2020)')
parser.add_argument('--end-date', type=str, default=None, help='End date for data collection (default: today)')
args = parser.parse_args()

# Инициализация клиента Binance
try:
    client = Client()
    logger.info("Клиент Binance успешно инициализирован")
except Exception as e:
    logger.error(f"Ошибка при инициализации клиента Binance: {e}")
    exit(1)

# Получаем список топ-N криптовалют (без стейблкоинов)
top_symbols = get_top_coins(limit=args.limit)
logger.info(f"Список символов (до проверки на Binance): {top_symbols}")

# Проверяем, какие символы действительно торгуются на Binance
valid_symbols = []
for symbol in top_symbols:
    if is_valid_symbol(client, symbol):
        valid_symbols.append(symbol)
    time.sleep(0.5)  # Задержка, чтобы избежать лимитов API

logger.info(f"Собираем данные для: {valid_symbols}")

# Параметры для сбора данных
interval = args.interval
start_str = args.start_date
end_str = args.end_date if args.end_date else datetime.now().strftime('%d %b, %Y')

# Минимальное количество записей для сохранения
min_records = 100

# Собираем данные для каждой криптовалюты
for symbol in valid_symbols:
    # Проверяем актуальность данных
    if check_data_freshness(symbol, end_str):
        continue  # Пропускаем, если данные актуальны
    
    logger.info(f"Собираем данные для {symbol}...")
    data = get_historical_data(client, symbol, interval, start_str, end_str)
    if data is not None and not data.empty:
        if len(data) < min_records:
            logger.warning(f"Недостаточно данных для {symbol}: {len(data)} записей, требуется минимум {min_records}")
            continue
        # Сохраняем в CSV
        data.to_csv(f'data/{symbol}_data.csv', index=False)
        logger.info(f"Данные для {symbol} сохранены в data/{symbol}_data.csv")
    else:
        logger.warning(f"Не удалось собрать данные для {symbol}")
    time.sleep(1)  # Задержка, чтобы избежать лимитов API

logger.info("Сбор данных завершен!")
