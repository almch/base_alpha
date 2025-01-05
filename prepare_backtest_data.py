import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from ranknet_model import prepare_data, get_token_rankings, RankNet

# Добавляем RankNet в список безопасных классов
torch.serialization.add_safe_globals([RankNet])

def prepare_daily_rankings(swaps_df, tokens_df, model, start_date, end_date):
    """
    Подготовка ежедневных рангов токенов
    """
    daily_rankings = []
    current_date = start_date
    
    print(f"Подготовка рангов с {start_date} по {end_date}")
    
    while current_date <= end_date:
        print(f"Обработка даты: {current_date}")
        
        # Преобразуем даты в datetime
        week_start = pd.to_datetime(current_date) - timedelta(days=7)
        current_datetime = pd.to_datetime(current_date)
        
        # Берем данные за последнюю неделю для каждого дня
        week_data = swaps_df[
            (pd.to_datetime(swaps_df['BLOCK_TIMESTAMP']) >= week_start) &
            (pd.to_datetime(swaps_df['BLOCK_TIMESTAMP']) < current_datetime)
        ]
        
        if len(week_data) == 0:
            print(f"Предупреждение: нет данных для {current_date}")
            current_date += timedelta(days=1)
            continue
            
        # Подготавливаем признаки
        features_df = prepare_data(week_data, tokens_df)
        
        if len(features_df) == 0:
            print(f"Предупреждение: нет признаков для {current_date}")
            current_date += timedelta(days=1)
            continue
        
        # Получаем ранги на текущий день
        try:
            rankings = get_token_rankings(model, features_df)
            rankings['date'] = current_date
            daily_rankings.append(rankings)
            print(f"Успешно добавлены ранги для {current_date}, токенов: {len(rankings)}")
        except Exception as e:
            print(f"Ошибка при получении рангов для {current_date}: {str(e)}")
        
        current_date += timedelta(days=1)
    
    if not daily_rankings:
        raise ValueError("Не удалось получить ранги ни для одного дня")
    
    # Объединяем все ранги
    daily_rankings_df = pd.concat(daily_rankings, ignore_index=True)
    print(f"Всего дней с рангами: {len(daily_rankings)}")
    print(f"Всего записей в итоговом DataFrame: {len(daily_rankings_df)}")
    
    return daily_rankings_df

def prepare_price_data(swaps_df, start_date, end_date):
    """
    Подготовка ежедневных цен токенов
    """
    # Группируем свапы по дням и токенам
    daily_prices = []
    
    for token in swaps_df['TOKEN_IN'].unique():
        token_swaps = swaps_df[
            (swaps_df['TOKEN_IN'] == token) |
            (swaps_df['TOKEN_OUT'] == token)
        ]
        
        # Агрегируем по дням
        token_daily = token_swaps.groupby(
            pd.to_datetime(token_swaps['BLOCK_TIMESTAMP']).dt.date
        ).agg({
            'AMOUNT_IN_USD': 'sum',
            'AMOUNT_IN': 'sum',
            'AMOUNT_OUT_USD': 'sum',
            'AMOUNT_OUT': 'sum'
        }).reset_index()
        
        # Вычисляем средневзвешенную цену
        token_daily['price'] = (
            (token_daily['AMOUNT_IN_USD'] + token_daily['AMOUNT_OUT_USD']) /
            (token_daily['AMOUNT_IN'] + token_daily['AMOUNT_OUT'])
        )
        
        token_daily['token_address'] = token
        daily_prices.append(token_daily[['BLOCK_TIMESTAMP', 'token_address', 'price']])
    
    price_data = pd.concat(daily_prices, ignore_index=True)
    
    # Заполняем пропущенные дни
    date_range = pd.date_range(start_date, end_date)
    tokens = price_data['token_address'].unique()
    
    # Создаем полный датасет с всеми комбинациями дат и токенов
    full_index = pd.MultiIndex.from_product(
        [date_range, tokens],
        names=['date', 'token_address']
    )
    
    price_data = price_data.set_index(['BLOCK_TIMESTAMP', 'token_address'])\
        .reindex(full_index)\
        .fillna(method='ffill')\
        .reset_index()
    
    return price_data

def load_parquet_files(directory_path, file_pattern):
    """
    Загружает и объединяет все parquet файлы из указанной директории
    
    Args:
        directory_path: путь к директории с файлами
        file_pattern: паттерн для поиска файлов (например, 'dex_swaps_3m')
    """
    import glob
    import os
    
    print(f"Загрузка файлов из {directory_path} с паттерном {file_pattern}")
    
    # Получаем список всех parquet файлов в директории
    files = glob.glob(os.path.join(directory_path, f"*{file_pattern}*.parquet"))
    print(f"Найдено файлов: {len(files)}")
    
    # Загружаем и объединяем все файлы
    dfs = []
    for file in files:
        print(f"Загрузка файла: {os.path.basename(file)}")
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Объединяем все DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Общее количество записей: {len(combined_df)}")
    
    return combined_df

def main():
    """
    Подготовка данных для бэктестинга
    """
    print("Загрузка данных...")
    # Загружаем все файлы свапов за 3 месяца
    swaps_df = load_parquet_files(
        'data/database/dex_swaps_3m', 
        'dex_swaps_3m'
    )
    
    # Загружаем данные о токенах
    tokens_df = pd.read_parquet('data/database/training_tokens.parquet')
    
    # Проверяем данные
    print(f"Количество свапов: {len(swaps_df)}")
    print(f"Количество токенов: {len(tokens_df)}")
    print(f"Диапазон дат в свапах: {swaps_df['BLOCK_TIMESTAMP'].min()} - {swaps_df['BLOCK_TIMESTAMP'].max()}")
    
    # Определяем временной диапазон
    start_date = pd.to_datetime(swaps_df['BLOCK_TIMESTAMP'].min()) + timedelta(days=7)
    end_date = pd.to_datetime(swaps_df['BLOCK_TIMESTAMP'].max())
    
    print(f"Период бэктестинга: {start_date} - {end_date}")
    
    print("Загрузка модели...")
    try:
        # Пытаемся загрузить модель напрямую
        model = torch.load('best_model.pth')
        if not isinstance(model, RankNet):
            # Если загруженный объект не является моделью RankNet,
            # создаем новую модель с правильными параметрами
            input_size = len(prepare_data(swaps_df.head(), tokens_df).drop(['TOKEN_ADDRESS'], axis=1).columns)
            model = RankNet(input_size=input_size)
            model.load_state_dict(torch.load('best_model.pth'))
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        # Создаем новую модель
        input_size = len(prepare_data(swaps_df.head(), tokens_df).drop(['TOKEN_ADDRESS'], axis=1).columns)
        model = RankNet(input_size=input_size)
        model.load_state_dict(torch.load('best_model.pth'))
    
    model.eval()  # Переводим модель в режим оценки
    
    print("Подготовка ежедневных рангов...")
    daily_rankings = prepare_daily_rankings(
        swaps_df, 
        tokens_df, 
        model, 
        start_date, 
        end_date
    )
    
    print("Подготовка данных о ценах...")
    price_data = prepare_price_data(
        swaps_df,
        start_date,
        end_date
    )
    
    # Сохраняем подготовленные данные
    daily_rankings.to_csv('daily_rankings.csv', index=False)
    price_data.to_csv('token_prices.csv', index=False)
    
    print("Данные подготовлены и сохранены:")
    print("- daily_rankings.csv")
    print("- token_prices.csv")
    
    # Запускаем бэктестинг
    print("\nЗапуск бэктестинга...")
    from backtest import run_backtest
    run_backtest('daily_rankings.csv', 'token_prices.csv')

if __name__ == "__main__":
    main() 