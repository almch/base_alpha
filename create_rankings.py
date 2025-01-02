import pandas as pd

def create_token_rankings():
    try:
        # Читаем данные
        swaps_df = pd.read_parquet('data/database/dex_swaps_1w.parquet')
        tokens_df = pd.read_parquet('data/database/training_tokens.parquet')
        
        # Обрабатываем TOKEN_IN
        token_in_volumes = swaps_df.groupby('TOKEN_IN').agg({
            'AMOUNT_IN_USD': 'sum',
            'TX_HASH': 'count',
            'SENDER': 'nunique'
        }).reset_index()
        
        token_in_volumes.columns = ['token', 'total_volume', 'trade_count', 'unique_traders']
        
        # Обрабатываем TOKEN_OUT
        token_out_volumes = swaps_df.groupby('TOKEN_OUT').agg({
            'AMOUNT_OUT_USD': 'sum',
            'TX_HASH': 'count',
            'SENDER': 'nunique'
        }).reset_index()
        
        token_out_volumes.columns = ['token', 'total_volume', 'trade_count', 'unique_traders']
        
        # Объединяем объемы
        token_volumes = pd.concat([token_in_volumes, token_out_volumes])
        token_volumes = token_volumes.groupby('token').sum().reset_index()
        
        # Нормализуем метрики
        for col in ['total_volume', 'trade_count', 'unique_traders']:
            if token_volumes[col].max() - token_volumes[col].min() != 0:
                token_volumes[f'{col}_normalized'] = (token_volumes[col] - token_volumes[col].min()) / \
                                                   (token_volumes[col].max() - token_volumes[col].min())
            else:
                token_volumes[f'{col}_normalized'] = 1
        
        # Вычисляем общий скор
        token_volumes['score'] = (
            token_volumes['total_volume_normalized'] * 0.4 +
            token_volumes['trade_count_normalized'] * 0.3 +
            token_volumes['unique_traders_normalized'] * 0.3
        )
        
        # Сортируем по скору
        token_volumes = token_volumes.sort_values('score', ascending=False)
        
        # Создаем DataFrame для результата
        result_df = pd.DataFrame()
        result_df['ADDRESS'] = tokens_df['TOKEN_ADDRESS']  # Исправлено на TOKEN_ADDRESS
        
        # Присваиваем ранги
        ranked_tokens = token_volumes['token'].tolist()
        ranks = {}
        
        for i, token in enumerate(ranked_tokens, 1):
            ranks[token] = i
            
        # Для оставшихся токенов
        remaining_tokens = set(tokens_df['TOKEN_ADDRESS']) - set(ranked_tokens)  # Исправлено на TOKEN_ADDRESS
        for i, token in enumerate(remaining_tokens, len(ranked_tokens) + 1):
            ranks[token] = i
            
        # Заполняем ранги
        result_df['RANK'] = result_df['ADDRESS'].map(ranks)
        
        # Проверяем, что все токены получили ранг
        if result_df['RANK'].isnull().any():
            print("Внимание: Некоторые токены не получили ранг!")
            # Присваиваем последний ранг всем токенам без ранга
            last_rank = max(ranks.values()) + 1
            result_df['RANK'] = result_df['RANK'].fillna(last_rank)
        
        # Убеждаемся, что ранги целочисленные
        result_df['RANK'] = result_df['RANK'].astype(int)
        
        # Сохраняем результат
        result_df.to_csv('token_rankings.csv', index=False)
        
        print(f"Файл token_rankings.csv успешно создан")
        print("\nПервые 10 токенов по рангу:")
        print(result_df.head(10))
        
        return result_df
        
    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден - {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    create_token_rankings() 