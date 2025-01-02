import pandas as pd

def analyze_token_volumes():
    try:
        # Читаем оба файла
        swaps_df = pd.read_parquet('data/database/dex_swaps_1w.parquet')
        tokens_df = pd.read_parquet('data/database/training_tokens.parquet')
        
        # Выводим информацию о структуре данных
        print("\nКолонки в dex_swaps_1w:")
        print(swaps_df.columns.tolist())
        
        print("\nКолонки в training_tokens:")
        print(tokens_df.columns.tolist())
        
        # Выводим первые несколько строк каждого датафрейма
        print("\nПример данных из dex_swaps_1w:")
        print(swaps_df.head())
        
        print("\nПример данных из training_tokens:")
        print(tokens_df.head())
        
    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден - {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    analyze_token_volumes() 