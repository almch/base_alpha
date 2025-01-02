import pandas as pd

def check_data_structure():
    try:
        # Читаем данные
        swaps_df = pd.read_parquet('data/database/dex_swaps_1w.parquet')
        tokens_df = pd.read_parquet('data/database/training_tokens.parquet')
        
        print("\nКолонки в dex_swaps_1w:")
        print(swaps_df.columns.tolist())
        
        print("\nПример данных dex_swaps_1w:")
        print(swaps_df.head())
        
        print("\nКолонки в training_tokens:")
        print(tokens_df.columns.tolist())
        
        print("\nПример данных training_tokens:")
        print(tokens_df.head())
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    check_data_structure() 