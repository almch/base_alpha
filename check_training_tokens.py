import pandas as pd

def check_tokens():
    tokens_df = pd.read_parquet('data/database/training_tokens.parquet')
    print("Колонки в training_tokens:")
    print(tokens_df.columns.tolist())
    print("\nПервые несколько строк:")
    print(tokens_df.head())

if __name__ == "__main__":
    check_tokens() 