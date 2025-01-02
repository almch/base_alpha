import pandas as pd

class TokenProcessor:
    @staticmethod
    def process_token_data(tokens_data):
        if not tokens_data:
            print("Нет данных для обработки")
            return pd.DataFrame()
            
        try:
            df = pd.DataFrame(tokens_data)
            
            # Преобразуем строковые значения в числовые
            df['volumeUSD'] = pd.to_numeric(df['volumeUSD'], errors='coerce')
            df['totalValueLockedUSD'] = pd.to_numeric(df['totalValueLockedUSD'], errors='coerce')
            
            # Заменяем NaN на 0
            df = df.fillna(0)
            
            # Округляем значения для лучшей читаемости
            df['volumeUSD'] = df['volumeUSD'].round(2)
            df['totalValueLockedUSD'] = df['totalValueLockedUSD'].round(2)
            
            return df
            
        except Exception as e:
            print(f"Ошибка при обработке данных: {str(e)}")
            return pd.DataFrame() 