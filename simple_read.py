import pandas as pd

# Читаем parquet файл
df = pd.read_parquet('data/database/dex_swaps_1w.parquet')

# Выводим все данные
print(df) 