import pandas as pd

def read_training_tokens():
    # Читаем parquet файл
    df = pd.read_parquet('data/database/training_tokens.parquet')
    
    # Выводим первые несколько строк
    print("\nПервые 5 строк данных:")
    print(df.head())
    
    # Выводим информацию о датасете
    print("\nИнформация о датасете:")
    print(df.info())
    
    # Выводим статистическое описание числовых колонок
    print("\nСтатистическое описание:")
    print(df.describe())
    
    # Выводим список колонок
    print("\nСписок колонок:")
    print(df.columns.tolist())
    
    return df

if __name__ == "__main__":
    try:
        df = read_training_tokens()
    except FileNotFoundError:
        print("Ошибка: Файл 'training_tokens.parquet' не найден в указанной директории")
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {str(e)}") 