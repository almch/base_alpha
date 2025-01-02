from blockchain.interactions.base_connection import BaseConnection
from data.collectors.uniswap_collector import UniswapCollector
from data.processors.token_processor import TokenProcessor

def main():
    # Проверяем подключение к Base
    base_conn = BaseConnection()
    if not base_conn.check_connection():
        print("Ошибка подключения к сети Base")
        return
    
    # Собираем данные из Uniswap
    collector = UniswapCollector()
    tokens_data = collector.get_top_tokens()
    
    # Обрабатываем данные
    processor = TokenProcessor()
    df = processor.process_token_data(tokens_data)
    
    # Выводим результаты
    print("\nТоп-10 торгуемых токенов на Uniswap (Base) за последние 24 часа:")
    print("=" * 80)
    for _, row in df.iterrows():
        print(f"Токен: {row['symbol']} ({row['name']})")
        print(f"Объем торгов: ${row['volumeUSD']:,.2f}")
        print(f"TVL: ${row['totalValueLockedUSD']:,.2f}")
        print("-" * 80)

if __name__ == "__main__":
    main() 