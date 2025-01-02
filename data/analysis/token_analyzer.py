import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TokenAnalyzer:
    def __init__(self):
        self.dex_swaps_df = None
        self.training_tokens_df = None
        
    def load_data(self, dex_swaps_path, training_tokens_path):
        """Загрузка данных из CSV файлов"""
        try:
            print("Загрузка данных...")
            self.dex_swaps_df = pd.read_csv(dex_swaps_path)
            self.training_tokens_df = pd.read_csv(training_tokens_path)
            
            # Конвертируем timestamp в datetime
            self.dex_swaps_df['timestamp'] = pd.to_datetime(self.dex_swaps_df['timestamp'])
            print(f"Загружено {len(self.dex_swaps_df)} свапов")
            
        except Exception as e:
            print(f"Ошибка при загрузке данных: {str(e)}")
            
    def analyze_volume_changes(self, days=7):
        """Анализ изменения объемов торгов за последние N дней"""
        try:
            print(f"Анализ изменения объемов за {days} дней...")
            
            # Получаем текущую дату из данных
            latest_date = self.dex_swaps_df['timestamp'].max()
            week_ago = latest_date - timedelta(days=days)
            
            # Группируем данные по токенам и периодам
            recent_volume = self.dex_swaps_df[
                self.dex_swaps_df['timestamp'] > week_ago
            ].groupby('token_address')['amount_usd'].sum()
            
            previous_volume = self.dex_swaps_df[
                (self.dex_swaps_df['timestamp'] <= week_ago) & 
                (self.dex_swaps_df['timestamp'] > week_ago - timedelta(days=days))
            ].groupby('token_address')['amount_usd'].sum()
            
            # Вычисляем изменение объема
            volume_change = pd.DataFrame({
                'recent_volume': recent_volume,
                'previous_volume': previous_volume
            }).fillna(0)
            
            volume_change['volume_change_pct'] = (
                (volume_change['recent_volume'] - volume_change['previous_volume']) / 
                volume_change['previous_volume'] * 100
            ).fillna(0)
            
            # Сортируем по изменению объема
            volume_change = volume_change.sort_values('volume_change_pct', ascending=False)
            
            # Добавляем информацию о токенах
            result = pd.merge(
                volume_change,
                self.training_tokens_df[['address', 'symbol']],
                left_index=True,
                right_on='address'
            )
            
            return result
            
    def print_top_tokens(self, result_df, top_n=10):
        """Вывод топ N токенов"""
        print(f"\nТоп {top_n} токенов по росту объема торгов:")
        print("-" * 80)
        print("Символ | Изменение объема (%) | Текущий объем ($) | Предыдущий объем ($)")
        print("-" * 80)
        
        for _, row in result_df.head(top_n).iterrows():
            print(f"{row['symbol']:6} | {row['volume_change_pct']:16.2f}% | {row['recent_volume']:15.2f} | {row['previous_volume']:18.2f}")

def test_analyzer():
    analyzer = TokenAnalyzer()
    analyzer.load_data(
        'data/dex_swaps_3m.csv',
        'data/training_tokens.csv'
    )
    results = analyzer.analyze_volume_changes(days=7)
    analyzer.print_top_tokens(results)

if __name__ == "__main__":
    test_analyzer() 