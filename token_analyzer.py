import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine

class TokenAnalyzer:
    def __init__(self, db_connection_string):
        self.engine = create_engine(db_connection_string)
        
    def load_data(self, days_lookback=7):
        """Загружает данные из всех необходимых таблиц"""
        # Получаем дату для фильтрации
        cutoff_date = datetime.now() - timedelta(days=days_lookback)
        
        # Загружаем базовые токены для анализа
        self.training_tokens = pd.read_sql(
            "SELECT * FROM training_tokens",
            self.engine
        )
        
        # Загружаем свапы на DEX
        self.dex_swaps = pd.read_sql(
            "SELECT * FROM dex_swaps_1w WHERE timestamp >= :cutoff",
            self.engine,
            params={'cutoff': cutoff_date}
        )
        
        # Загружаем трансферы токенов
        self.token_transfers = pd.read_sql(
            "SELECT * FROM token_transfers_1w WHERE timestamp >= :cutoff",
            self.engine,
            params={'cutoff': cutoff_date}
        )
        
        # Загружаем нативные трансферы ETH
        self.native_transfers = pd.read_sql(
            "SELECT * FROM native_transfers_1w WHERE timestamp >= :cutoff",
            self.engine,
            params={'cutoff': cutoff_date}
        )

    def analyze_token_activity(self, token_address):
        """Анализирует активность конкретного токена"""
        # Анализ DEX свапов
        token_swaps = self.dex_swaps[self.dex_swaps['token_address'] == token_address]
        
        # Анализ трансферов токена
        token_transfers = self.token_transfers[self.token_transfers['token_address'] == token_address]
        
        # Находим связанные кошельки
        related_wallets = set(token_transfers['from_address'].unique()) | \
                         set(token_transfers['to_address'].unique())
        
        # Анализ ETH трансферов связанных кошельков
        related_eth_transfers = self.native_transfers[
            (self.native_transfers['from_address'].isin(related_wallets)) |
            (self.native_transfers['to_address'].isin(related_wallets))
        ]
        
        return {
            'total_swaps': len(token_swaps),
            'unique_traders': len(set(token_swaps['trader_address'])),
            'total_volume_usd': token_swaps['amount_usd'].sum(),
            'transfer_count': len(token_transfers),
            'active_wallets': len(related_wallets),
            'related_eth_transfers': len(related_eth_transfers)
        }

    def get_token_metrics(self):
        """Получает метрики для всех токенов из training_tokens"""
        metrics = []
        
        for _, token in self.training_tokens.iterrows():
            token_metrics = self.analyze_token_activity(token['address'])
            metrics.append({
                'address': token['address'],
                'symbol': token['symbol'],
                **token_metrics
            })
            
        return pd.DataFrame(metrics)

    def find_similar_tokens(self, token_address, n=5):
        """Находит похожие токены на основе метрик активности"""
        all_metrics = self.get_token_metrics()
        
        # Нормализация метрик
        numeric_columns = ['total_swaps', 'unique_traders', 'total_volume_usd', 
                         'transfer_count', 'active_wallets', 'related_eth_transfers']
        
        normalized_metrics = all_metrics[numeric_columns].apply(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # Находим токен для сравнения
        base_token = normalized_metrics[all_metrics['address'] == token_address].iloc[0]
        
        # Вычисляем евклидово расстояние до всех токенов
        distances = normalized_metrics.apply(
            lambda x: np.sqrt(((x - base_token) ** 2).sum()),
            axis=1
        )
        
        # Получаем самые похожие токены
        similar_indices = distances.nsmallest(n + 1).index[1:]  # Исключаем сам токен
        
        return all_metrics.iloc[similar_indices] 