import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class TokenBacktester:
    def __init__(self, initial_balance=6000, tokens_per_batch=5, investment_per_token=200):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.tokens_per_batch = tokens_per_batch
        self.investment_per_token = investment_per_token
        self.positions = {}  # {date: {token: {'amount': amount, 'price': price}}}
        self.daily_balance = []
        self.trades_history = []
        
    def simulate_trading(self, rankings_df, price_data_df):
        """
        Симуляция торговли на основе ежедневных рекомендаций
        
        Args:
            rankings_df: DataFrame с ежедневными рангами токенов
            price_data_df: DataFrame с ценами токенов
        """
        # Сортируем данные по дате
        dates = sorted(rankings_df['date'].unique())
        start_date = dates[7]  # Начинаем с 8-го дня
        
        print("Начало бэктестинга...")
        print(f"Начальный баланс: ${self.initial_balance:,.2f}")
        
        for current_date in dates[7:]:
            # Продаем токены, купленные 7 дней назад
            if current_date in self.positions:
                self._sell_positions(current_date, price_data_df)
            
            # Покупаем новые токены
            daily_rankings = rankings_df[rankings_df['date'] == current_date]
            top_tokens = daily_rankings.nsmallest(self.tokens_per_batch, 'RANK')
            self._buy_positions(current_date, top_tokens, price_data_df)
            
            # Записываем ежедневный баланс
            self._update_daily_balance(current_date, price_data_df)
        
        self._generate_report()
        
    def _buy_positions(self, date, top_tokens, price_data_df):
        """Покупка токенов"""
        self.positions[date] = {}
        
        for _, token_row in top_tokens.iterrows():
            token = token_row['ADDRESS']
            price = self._get_token_price(token, date, price_data_df)
            
            if price > 0:
                amount = self.investment_per_token / price
                self.positions[date][token] = {
                    'amount': amount,
                    'price': price
                }
                self.current_balance -= self.investment_per_token
                
                self.trades_history.append({
                    'date': date,
                    'token': token,
                    'action': 'BUY',
                    'amount': amount,
                    'price': price,
                    'value': self.investment_per_token
                })
    
    def _sell_positions(self, buy_date, price_data_df):
        """Продажа токенов"""
        if buy_date not in self.positions:
            return
            
        for token, position in self.positions[buy_date].items():
            current_price = self._get_token_price(token, buy_date + timedelta(days=7), price_data_df)
            value = position['amount'] * current_price
            self.current_balance += value
            
            self.trades_history.append({
                'date': buy_date + timedelta(days=7),
                'token': token,
                'action': 'SELL',
                'amount': position['amount'],
                'price': current_price,
                'value': value
            })
        
        del self.positions[buy_date]
    
    def _get_token_price(self, token, date, price_data_df):
        """Получение цены токена на определенную дату"""
        price_data = price_data_df[
            (price_data_df['token_address'] == token) & 
            (price_data_df['date'] == date)
        ]
        return price_data['price'].iloc[0] if not price_data.empty else 0
    
    def _update_daily_balance(self, date, price_data_df):
        """Обновление ежедневного баланса"""
        total_value = self.current_balance
        
        # Добавляем стоимость открытых позиций
        for buy_date, positions in self.positions.items():
            for token, position in positions.items():
                current_price = self._get_token_price(token, date, price_data_df)
                total_value += position['amount'] * current_price
        
        self.daily_balance.append({
            'date': date,
            'balance': total_value
        })
    
    def _generate_report(self):
        """Генерация отчета о результатах торговли"""
        final_balance = self.daily_balance[-1]['balance']
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        print("\nРезультаты бэктестинга:")
        print(f"Начальный баланс: ${self.initial_balance:,.2f}")
        print(f"Конечный баланс: ${final_balance:,.2f}")
        print(f"Общая доходность: {total_return:.2f}%")
        
        # Визуализация баланса
        self._plot_balance_history()
        self._plot_token_returns()
        
        # Сохранение истории торгов
        self._save_trade_history()
    
    def _plot_balance_history(self):
        """Построение графика изменения баланса"""
        df = pd.DataFrame(self.daily_balance)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['balance'])
        plt.title('История баланса портфеля')
        plt.xlabel('Дата')
        plt.ylabel('Баланс ($)')
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Сохраняем график
        plt.savefig('balance_history.png', bbox_inches='tight')
        plt.close()
    
    def _plot_token_returns(self):
        """Построение графика доходности по токенам"""
        trades_df = pd.DataFrame(self.trades_history)
        token_returns = []
        
        for token in trades_df['token'].unique():
            token_trades = trades_df[trades_df['token'] == token]
            buys = token_trades[token_trades['action'] == 'BUY']
            sells = token_trades[token_trades['action'] == 'SELL']
            
            if not buys.empty and not sells.empty:
                total_invested = buys['value'].sum()
                total_returned = sells['value'].sum()
                returns = (total_returned - total_invested) / total_invested * 100
                token_returns.append({'token': token, 'return': returns})
        
        if token_returns:
            returns_df = pd.DataFrame(token_returns)
            returns_df = returns_df.sort_values('return', ascending=False)
            
            plt.figure(figsize=(12, 6))
            plt.bar(returns_df['token'], returns_df['return'])
            plt.title('Доходность по токенам')
            plt.xlabel('Токен')
            plt.ylabel('Доходность (%)')
            plt.xticks(rotation=90)
            
            plt.savefig('token_returns.png', bbox_inches='tight')
            plt.close()
    
    def _save_trade_history(self):
        """Сохранение истории торгов в CSV"""
        trades_df = pd.DataFrame(self.trades_history)
        trades_df.to_csv('trade_history.csv', index=False)

def run_backtest(rankings_file, price_data_file):
    """
    Запуск бэктестинга
    
    Args:
        rankings_file: путь к файлу с ежедневными рангами
        price_data_file: путь к файлу с ценами токенов
    """
    # Загрузка данных
    rankings_df = pd.read_csv(rankings_file)
    price_data_df = pd.read_csv(price_data_file)
    
    # Создание и запуск бэктестера
    backtester = TokenBacktester()
    backtester.simulate_trading(rankings_df, price_data_df)

if __name__ == "__main__":
    run_backtest('daily_rankings.csv', 'token_prices.csv') 