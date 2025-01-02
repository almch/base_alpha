import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
from datetime import datetime

class UniswapTracker:
    def __init__(self):
        # Используем официальный endpoint для Uniswap V2
        self.graph_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"
        print(f"Инициализация с URL: {self.graph_url}")
        
    def get_latest_swaps(self, limit=3):
        # Используем правильный формат GraphQL запроса
        query = """
        {
          swaps(
            first: %d,
            orderBy: timestamp,
            orderDirection: desc
          ) {
            timestamp
            pair {
              token0 {
                symbol
                decimals
              }
              token1 {
                symbol
                decimals
              }
            }
            amount0In
            amount0Out
            amount1In
            amount1Out
            amountUSD
          }
        }
        """ % limit

        try:
            print("Отправка запроса к API...")
            response = requests.post(
                self.graph_url,
                json={'query': query},
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            print(f"Получен ответ. Статус код: {response.status_code}")
            print(f"Ответ API: {response.text[:200]}...")
            
            if response.status_code == 200:
                data = response.json()
                if 'errors' in data:
                    print(f"GraphQL ошибки: {data['errors']}")
                    return []
                
                swaps = data.get('data', {}).get('swaps', [])
                print(f"Найдено свапов: {len(swaps)}")
                return swaps
            else:
                print(f"Ошибка API: {response.status_code}")
                print(f"Текст ошибки: {response.text}")
                return []
                
        except Exception as e:
            print(f"Исключение при получении свапов: {str(e)}")
            return []

    def print_swaps(self, swaps):
        print("\nПоследние свапы Uniswap:")
        print("-" * 50)
        
        if not swaps:
            print("Свапы не найдены!")
            return
            
        for swap in swaps:
            try:
                # Определяем направление свапа
                amount0_in = float(swap['amount0In'])
                amount0_out = float(swap['amount0Out'])
                amount1_in = float(swap['amount1In'])
                amount1_out = float(swap['amount1Out'])
                
                # Если amount0In > 0, значит свап идет от token0 к token1
                if amount0_in > 0:
                    from_token = swap['pair']['token0']['symbol']
                    to_token = swap['pair']['token1']['symbol']
                    from_amount = amount0_in
                    to_amount = amount1_out
                else:
                    from_token = swap['pair']['token1']['symbol']
                    to_token = swap['pair']['token0']['symbol']
                    from_amount = amount1_in
                    to_amount = amount0_out
                
                timestamp = datetime.fromtimestamp(int(swap['timestamp']))
                print(f"Время: {timestamp.strftime('%H:%M:%S')}")
                print(f"{from_token}: {from_amount:.4f} → {to_token}: {to_amount:.4f}")
                print(f"Объем: ${float(swap['amountUSD']):.2f}")
                print("-" * 50)
            except Exception as e:
                print(f"Ошибка при обработке свапа: {str(e)}")
                print(f"Данные свапа: {swap}")
                print("-" * 50)

def test_uniswap_tracker():
    print("Начало теста...")
    tracker = UniswapTracker()
    swaps = tracker.get_latest_swaps()
    tracker.print_swaps(swaps)
    print("Тест завершен.")

if __name__ == "__main__":
    test_uniswap_tracker()