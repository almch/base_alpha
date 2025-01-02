import requests
import pandas as pd
from datetime import datetime, timedelta

class UniswapCollector:
    def __init__(self):
        self.graph_url = "https://api.thegraph.com/subgraphs/name/ianlapham/base-uniswap-v3"
        
    def get_top_tokens(self):
        query = """
        {
          tokens(
            first: 10,
            orderBy: volumeUSD,
            orderDirection: desc,
            where: {
              totalValueLockedUSD_gt: "100"
            }
          ) {
            id
            symbol
            name
            volumeUSD
            totalValueLockedUSD
            decimals
            derivedETH
          }
        }
        """
        
        try:
            headers = {
                'Content-Type': 'application/json',
            }
            response = requests.post(self.graph_url, json={'query': query}, headers=headers)
            
            # Отладочная информация
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'errors' in data:
                    print(f"GraphQL errors: {data['errors']}")
                    return []
                tokens = data.get('data', {}).get('tokens', [])
                if not tokens:
                    print("Не удалось получить данные о токенах")
                return tokens
            else:
                print(f"Ошибка HTTP: {response.status_code}")
                return []
            
        except Exception as e:
            print(f"Ошибка при получении данных: {str(e)}")
            return [] 