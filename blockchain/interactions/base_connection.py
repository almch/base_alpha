from web3 import Web3
from decouple import config

class BaseConnection:
    def __init__(self):
        # Используем RPC URL для Base
        self.w3 = Web3(Web3.HTTPProvider(config('BASE_RPC_URL')))
        
    def check_connection(self):
        try:
            is_connected = self.w3.is_connected()
            chain_id = self.w3.eth.chain_id
            latest_block = self.w3.eth.block_number
            
            print(f"Подключение: {'Успешно' if is_connected else 'Ошибка'}")
            print(f"Chain ID: {chain_id}")
            print(f"Последний блок: {latest_block}")
            
            return is_connected
        except Exception as e:
            print(f"Ошибка подключения: {str(e)}")
            return False 