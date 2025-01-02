from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
from data.database.db_manager import TokenData
from data.collectors.uniswap_collector import UniswapCollector
from data.processors.token_processor import TokenProcessor

app = FastAPI()

# Создаем подключение к БД
engine = create_engine('sqlite:///data/database/uniswap_data.db')
Session = sessionmaker(bind=engine)

@app.get("/tokens/top")
async def get_top_tokens():
    try:
        collector = UniswapCollector()
        tokens_data = collector.get_top_tokens()
        processor = TokenProcessor()
        df = processor.process_token_data(tokens_data)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tokens/history")
async def get_tokens_history():
    session = Session()
    try:
        results = session.query(TokenData).all()
        data = [{
            'symbol': r.symbol,
            'name': r.name,
            'volumeUSD': r.volumeUSD,
            'totalValueLockedUSD': r.totalValueLockedUSD,
            'timestamp': r.timestamp
        } for r in results]
        return data
    finally:
        session.close() 