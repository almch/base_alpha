from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class TokenData(Base):
    __tablename__ = 'token_data'
    
    id = Column(String, primary_key=True)
    symbol = Column(String)
    name = Column(String)
    volumeUSD = Column(Float)
    totalValueLockedUSD = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine('sqlite:///data/database/uniswap_data.db')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_tokens_data(self, tokens_df):
        for _, row in tokens_df.iterrows():
            token_data = TokenData(
                id=row['id'],
                symbol=row['symbol'],
                name=row['name'],
                volumeUSD=row['volumeUSD'],
                totalValueLockedUSD=row['totalValueLockedUSD']
            )
            self.session.merge(token_data)
        self.session.commit() 