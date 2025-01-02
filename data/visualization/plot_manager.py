import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class PlotManager:
    @staticmethod
    def create_volume_bar_chart(df):
        fig = px.bar(
            df,
            x='symbol',
            y='volumeUSD',
            title='Объем торгов токенов на Uniswap (Base)',
            labels={'volumeUSD': 'Объем торгов (USD)', 'symbol': 'Токен'}
        )
        fig.write_html('data/visualization/volume_chart.html')
    
    @staticmethod
    def create_tvl_pie_chart(df):
        fig = px.pie(
            df,
            values='totalValueLockedUSD',
            names='symbol',
            title='Распределение TVL на Uniswap (Base)'
        )
        fig.write_html('data/visualization/tvl_chart.html') 