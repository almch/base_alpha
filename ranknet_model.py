import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import os

class TokenDataset(Dataset):
    """
    Кастомный класс для работы с данными токенов в формате PyTorch.
    Наследуется от torch.utils.data.Dataset для совместимости с DataLoader.
    """
    def __init__(self, features, labels=None):
        # Преобразуем numpy массивы в PyTorch тензоры
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        # Возвращает количество образцов в датасете
        return len(self.features)

    def __getitem__(self, idx):
        # Возвращает пару (признаки, метка) по индексу
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class RankNet(nn.Module):
    """
    Нейронная сеть для ранжирования токенов.
    Архитектура: 3 полносвязных слоя с ReLU активацией и dropout.
    """
    def __init__(self, input_size):
        super(RankNet, self).__init__()
        self.net = nn.Sequential(
            # Первый слой: input_size -> 64 нейрона
            nn.Linear(input_size, 64),
            nn.ReLU(),  # Функция активации
            nn.Dropout(0.2),  # Dropout для предотвращения переобучения
            
            # Второй слой: 64 -> 32 нейрона
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Выходной слой: 32 -> 1 (скор для ранжирования)
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Прямой проход через сеть
        return self.net(x)

def prepare_data(swaps_df, tokens_df):
    """
    Подготовка признаков для модели из сырых данных о свапах.
    
    Args:
        swaps_df: DataFrame со свапами
        tokens_df: DataFrame с информацией о токенах
    
    Returns:
        DataFrame с подготовленными признаками
    """
    token_features = []
    
    # Обрабатываем каждый токен отдельно
    for token in tokens_df['TOKEN_ADDRESS']:
        # Фильтруем свапы для текущего токена
        token_in_swaps = swaps_df[swaps_df['TOKEN_IN'] == token]
        token_out_swaps = swaps_df[swaps_df['TOKEN_OUT'] == token]
        
        # Собираем признаки для токена
        features = {
            'TOKEN_ADDRESS': token,
            # Общий объем входящих и исходящих свапов
            'total_volume_in': token_in_swaps['AMOUNT_IN_USD'].sum(),
            'total_volume_out': token_out_swaps['AMOUNT_OUT_USD'].sum(),
            # Количество сделок
            'num_trades_in': len(token_in_swaps),
            'num_trades_out': len(token_out_swaps),
            # Количество уникальных трейдеров
            'unique_traders': len(set(token_in_swaps['SENDER'].unique()) | 
                                set(token_out_swaps['SENDER'].unique())),
            # Средний размер сделки
            'avg_trade_size_in': token_in_swaps['AMOUNT_IN_USD'].mean() if len(token_in_swaps) > 0 else 0,
            'avg_trade_size_out': token_out_swaps['AMOUNT_OUT_USD'].mean() if len(token_out_swaps) > 0 else 0,
        }
        
        token_features.append(features)
    
    features_df = pd.DataFrame(token_features)
    
    # Список числовых признаков для нормализации
    feature_columns = ['total_volume_in', 'total_volume_out', 'num_trades_in', 
                      'num_trades_out', 'unique_traders', 'avg_trade_size_in', 
                      'avg_trade_size_out']
    
    # Нормализация признаков (приведение к стандартному нормальному распределению)
    scaler = StandardScaler()
    features_df[feature_columns] = scaler.fit_transform(features_df[feature_columns])
    
    return features_df

def calculate_price_growth(token, price_history_df):
    """
    Вычисляет рост цены токена за период
    """
    token_prices = price_history_df[price_history_df['token_address'] == token]
    if len(token_prices) < 2:
        return 0
    
    initial_price = token_prices['price'].iloc[0]
    final_price = token_prices['price'].iloc[-1]
    
    if initial_price == 0:
        return 0
    
    return (final_price - initial_price) / initial_price

def calculate_trading_volume(token, volume_data):
    """
    Вычисляет нормализованный объем торгов
    """
    token_volume = volume_data[volume_data['TOKEN_IN'] == token]
    total_volume = token_volume['AMOUNT_IN_USD'].sum()
    
    # Добавляем объемы, где токен является исходящим
    token_volume_out = volume_data[volume_data['TOKEN_OUT'] == token]
    total_volume += token_volume_out['AMOUNT_OUT_USD'].sum()
    
    return total_volume

def calculate_social_score(token, social_data):
    """
    Вычисляет социальный скор на основе активности
    """
    if social_data is None:
        return 0
        
    token_social = social_data[social_data['token_address'] == token]
    if len(token_social) == 0:
        return 0
    
    # Можно добавить другие метрики социальной активности
    return token_social['activity_score'].mean()

def create_combined_labels(tokens_df, swaps_df, price_history_df=None, social_data=None):
    """
    Создает комбинированные метки для обучения
    
    Args:
        tokens_df: DataFrame с информацией о токенах
        swaps_df: DataFrame со свапами
        price_history_df: DataFrame с историей цен (опционально)
        social_data: DataFrame с социальными метриками (опционально)
    """
    labels = []
    
    for token in tokens_df['TOKEN_ADDRESS']:
        # Рост цены (если есть данные)
        price_growth = calculate_price_growth(token, price_history_df) if price_history_df is not None else 0
        
        # Объем торгов
        trading_volume = calculate_trading_volume(token, swaps_df)
        
        # Социальная активность (если есть данные)
        social_score = calculate_social_score(token, social_data)
        
        # Нормализуем метрики
        max_volume = swaps_df['AMOUNT_IN_USD'].max()
        normalized_volume = trading_volume / max_volume if max_volume > 0 else 0
        
        # Комбинируем метрики
        # Если нет данных о ценах и социальной активности, используем только объем торгов
        if price_history_df is None and social_data is None:
            combined_score = normalized_volume
        else:
            combined_score = (
                0.4 * price_growth +
                0.4 * normalized_volume +
                0.2 * social_score
            )
        
        labels.append(combined_score)
    
    return np.array(labels)

def train_epoch(model, train_loader, optimizer, criterion):
    """
    Обучение модели на одной эпохе
    """
    total_loss = 0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        
        # Создаем пары для сравнения
        batch_size = len(batch_features)
        i, j = torch.randint(0, batch_size, (2, batch_size))
        
        score_i = model(batch_features[i])
        score_j = model(batch_features[j])
        
        target = (batch_labels[i] > batch_labels[j]).float().reshape(-1, 1)
        
        diff = score_i - score_j
        loss = criterion(diff, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion):
    """
    Валидация модели
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_size = len(batch_features)
            i, j = torch.randint(0, batch_size, (2, batch_size))
            
            score_i = model(batch_features[i])
            score_j = model(batch_features[j])
            
            target = (batch_labels[i] > batch_labels[j]).float().reshape(-1, 1)
            
            diff = score_i - score_j
            loss = criterion(diff, target)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_ranknet_improved(features_df, swaps_df, num_epochs=100, batch_size=32):
    """
    Улучшенный процесс обучения с валидацией и мониторингом
    """
    features = features_df.drop(['TOKEN_ADDRESS'], axis=1).values
    labels = create_combined_labels(features_df, swaps_df)
    
    # Разделение на train/validation
    train_idx, val_idx = train_test_split(range(len(features)), test_size=0.2, random_state=42)
    
    train_dataset = TokenDataset(features[train_idx], labels[train_idx])
    val_dataset = TokenDataset(features[val_idx], labels[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = RankNet(input_size=features.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    best_model = None
    
    # История обучения для визуализации
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    print("Начало обучения...")
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        
        # Валидация
        model.eval()
        val_loss = validate_epoch(model, val_loader, criterion)
        
        # Сохраняем метрики
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            print(f'Epoch {epoch+1}: Новая лучшая модель! Val Loss: {val_loss:.4f}')
        
        # Обновляем learning rate
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Визуализация процесса обучения
    visualize_training(history)
    
    # Сохраняем модель напрямую
    torch.save(best_model, 'best_model.pth')
    
    return best_model

def cross_validate_model(features_df, swaps_df, n_splits=5):
    """
    Кросс-валидация модели
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(features_df)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # Обучаем модель
        model = train_ranknet_improved(
            features_df.iloc[train_idx].reset_index(drop=True),
            swaps_df
        )
        
        # Оцениваем качество
        val_features = features_df.iloc[val_idx].reset_index(drop=True)
        val_score = validate_epoch(
            model,
            DataLoader(TokenDataset(
                val_features.drop(['TOKEN_ADDRESS'], axis=1).values,
                create_combined_labels(val_features, swaps_df)
            )),
            nn.BCEWithLogitsLoss()
        )
        
        scores.append(val_score)
        print(f"Validation Score: {val_score:.4f}")
    
    print(f"\nСредний скор: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return np.mean(scores)

def visualize_training(history):
    """
    Визуализация процесса обучения
    """
    plt.figure(figsize=(15, 5))
    
    # График loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss over time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # График learning rate
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate over time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    # Сохраняем график
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = 'training_plots'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/training_history_{timestamp}.png')
    plt.close()

def get_token_rankings(model, features_df):
    """
    Получение финальных рангов токенов на основе обученной модели.
    
    Args:
        model: обученная модель RankNet
        features_df: DataFrame с признаками токенов
    
    Returns:
        DataFrame с рангами токенов
    """
    model.eval()  # Переключаем модель в режим оценки
    with torch.no_grad():  # Отключаем вычисление градиентов
        features = torch.FloatTensor(features_df.drop(['TOKEN_ADDRESS'], axis=1).values)
        scores = model(features).squeeze().numpy()
    
    # Создаем DataFrame с результатами
    rankings = pd.DataFrame({
        'ADDRESS': features_df['TOKEN_ADDRESS'],
        'RANK': (-scores).argsort().argsort() + 1  # Преобразуем скоры в ранги (1 - лучший)
    })
    
    return rankings

def load_parquet_files(directory_path, file_pattern):
    """
    Загружает и объединяет все parquet файлы из указанной директории
    """
    import glob
    import os
    
    print(f"Загрузка файлов из {directory_path} с паттерном {file_pattern}")
    
    # Получаем список всех parquet файлов в директории
    files = glob.glob(os.path.join(directory_path, f"*{file_pattern}*.parquet"))
    print(f"Найдено файлов: {len(files)}")
    
    # Загружаем и объединяем все файлы
    dfs = []
    for file in files:
        print(f"Загрузка файла: {os.path.basename(file)}")
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Объединяем все DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Общее количество записей: {len(combined_df)}")
    
    return combined_df

def main():
    """
    Основная функция для запуска всего процесса
    """
    print("Загрузка данных...")
    # Загружаем все файлы свапов за 3 месяца
    swaps_df = load_parquet_files(
        'data/database/dex_swaps_3m', 
        'dex_swaps_3m'
    )
    
    # Загружаем данные о токенах
    tokens_df = pd.read_parquet('data/database/training_tokens.parquet')
    
    print("Подготовка признаков...")
    features_df = prepare_data(swaps_df, tokens_df)
    
    print("Запуск кросс-валидации...")
    mean_score = cross_validate_model(features_df, swaps_df)
    
    print("\nОбучение финальной модели...")
    final_model = train_ranknet_improved(features_df, swaps_df)
    
    print("Получение финальных рангов...")
    rankings = get_token_rankings(final_model, features_df)
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rankings.to_csv(f'token_rankings_{timestamp}.csv', index=False)
    print(f"Ранги сохранены в файл token_rankings_{timestamp}.csv")
    print(f"Средний скор по кросс-валидации: {mean_score:.4f}")

if __name__ == "__main__":
    main() 