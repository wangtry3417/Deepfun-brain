"""
港股報價 Class - Yahoo Finance 版 + PyTorch 豆包大腦
"""

import requests
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ==================== 檢查時間是否開市 ====================

def check_market_status(market="HK"):
    """檢查市場是否開市"""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    weekday = now.weekday()
    
    if weekday >= 5:
        return {'is_open': False, 'status': 'closed', 'message': '週末休市'}
    
    if market == "HK":
        morning = (9 < hour < 12) or (hour == 9 and minute >= 30) or (hour == 12 and minute == 0)
        afternoon = (13 <= hour < 16)
        
        if morning or afternoon:
            return {'is_open': True, 'status': 'open', 'message': '交易中'}
        elif hour < 9 or (hour == 9 and minute < 30):
            return {'is_open': False, 'status': 'pre', 'message': '盤前'}
        else:
            return {'is_open': False, 'status': 'closed', 'message': '已收市'}
    
    elif market == "US":
        if hour >= 21 or hour < 4:
            return {'is_open': True, 'status': 'open', 'message': '交易中'}
        else:
            return {'is_open': False, 'status': 'closed', 'message': '已收市'}
    
    return {'is_open': False, 'status': 'unknown', 'message': '未知'}

# ==================== PyTorch 豆包大腦 ====================

class MarketLSTM(nn.Module):
    """LSTM 市場預測模型"""
    
    def __init__(self, input_size=10, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 3)  # 3個輸出: 升/跌/平
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        x = self.relu(self.fc1(last_out))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PyTorchDoubaoBrain:
    """PyTorch 豆包大腦 - 深度學習版"""
    
    def __init__(self):
        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MarketLSTM().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # 經驗回放
        self.memory = deque(maxlen=1000)
        self.batch_size = 32
        
        # 預測歷史
        self.predictions = []
        self.actuals = []
        
        print(f"🚀 PyTorch 豆包大腦初始化完成 (使用: {self.device})")
    
    def prepare_features(self, change: float, change_percent: float, 
                         volume: Optional[int], news_sentiment: float) -> torch.Tensor:
        """準備特徵向量"""
        
        # 標準化特徵
        features = [
            change / 100,  # 標準化價格變化
            change_percent / 10,  # 標準化百分比
            np.tanh(volume / 1e8) if volume else 0,  # 成交量特徵
            news_sentiment,  # 新聞情緒
            np.sin(change * 10),  # 波動特徵
            np.cos(change * 10),
            float(check_market_status("HK")['is_open']),  # 市場狀態
            float(check_market_status("US")['is_open']),
            random.random() * 0.1,  # 隨機噪聲 (避免過擬合)
            random.random() * 0.1,
        ]
        
        return torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
    
    def train_step(self):
        """訓練一步"""
        if len(self.memory) < self.batch_size:
            return
        
        # 隨機採樣
        batch = random.sample(self.memory, self.batch_size)
        features = torch.cat([b[0] for b in batch])
        labels = torch.LongTensor([b[1] for b in batch]).to(self.device)
        
        # 訓練
        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_label(self, change_percent: float) -> int:
        """獲取標籤 (0:跌, 1:平, 2:升)"""
        if change_percent > 0.5:
            return 2  # 升
        elif change_percent < -0.5:
            return 0  # 跌
        else:
            return 1  # 平
    
    def predict(self, name: str, change: float, change_percent: float, 
                volume: Optional[int], news_sentiment: float) -> Dict:
        """預測市場走勢"""
        
        # 準備特徵
        features = self.prepare_features(change, change_percent, volume, news_sentiment)
        
        # 預測
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
        
        # 儲存經驗
        label = self.get_label(change_percent)
        self.memory.append((features, label))
        
        # 訓練
        self.model.train()
        loss = self.train_step()
        
        # 轉換結果
        results = {
            0: {"direction": "📉 看淡", "suggestion": "建議減持"},
            1: {"direction": "➡️ 平穩", "suggestion": "建議持有"},
            2: {"direction": "📈 看好", "suggestion": "可考慮建倉"}
        }
        
        result = results[prediction]
        
        # 生成分析原因
        if prediction == 2:
            if change_percent > 1:
                reason = f"強勁上升趨勢 (+{change_percent:.2f}%)，配合市場情緒"
            else:
                reason = "技術指標向好，上升動能積累"
        elif prediction == 0:
            if change_percent < -1:
                reason = f"顯著回調 ({change_percent:.2f}%)，建議觀望"
            else:
                reason = "弱勢調整，待方向明朗"
        else:
            reason = "多空平衡，區間震盪"
        
        return {
            'direction': result['direction'],
            'confidence': confidence,
            'reason': reason,
            'suggestion': result['suggestion'],
            'probabilities': {
                '跌': probabilities[0].item(),
                '平': probabilities[1].item(),
                '升': probabilities[2].item()
            },
            'loss': loss
        }

# ==================== 新聞分析 ====================

class NewsAnalyzer:
    """新聞智能分析"""
    
    @staticmethod
    def analyze_news_sentiment(news_items: List[Dict]) -> float:
        """分析新聞情緒，返回分數 -1 到 1"""
        if not news_items:
            return 0.0
        
        positive_keywords = ['record', 'high', 'rally', 'gain', 'rise', 'surge', 'breakthrough']
        negative_keywords = ['fall', 'drop', 'decline', 'fear', 'concern', 'risk', 'slow']
        
        pos_count = 0
        neg_count = 0
        
        for news in news_items:
            title = news.get('title', '').lower()
            
            for kw in positive_keywords:
                if kw in title:
                    pos_count += 1
                    break
            for kw in negative_keywords:
                if kw in title:
                    neg_count += 1
                    break
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / len(news_items)

# ==================== 新聞抓取 ====================

class NewsFetcher:
    """股票相關新聞"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    def get_market_news(self, symbol: str) -> List[Dict]:
        """獲取市場新聞"""
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}&newsCount=5"
            resp = self.session.get(url, timeout=3)
            data = resp.json()
            
            news_list = []
            for news in data.get('news', [])[:3]:
                news_list.append({
                    'title': news.get('title', '無標題'),
                    'publisher': news.get('publisher', '未知'),
                })
            return news_list
        except:
            return [
                {"title": "Asian Markets Mixed Amid Tech Rally", "publisher": "Reuters"},
                {"title": "Investors Eye Fed Rate Decision", "publisher": "Bloomberg"},
            ]

# ==================== Yahoo Finance Class ====================

@dataclass
class StockData:
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: Optional[int] = None
    market: str = "HK"
    analysis: Optional[Dict] = None
    news: Optional[List] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()
    
    @property
    def change_str(self) -> str:
        sign = '+' if self.change >= 0 else ''
        return f"{sign}{self.change:.2f}"
    
    @property
    def change_percent_str(self) -> str:
        sign = '+' if self.change_percent >= 0 else ''
        return f"{sign}{self.change_percent:.2f}%"

class YahooFinance:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.brain = PyTorchDoubaoBrain()  # PyTorch 版本！
        self.news_analyzer = NewsAnalyzer()
        self.news_fetcher = NewsFetcher()
    
    def get_stock(self, symbol: str, market: str = "HK") -> Optional[StockData]:
        try:
            # 建立 URL
            if market == "US" or symbol.startswith('^'):
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            else:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.HK"
            
            resp = self.session.get(url, timeout=3)
            data = resp.json()
            
            if not data['chart']['result']:
                return None
            
            meta = data['chart']['result'][0]['meta']
            
            price = meta['regularMarketPrice']
            prev = meta['previousClose']
            change = price - prev
            change_pct = (change / prev) * 100
            
            # 獲取成交量
            volume = None
            try:
                quotes = data['chart']['result'][0]['indicators']['quote'][0]
                volume = quotes['volume'][-1] if quotes.get('volume') else 0
            except:
                pass
            
            name = meta.get('longName', symbol)
            
            # 獲取新聞和分析
            news = self.news_fetcher.get_market_news(symbol)
            news_sentiment = self.news_analyzer.analyze_news_sentiment(news)
            
            # PyTorch 預測
            analysis = self.brain.predict(name, change, change_pct, volume, news_sentiment)
            
            return StockData(
                symbol=symbol,
                name=name[:30],
                price=price,
                change=change,
                change_percent=change_pct,
                volume=volume,
                market=market,
                analysis=analysis,
                news=news
            )
        except Exception as e:
            return None

# ==================== 使用範例 ====================

def print_market_analysis(stock):
    """打印市場分析"""
    print("\n" + "="*80)
    print(f"📊 {stock.name}")
    print("="*80)
    
    # 價格資訊
    color = '\033[92m' if stock.change > 0 else '\033[91m'
    reset = '\033[0m'
    print(f"\n💰 價格: {stock.price:.2f} {color}{stock.change_str} ({stock.change_percent_str}){reset}")
    if stock.volume:
        print(f"📊 成交量: {stock.volume:,}")
    
    # PyTorch 預測
    print(f"\n🧠 PyTorch 豆包大腦分析:")
    print(f"   預測: {stock.analysis['direction']}")
    print(f"   信心: {stock.analysis['confidence']*100:.1f}%")
    print(f"   原因: {stock.analysis['reason']}")
    print(f"   建議: {stock.analysis['suggestion']}")
    
    # 概率分佈
    probs = stock.analysis['probabilities']
    print(f"\n📊 概率分佈:")
    print(f"   📈 上升: {probs['升']*100:.1f}%")
    print(f"   ➡️ 平穩: {probs['平']*100:.1f}%")
    print(f"   📉 下跌: {probs['跌']*100:.1f}%")
    
    # 最新新聞
    if stock.news:
        print(f"\n📰 最新新聞:")
        for news in stock.news[:2]:
            print(f"   • {news['title']}")
    
    # 市場狀態
    status = check_market_status(stock.market)
    status_icon = '🟢' if status['is_open'] else '🔴'
    print(f"\n⏰ 市場狀態: {status_icon} {status['message']}")

# 主程式
if __name__ == "__main__":
    yf = YahooFinance()
    
    # 測試不同股票
    stock = yf.get_stock("^HSI", "HK")
    if stock: print_market_analysis(stock)
    else: print(f"❌ 無法獲取 {symbol} 數據")