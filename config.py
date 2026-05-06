import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Configuration
    API_KEY = os.getenv('BINANCE_API_KEY')
    API_SECRET = os.getenv('BINANCE_SECRET_KEY')
    TESTNET = os.getenv('TESTNET', 'true').lower() == 'true'
    
    # Trading Parameters
    SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')
    QUANTITY = float(os.getenv('QUANTITY', '0.001'))  # BTC amount to trade
    TIMEFRAME = '5m'  # Candlestick timeframe
    
    # Strategy Parameters (RSI + EMA)
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    EMA_FAST = 9
    EMA_SLOW = 21
    
    # Risk Management [citation:4]
    STOP_LOSS_PERCENT = 0.02  # 2% stop loss
    TAKE_PROFIT_PERCENT = 0.04  # 4% take profit
    MAX_POSITION_SIZE = 0.005  # Max BTC per trade
    MAX_DAILY_TRADES = 10  # Limit trades per day
    CAPITAL_ALLOCATION = 0.25  # Use 25% of balance per trade
    
    # Bot Behavior
    CHECK_INTERVAL = 60  # Check every 60 seconds
    LOG_FILE = 'logs/trades.json'