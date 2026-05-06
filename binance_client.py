from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from config import Config

class BinanceTradingClient:
    def __init__(self):
        """Initialize Binance client connection [citation:2][citation:7]"""
        self.config = Config()
        
        # Create client with testnet mode
        if self.config.TESTNET:
            self.client = Client(
                self.config.API_KEY, 
                self.config.API_SECRET, 
                testnet=True
            )
            print("✅ Connected to Binance TESTNET")
        else:
            self.client = Client(self.config.API_KEY, self.config.API_SECRET)
            print("⚠️ Connected to Binance MAINNET - Real money!")
        
        self.symbol = self.config.SYMBOL
        self.logger = logging.getLogger(__name__)
    
    def get_account_balance(self):
        """Fetch account balance [citation:7]"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked
                    print(f"💰 Balance: {total:.2f} USDT (Free: {free:.2f})")
                    return {'free': free, 'locked': locked, 'total': total}
            return None
        except BinanceAPIException as e:
            print(f"❌ Error fetching balance: {e}")
            return None
    
    def get_market_data(self, limit=50):
        """Fetch real-time market data (OHLCV) [citation:7]"""
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.config.TIMEFRAME,
                limit=limit
            )
            
            # Parse klines data
            data = []
            for k in klines:
                data.append({
                    'timestamp': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            
            print(f"📊 Fetched {len(data)} candlesticks for {self.symbol}")
            return data
            
        except BinanceAPIException as e:
            print(f"❌ Error fetching market data: {e}")
            return None
    
    def get_current_price(self):
        """Get current price of symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"❌ Error fetching price: {e}")
            return None
    
    def place_market_order(self, side, quantity):
        """Place market order (immediate execution) [citation:2]"""
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,  # 'BUY' or 'SELL'
                type='MARKET',
                quantity=quantity
            )
            print(f"✅ {side} MARKET order executed: {quantity} {self.symbol}")
            return order
        except BinanceAPIException as e:
            print(f"❌ Market order failed: {e}")
            return None
    
    def place_limit_order(self, side, quantity, price):
        """Place limit order with specific price"""
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=str(price)
            )
            print(f"📝 {side} LIMIT order placed: {quantity} @ {price}")
            return order
        except BinanceAPIException as e:
            print(f"❌ Limit order failed: {e}")
            return None