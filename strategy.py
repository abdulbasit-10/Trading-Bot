import pandas as pd
import numpy as np
from config import Config

class TradingStrategy:
    def __init__(self):
        self.config = Config()
        self.position = None  # Track current position
        self.entry_price = 0
        self.trades = []
    
    def calculate_indicators(self, market_data):
        """
        Calculate RSI and EMA indicators [citation:3][citation:8]
        """
        df = pd.DataFrame(market_data)
        
        # Calculate RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate EMAs (Exponential Moving Averages)
        df['ema_fast'] = df['close'].ewm(span=self.config.EMA_FAST, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.config.EMA_SLOW, adjust=False).mean()
        
        return df
    
    def generate_signal(self, market_data):
        """
        Generate trading signals based on RSI + EMA confirmation [citation:3]
        
        BUY Signal: RSI < 30 (oversold) AND Fast EMA > Slow EMA (uptrend)
        SELL Signal: RSI > 70 (overbought) OR Fast EMA < Slow EMA (downtrend)
        """
        df = self.calculate_indicators(market_data)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        current_price = latest['close']
        rsi = latest['rsi']
        ema_fast = latest['ema_fast']
        ema_slow = latest['ema_slow']
        
        # Entry Conditions
        buy_signal = (
            rsi < self.config.RSI_OVERSOLD and  # Oversold condition
            ema_fast > ema_slow and              # Uptrend confirmation
            self.position is None                 # No open position
        )
        
        # Exit Conditions
        sell_signal = (
            self.position == 'LONG' and (
                rsi > self.config.RSI_OVERBOUGHT or  # Overbought
                ema_fast < ema_slow or                # Downtrend
                self.check_stop_loss(current_price) or
                self.check_take_profit(current_price)
            )
        )
        
        if buy_signal:
            signal = {
                'action': 'BUY',
                'price': current_price,
                'reason': f'RSI={rsi:.1f} (oversold), EMA_Fast={ema_fast:.0f} > EMA_Slow={ema_slow:.0f}',
                'timestamp': pd.Timestamp.now()
            }
            self.position = 'LONG'
            self.entry_price = current_price
            return signal
            
        elif sell_signal:
            signal = {
                'action': 'SELL',
                'price': current_price,
                'reason': self.get_exit_reason(rsi, ema_fast, ema_slow, current_price),
                'timestamp': pd.Timestamp.now(),
                'pnl': ((current_price - self.entry_price) / self.entry_price) * 100 if self.entry_price else 0
            }
            self.position = None
            self.entry_price = 0
            return signal
            
        return None
    
    def get_exit_reason(self, rsi, ema_fast, ema_slow, current_price):
        """Determine why position is being closed"""
        if rsi > self.config.RSI_OVERBOUGHT:
            return f"RSI overbought: {rsi:.1f}"
        elif ema_fast < ema_slow:
            return f"EMA crossover: Fast={ema_fast:.0f} < Slow={ema_slow:.0f}"
        elif self.check_stop_loss(current_price):
            return f"Stop loss triggered: -{self.config.STOP_LOSS_PERCENT*100}%"
        elif self.check_take_profit(current_price):
            return f"Take profit triggered: +{self.config.TAKE_PROFIT_PERCENT*100}%"
        return "Exit signal"
    
    def check_stop_loss(self, current_price):
        """Check if stop loss is hit"""
        if self.position == 'LONG' and self.entry_price > 0:
            loss_percent = (self.entry_price - current_price) / self.entry_price
            return loss_percent >= self.config.STOP_LOSS_PERCENT
        return False
    
    def check_take_profit(self, current_price):
        """Check if take profit is hit"""
        if self.position == 'LONG' and self.entry_price > 0:
            profit_percent = (current_price - self.entry_price) / self.entry_price
            return profit_percent >= self.config.TAKE_PROFIT_PERCENT
        return False