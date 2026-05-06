from datetime import datetime, timedelta
from config import Config

class RiskManager:
    def __init__(self, client):
        self.client = client
        self.config = Config()
        self.daily_trades = 0
        self.last_trade_date = None
        self.trade_history = []
    
    def can_trade(self):
        """
        Validate if we can place new trades based on risk rules [citation:4]
        """
        # Check daily trade limit
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
        
        if self.daily_trades >= self.config.MAX_DAILY_TRADES:
            print(f"⚠️ Daily trade limit reached: {self.daily_trades}/{self.config.MAX_DAILY_TRADES}")
            return False
        
        # Check account balance
        balance = self.client.get_account_balance()
        if not balance or balance['free'] < 10:
            print("❌ Insufficient balance for trading")
            return False
        
        return True
    
    def calculate_position_size(self, current_price):
        """
        Calculate safe position size based on capital allocation [citation:4]
        """
        balance = self.client.get_account_balance()
        if not balance:
            return self.config.QUANTITY
        
        # Use allocated percentage of available balance
        available_capital = balance['free'] * self.config.CAPITAL_ALLOCATION
        position_size = available_capital / current_price
        
        # Apply max position limit
        position_size = min(position_size, self.config.MAX_POSITION_SIZE)
        
        # Ensure minimum notional value (Binance requires $10 minimum)
        min_notional = 10 / current_price
        if position_size < min_notional:
            print(f"⚠️ Position too small: {position_size} BTC (min: {min_notional})")
            return None
        
        print(f"📊 Position size calculated: {position_size:.6f} BTC (~${position_size*current_price:.2f})")
        return round(position_size, 6)
    
    def validate_order(self, side, quantity, price):
        """
        Validate order before execution
        """
        if quantity <= 0:
            return False, "Invalid quantity"
        
        # Validate price is within reasonable range
        current_price = self.client.get_current_price()
        if abs(price - current_price) / current_price > 0.1:  # 10% slippage protection
            return False, "Price too far from market"
        
        return True, "Order valid"
    
    def update_trade_stats(self, trade_result):
        """
        Update trading statistics after trade execution
        """
        self.daily_trades += 1
        self.trade_history.append(trade_result)
        print(f"📈 Trade #{self.daily_trades} completed: P&L = {trade_result.get('pnl', 0):.2f}%")
    
    def get_performance_summary(self):
        """
        Generate performance summary
        """
        if not self.trade_history:
            return "No trades executed yet"
        
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        summary = f"""
        📊 Performance Summary:
        - Total Trades: {total_trades}
        - Winning Trades: {len(winning_trades)}
        - Losing Trades: {len(losing_trades)}
        - Win Rate: {win_rate:.1f}%
        - Total P&L: {total_pnl:.2f}%
        - Average P&L per Trade: {avg_pnl:.2f}%
        """
        return summary