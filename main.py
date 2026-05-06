import time
import signal
import sys
from datetime import datetime
from binance_client import BinanceTradingClient
from strategy import TradingStrategy
from risk_manager import RiskManager
from trade_logger import TradeLogger
from config import Config

class CryptoTradingBot:
    def __init__(self):
        """Initialize the trading bot"""
        print("🤖 Initializing Crypto Trading Bot...")
        print("="*50)
        
        self.config = Config()
        self.client = BinanceTradingClient()
        self.strategy = TradingStrategy()
        self.risk_manager = RiskManager(self.client)
        self.logger = TradeLogger()
        
        self.running = True
        signal.signal(signal.SIGINT, self.shutdown)
        
        # Display initial status
        self.display_status()
    
    def display_status(self):
        """Display bot configuration and status"""
        print("\n📋 Bot Configuration:")
        print(f"  • Exchange: {'Binance Testnet' if self.config.TESTNET else 'Binance Mainnet'}")
        print(f"  • Symbol: {self.config.SYMBOL}")
        fmt = self.config.TIMEFRAME
        print(f"  • Strategy: RSI({self.config.RSI_PERIOD}) + EMA({self.config.EMA_FAST},{self.config.EMA_SLOW})")
        print(f"  • Risk Management: SL={self.config.STOP_LOSS_PERCENT*100}%, TP={self.config.TAKE_PROFIT_PERCENT*100}%")
        print(f"  • Check Interval: {self.config.CHECK_INTERVAL}s")
        print("="*50 + "\n")
    
    def execute_trade(self, signal):
        """Execute trade based on signal"""
        if not self.risk_manager.can_trade():
            self.logger.log_system_event("WARNING", "Risk manager blocked trade execution")
            return False
        
        current_price = self.client.get_current_price()
        if not current_price:
            return False
        
        if signal['action'] == 'BUY':
            # Calculate position size
            quantity = self.risk_manager.calculate_position_size(current_price)
            if not quantity:
                return False
            
            # Validate order
            valid, message = self.risk_manager.validate_order('BUY', quantity, current_price)
            if not valid:
                self.logger.log_system_event("ERROR", f"Order validation failed: {message}")
                return False
            
            # Execute buy order
            order = self.client.place_market_order('BUY', quantity)
            if order:
                self.logger.log_trade(signal)
                self.logger.log_system_event("INFO", f"BUY executed: {quantity} BTC @ ${current_price:.2f}")
                return True
        
        elif signal['action'] == 'SELL':
            # For demo, sell all BTC balance
            balance = self.client.get_account_balance()
            if not balance:
                return False
            
            # Get BTC balance (simplified - in production, get actual BTC balance)
            btc_balance = 0.001  # This should be fetched properly
            
            order = self.client.place_market_order('SELL', btc_balance)
            if order:
                # Calculate P&L for this trade
                if hasattr(self.strategy, 'entry_price') and self.strategy.entry_price > 0:
                    signal['pnl'] = ((current_price - self.strategy.entry_price) / self.strategy.entry_price) * 100
                
                self.logger.log_trade(signal)
                self.risk_manager.update_trade_stats(signal)
                self.logger.log_system_event("INFO", f"SELL executed: {btc_balance} BTC @ ${current_price:.2f}")
                return True
        
        return False
    
    def run_once(self):
        """Run one iteration of the trading bot"""
        try:
            # Fetch market data
            market_data = self.client.get_market_data(limit=50)
            if not market_data:
                self.logger.log_system_event("WARNING", "Failed to fetch market data")
                return
            
            # Generate trading signal
            signal = self.strategy.generate_signal(market_data)
            
            if signal:
                print(f"\n🎯 Signal Generated at {datetime.now().strftime('%H:%M:%S')}:")
                print(f"   Action: {signal['action']}")
                print(f"   Price: ${signal['price']:.2f}")
                print(f"   Reason: {signal['reason']}")
                
                # Execute trade
                success = self.execute_trade(signal)
                if success:
                    print(f"✅ Trade executed successfully!")
                else:
                    print(f"❌ Trade execution failed")
            
            # Log current status
            current_price = self.client.get_current_price()
            if current_price:
                balance = self.client.get_account_balance()
                if balance:
                    print(f"💹 Status - Price: ${current_price:.2f} | Balance: ${balance['free']:.2f} USDT")
                    
        except Exception as e:
            self.logger.log_system_event("ERROR", f"Error in main loop: {str(e)}")
    
    def run(self):
        """Main bot loop"""
        print("🚀 Starting trading bot...\n")
        
        iteration = 0
        while self.running:
            iteration += 1
            print(f"\n{'='*40}")
            print(f"Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*40}")
            
            self.run_once()
            
            # Wait for next interval
            time.sleep(self.config.CHECK_INTERVAL)
    
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        print("\n\n🛑 Shutting down bot...")
        self.running = False
        
        # Generate final report
        print("\n" + "="*50)
        print("FINAL PERFORMANCE REPORT")
        print("="*50)
        
        # Display risk manager summary
        print(self.risk_manager.get_performance_summary())
        
        # Display trade logger report
        self.logger.get_performance_report()
        
        print("\n✅ Bot shutdown complete")
        sys.exit(0)

if __name__ == "__main__":
    bot = CryptoTradingBot()
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.shutdown(None, None)