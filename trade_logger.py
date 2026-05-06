import json
import logging
from datetime import datetime
from pathlib import Path

class TradeLogger:
    def __init__(self, log_file='logs/trades.json'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_trade(self, trade_data):
        """
        Log trade entry and exit with timestamp [citation:5]
        """
        trade_data['log_time'] = datetime.now().isoformat()
        
        # Load existing trades
        trades = self.load_trades()
        trades.append(trade_data)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(trades, f, indent=2)
        
        # Log to console
        action = trade_data.get('action', 'UNKNOWN')
        price = trade_data.get('price', 0)
        reason = trade_data.get('reason', '')
        pnl = trade_data.get('pnl', None)
        
        if pnl is not None:
            self.logger.info(f"📊 TRADE CLOSED: {action} at ${price:.2f} | P&L: {pnl:.2f}% | {reason}")
        else:
            self.logger.info(f"🚀 TRADE OPENED: {action} at ${price:.2f} | {reason}")
        
        return trade_data
    
    def load_trades(self):
        """Load trade history from file"""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []
    
    def get_performance_report(self):
        """
        Generate performance report for demo [citation:10]
        """
        trades = self.load_trades()
        
        if not trades:
            return "No trades recorded yet"
        
        # Separate buy and sell trades
        buys = [t for t in trades if t.get('action') == 'BUY']
        sells = [t for t in trades if t.get('action') == 'SELL']
        
        report = {
            'total_trades': len(sells),
            'trades': trades,
            'summary': {
                'total_trades_executed': len(sells),
                'current_position': 'OPEN' if len(buys) > len(sells) else 'CLOSED',
                'last_update': datetime.now().isoformat()
            }
        }
        
        # Display report
        print("\n" + "="*50)
        print("TRADING PERFORMANCE REPORT")
        print("="*50)
        print(f"Total Trades: {report['summary']['total_trades_executed']}")
        print(f"Position Status: {report['summary']['current_position']}")
        print(f"Trade History: {len(trades)} events recorded")
        print("="*50)
        
        # Show recent trades
        print("\n📋 Recent Trades:")
        for trade in trades[-5:]:  # Last 5 trades
            action = trade.get('action')
            price = trade.get('price')
            reason = trade.get('reason', '')[:50]
            pnl = trade.get('pnl')
            
            if pnl:
                print(f"  • {action} @ ${price:.2f} | P&L: {pnl:+.2f}% | {reason}")
            else:
                print(f"  • {action} @ ${price:.2f} | {reason}")
        
        return report
    
    def log_system_event(self, event_type, message):
        """Log system events and errors"""
        if event_type == 'ERROR':
            self.logger.error(message)
        elif event_type == 'WARNING':
            self.logger.warning(message)
        else:
            self.logger.info(message)