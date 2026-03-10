import logging
import sys
from pathlib import Path

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)


class CustomLogger:
    """Custom logger with colored output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',        # Green
        'WARNING': '\033[33m',     # Yellow
        'ERROR': '\033[31m',       # Red
        'CRITICAL': '\033[35m',    # Magenta
        'SUCCESS': '\033[32m\033[1m',  # Bold Green
        'RESET': '\033[0m'
    }
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def debug(self, message: str):
        self.logger.debug(message)
        print(f"{self.COLORS['DEBUG']}[DEBUG] {message}{self.COLORS['RESET']}")
    
    def info(self, message: str):
        self.logger.info(message)
        print(f"{self.COLORS['INFO']}[INFO] {message}{self.COLORS['RESET']}")
    
    def warning(self, message: str):
        self.logger.warning(message)
        print(f"{self.COLORS['WARNING']}[WARNING] {message}{self.COLORS['RESET']}")
    
    def error(self, message: str):
        self.logger.error(message)
        print(f"{self.COLORS['ERROR']}[ERROR] {message}{self.COLORS['RESET']}")
    
    def critical(self, message: str):
        self.logger.critical(message)
        print(f"{self.COLORS['CRITICAL']}[CRITICAL] {message}{self.COLORS['RESET']}")
    
    def success(self, message: str):
        self.logger.info(f"SUCCESS: {message}")
        print(f"{self.COLORS['SUCCESS']}[SUCCESS] {message}{self.COLORS['RESET']}")


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(stdout_reconfigure):
        try:
            stdout_reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
    if callable(stderr_reconfigure):
        try:
            stderr_reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "bot.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_logger(name: str) -> CustomLogger:
    """Get a custom logger instance"""
    return CustomLogger(name)
