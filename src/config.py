import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import csv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration management for the bot"""
    
    # API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    # Bot settings
    HEADLESS: bool = os.getenv("HEADLESS", "false").lower() == "true"
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    BOOKING_RETRIES: int = int(os.getenv("BOOKING_RETRIES", "3"))
    TIMEOUT: int = int(os.getenv("TIMEOUT", "30000"))
    
    # URLs
    LOGIN_URL: str = "https://appointment.thespainvisa.com/Global/account/login"
    LOCAL_PASSWORD_HTML: str = os.getenv("LOCAL_PASSWORD_HTML", "")
    
    # OCR settings
    OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', '0.5'))
    
    
    
    # Captcha interaction timing (milliseconds)
    CAPTCHA_PAGE_WAIT_MS: int = int(os.getenv("CAPTCHA_PAGE_WAIT_MS", "5000"))
    CAPTCHA_CLICK_DELAY_MS: int = int(os.getenv("CAPTCHA_CLICK_DELAY_MS", "40"))
    CAPTCHA_AFTER_SELECTION_WAIT_MS: int = int(os.getenv("CAPTCHA_AFTER_SELECTION_WAIT_MS", "60"))
    CAPTCHA_SELECTION_RETRIES: int = int(os.getenv("CAPTCHA_SELECTION_RETRIES", "2"))
    
    
    # File paths
    APPLICANTS_FILE: str = os.getenv("APPLICANTS_FILE", "applicants.csv")
    SCREENSHOTS_DIR: str = "screenshots"
    LOGS_DIR: str = "logs"
    
    # User agents for rotation
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    ]
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        # Create directories if they don't exist
        Path(cls.SCREENSHOTS_DIR).mkdir(exist_ok=True)
        Path(cls.LOGS_DIR).mkdir(exist_ok=True)
        
        # Check if applicants file exists
        if not Path(cls.APPLICANTS_FILE).exists():
            raise FileNotFoundError(f"Applicants file not found: {cls.APPLICANTS_FILE}")
    
    @classmethod
    def load_applicants(cls) -> List[Dict]:
        """Load applicants from CSV file"""
        applicants = []
        
        try:
            with open(cls.APPLICANTS_FILE, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Clean and validate the data
                    if row.get('email') and row.get('password'):
                        applicants.append(row)
                    else:
                        print("Skipping invalid row: missing email or password")
            
            print(f"Loaded {len(applicants)} applicants from {cls.APPLICANTS_FILE}")
            return applicants
            
        except Exception as e:
            print(f"Error loading applicants: {e}")
            return []
