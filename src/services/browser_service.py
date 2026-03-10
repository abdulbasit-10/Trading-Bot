from typing import List
from playwright.sync_api import Page
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]
from pydantic import SecretStr
from utils.logger import get_logger

logger = get_logger(__name__)


class BrowserService:
    """Browser automation service with AI capabilities"""
    
    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize Gemini LLM"""
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.1,
                api_key=SecretStr(self.api_key)
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def solve_captcha(self, page: Page, target_number: str) -> List[str]:
        if not self.llm:
            logger.warning("LLM not available")
            return []
        logger.warning("AI captcha solving disabled for sequential-only mode")
        return []
    
    def _get_selected_containers(self, page: Page) -> List[str]:
        try:
            return page.evaluate("""
            () => {
                const selected = [];
                const containers = document.querySelectorAll('.col-4');
                
                for (const container of containers) {
                    const img = container.querySelector('.captcha-img');
                    if (img && img.classList.contains('img-selected')) {
                        if (container.id) {
                            selected.push(container.id);
                        }
                    }
                }
                return selected;
            }
            """)
        except Exception:
            return []
