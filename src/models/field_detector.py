from typing import Optional, Tuple
from playwright.sync_api import Page, Locator
from utils.logger import get_logger

logger = get_logger(__name__)


class FieldDetector:
    """Detects and handles form fields using robust strategies."""

    def find_visible_email_field(self, page: Page) -> Optional[Tuple[str, Locator]]:
        """Find the visible email field using a robust strategy."""
        logger.info("Looking for visible email field...")

        selectors = [
            'input[type="email"]:visible',
            'input.entry-disabled[type="text"]:visible',
            'input[name*="email" i]:visible',
            'input[id*="email" i]:visible',
            'input[placeholder*="email" i]:visible',
        ]

        for selector in selectors:
            try:
                field = page.locator(selector).first
                if field.count() > 0 and field.is_visible():
                    field_id = field.get_attribute('id') or "email-field-locator"
                    logger.info(f"Found visible email field with selector: {selector}")
                    return field_id, field
            except Exception as e:
                logger.debug(f"Error finding email field with selector {selector}: {e}")
                continue
        
        logger.error("Could not find a visible email field.")
        return None

    def find_visible_password_field(self, page: Page) -> Optional[Tuple[str, Locator]]:
        """Find the visible password field using a robust strategy."""
        logger.info("Looking for visible password field...")

        selectors = [
            'input[type="password"]:visible',
            'input.entry-disabled[type="password"]:visible',
            'input[name*="password" i]:visible',
            'input[id*="password" i]:visible',
            'input[placeholder*="password" i]:visible',
        ]
        
        for selector in selectors:
            try:
                field = page.locator(selector).first
                if field.count() > 0 and field.is_visible():
                    field_id = field.get_attribute('id') or "password-field-locator"
                    logger.info(f"Found visible password field with selector: {selector}")
                    return field_id, field
            except Exception as e:
                logger.debug(f"Error finding password field with selector {selector}: {e}")
                continue
        
        logger.error("Could not find a visible password field.")
        return None
