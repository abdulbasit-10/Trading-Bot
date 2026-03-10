# mypy: ignore-errors
"""
Spain Visa Automator with Better CAPTCHA Handling
==========================================================
"""

import random
import time
import re
import io
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeout

from src.config import Config
from src.models.captcha_solver import CaptchaSolver
from src.models.field_detector import FieldDetector
from src.models.applicant import Applicant
from src.services.ocr_service import ocr_service
from utils.logger import get_logger, setup_logging
from utils.helpers import save_screenshot, save_result

logger = get_logger(__name__)


class SpainVisaAutomator:
    """automation class for Spain Visa login."""
    
    def __init__(self, 
                 headless: bool = Config.HEADLESS,
                 max_retries: int = Config.MAX_RETRIES,
                 gemini_api_key: Optional[str] = Config.GEMINI_API_KEY):
        
        self.headless = headless
        self.max_retries = max_retries
        self.timeout = Config.TIMEOUT
        
        # Initialize components
        self.captcha_solver = CaptchaSolver(gemini_api_key)
        self.field_detector = FieldDetector()
        
        # Store current password for use in methods
        self.current_password = None
        
        # Statistics
        self.stats = {
            'total_applicants': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
        }
        self.results = []
        
        setup_logging()
    
    def process_applicants(self, applicants: List[Applicant]) -> Dict:
        """Process multiple applicants."""
        self.stats['total_applicants'] = len(applicants)
        logger.info(f"Starting to process {len(applicants)} applicants")
        
        for idx, applicant in enumerate(applicants, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing applicant {idx}/{len(applicants)}: {applicant.email}")
            logger.info(f"{'='*60}")
            
            # Store current password
            self.current_password = applicant.password
            
            success = self.login(applicant)
            
            result = {
                'applicant': applicant.to_dict(),
                'success': success,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.results.append(result)
            
            self.stats['processed'] += 1
            if success:
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
            
            save_result(self.results, "results.json")
            self._print_progress()
            
            # Delay between applicants
            if idx < len(applicants):
                delay = random.uniform(30, 60)
                logger.info(f"Waiting {delay:.1f} seconds before next applicant...")
                time.sleep(delay)
        
        self._print_statistics()
        self._save_final_results()
        return {'stats': self.stats, 'results': self.results}
    
    def login(self, applicant: Applicant) -> bool:
        """Execute login flow for a single applicant."""
        browser = None
        context = None
        page = None

        try:
            with sync_playwright() as p:
                browser = self._launch_browser(p)

                for attempt in range(self.max_retries):
                    try:
                        logger.info(f"\n{'='*50}")
                        logger.info(f"Attempt {attempt + 1}/{self.max_retries} for {applicant.email}")
                        logger.info(f"{'='*50}")

                        # Create new context for each attempt
                        if context:
                            try:
                                context.close()
                            except Exception:
                                pass
                        
                        context = self._create_context(browser)
                        page = context.new_page()
                        page.set_default_timeout(self.timeout)

                        # Store current password
                        self.current_password = applicant.password

                        # Step 1: Navigate to login
                        if not self._navigate_to_login(page):
                            if not self._navigate_to_alternative(page):
                                raise Exception("Failed to navigate to any login page")

                        # Step 2: Handle email page
                        if not self._handle_email_page(page, applicant.email):
                            logger.error("Failed to handle email page")
                            continue

                        # Step 3: Wait for CAPTCHA page
                        if not self._wait_for_captcha_page(page):
                            logger.error("Failed to load CAPTCHA page")
                            continue

                        # Step 4: Solve CAPTCHA and handle password
                        if self._solve_captcha_and_handle_password(page, applicant.password):
                            logger.info(f"✅ Login successful for {applicant.email}!")
                            save_screenshot(page, f"success_{applicant.email.replace('@', '_')}")
                            
                            if "verification" in page.url.lower():
                                logger.warning("Verification step detected; waiting for completion")
                                start_time = time.time()
                                while time.time() - start_time < 120:
                                    if self._is_logged_in(page):
                                        break
                                    time.sleep(2)
                            
                            return True
                        
                        raise Exception("CAPTCHA solving failed")

                    except PlaywrightTimeout as e:
                        logger.error(f"⏰ Timeout error: {e}")
                        if page:
                            save_screenshot(page, f"timeout_{applicant.email.replace('@', '_')}")
                    except Exception as e:
                        logger.error(f"❌ Attempt {attempt + 1} failed: {str(e)[:200]}")
                        if page:
                            save_screenshot(page, f"error_{applicant.email.replace('@', '_')}")
                    
                    # Wait before retry
                    if attempt < self.max_retries - 1:
                        wait_time = self._get_wait_time(attempt)
                        logger.info(f"⏳ Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)

                logger.error(f"❌ All attempts failed for {applicant.email}")
                return False

        except Exception as e:
            logger.error(f"Fatal error in login: {e}")
            return False
        finally:
            try:
                if context:
                    context.close()
                if browser:
                    browser.close()
            except Exception:
                pass

    def _solve_captcha_and_handle_password(self, page: Page, password: str) -> bool:
        """
        Solve captcha and handle password with retry logic.
        This method will re-enter password after any page reload.
        """
        logger.info("🔐 Starting captcha solving flow...")
        
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                logger.info(f"🔍 CAPTCHA attempt {attempt + 1}/{max_attempts}")
                time.sleep(2)
                
                # Step 1: Get target number
                target = self._get_target_number(page)
                if target is None:
                    logger.error("❌ Cannot read target number")
                    self._refresh_captcha(page)
                    continue
                logger.info(f"🎯 Target number: {target}")
                
                # Step 2: Extract grid numbers with positions
                grid_data = self._extract_grid_with_positions(page)
                if not grid_data or len(grid_data) != 9:
                    logger.warning(f"⚠️ Incomplete grid: {len(grid_data) if grid_data else 0}/9")
                    self._refresh_captcha(page)
                    continue
                
                # Step 3: Log grid for debugging
                self._log_grid(grid_data, target)
                
                # Step 4: Find matching cells
                matches = [i for i, data in enumerate(grid_data) if data['number'] == target]
                if not matches:
                    logger.warning(f"⚠️ Target {target} not found in grid")
                    self._refresh_captcha(page)
                    continue
                logger.info(f"✅ Found matches at indices: {matches}")
                
                # Step 5: Click matching cells
                if not self._click_cells(page, grid_data, matches):
                    logger.error("❌ Failed to click cells")
                    continue
                
                time.sleep(1)
                
                # Step 6: Fill password
                if password:
                    if self._fill_password_reliable(page, password):
                        logger.info("✅ Password filled successfully")
                    else:
                        logger.warning("⚠️ Could not fill password, trying anyway")
                else:
                    logger.warning("⚠️ Password missing, trying anyway")
                
                time.sleep(0.5)
                
                # Step 7: Submit
                if not self._click_submit_button(page):
                    logger.error("❌ Failed to click submit")
                    continue
                
                time.sleep(3)
                
                # Step 8: Verify success
                if self._is_logged_in(page) or self._verify_success(page):
                    logger.info("✅ CAPTCHA solved and logged in successfully!")
                    self.captcha_solver.stats["success"] += 1
                    return True
                else:
                    logger.warning("⚠️ Submit completed but not logged in, may need retry")
                    
            except Exception as exc:
                logger.error(f"❌ Error in attempt {attempt + 1}: {exc}", exc_info=True)
                
        logger.error("❌ All CAPTCHA attempts failed")
        self.captcha_solver.stats["failed"] += 1
        return False

    def _get_target_number(self, page: Page) -> Optional[int]:
        """
        Extract target number from the visible instruction text.
        Looks for "Please select all boxes with number X".
        """
        try:
            # Method 1: Get from box-label elements (visible only)
            labels = page.locator(".box-label").all()
            for label in labels:
                try:
                    if not label.is_visible():
                        continue
                    text = label.text_content() or ""
                    # Look for pattern "Please select all boxes with number NNN"
                    match = re.search(r"Please select all boxes with number\s+(\d+)", text)
                    if match:
                        return int(match.group(1))
                except Exception:
                    continue
            
            # Method 2: JavaScript approach to get visible text only
            result = page.evaluate("""
                () => {
                    const labels = document.querySelectorAll('.box-label');
                    for (const label of labels) {
                        const style = window.getComputedStyle(label);
                        if (style.display === 'none' || style.visibility === 'hidden') continue;
                        const text = label.textContent || '';
                        const match = text.match(/Please select all boxes with number\s+(\d+)/);
                        if (match) return parseInt(match[1]);
                    }
                    return null;
                }
            """)
            if result:
                return result
                
        except Exception as exc:
            logger.debug(f"Target extraction error: {exc}")
            
        return None

    def _extract_grid_with_positions(self, page: Page) -> List[Dict]:
        """
        Extract numbers from all 9 CAPTCHA cells along with their click positions.
        Returns list of dicts: {'number': int, 'x': float, 'y': float, 'element': element}
        """
        grid_data = []
        
        try:
            # Get all captcha image elements
            images = page.locator(".captcha-img").all()
            
            if len(images) != 9:
                logger.warning(f"Expected 9 captcha images, found {len(images)}")
                return []
            
            for idx, img in enumerate(images):
                try:
                    # Get bounding box for click position
                    bbox = img.bounding_box()
                    if not bbox:
                        continue
                    
                    # Calculate center point for clicking
                    center_x = bbox["x"] + bbox["width"] / 2
                    center_y = bbox["y"] + bbox["height"] / 2
                    
                    # Take screenshot of this cell
                    screenshot = img.screenshot(type="png")
                    
                    # Process image and extract number
                    number = self._extract_number_from_image(screenshot)
                    
                    grid_data.append({
                        "number": number,
                        "x": center_x,
                        "y": center_y,
                        "bbox": bbox,
                        "index": idx
                    })
                    
                    logger.debug(f"Cell {idx}: number={number}, pos=({center_x:.1f}, {center_y:.1f})")
                    
                except Exception as exc:
                    logger.debug(f"Error processing cell {idx}: {exc}")
                    grid_data.append({
                        "number": None,
                        "x": 0,
                        "y": 0,
                        "bbox": None,
                        "index": idx
                    })
                    
        except Exception as exc:
            logger.error(f"Grid extraction error: {exc}")
            
        return grid_data

    def _extract_number_from_image(self, image_data: bytes) -> Optional[int]:
        """
        Extract number from CAPTCHA cell image using multiple preprocessing methods.
        """
        try:
            img = Image.open(io.BytesIO(image_data))
            
            # Try multiple preprocessing pipelines
            pipelines = [
                self._preprocess_v1,
                self._preprocess_v2,
                self._preprocess_v3,
                self._preprocess_v4,
                self._preprocess_v5,
            ]
            
            for preprocess in pipelines:
                try:
                    processed = preprocess(img)
                    number = ocr_service.extract_number_from_image(processed)
                    if number is not None and 10 <= number <= 999:
                        return number
                except Exception:
                    continue
                    
        except Exception as exc:
            logger.debug(f"Number extraction error: {exc}")
            
        return None

    def _preprocess_v1(self, img: Image.Image) -> bytes:
        """Grayscale + high contrast + sharpen"""
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(4.0)
        img = ImageEnhance.Sharpness(img).enhance(3.0)
        img = img.filter(ImageFilter.SHARPEN)
        # Resize for better OCR
        w, h = img.size
        img = img.resize((w * 4, h * 4), Image.Resampling.LANCZOS)
        return self._to_bytes(img)

    def _preprocess_v2(self, img: Image.Image) -> bytes:
        """Inverted grayscale"""
        img = img.convert("L")
        img = ImageOps.invert(img)
        img = ImageEnhance.Contrast(img).enhance(3.0)
        w, h = img.size
        img = img.resize((w * 4, h * 4), Image.Resampling.LANCZOS)
        return self._to_bytes(img)

    def _preprocess_v3(self, img: Image.Image) -> bytes:
        """Adaptive threshold using OpenCV"""
        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # Resize first
        h, w = gray.shape
        gray = cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return self._to_bytes(Image.fromarray(thresh))

    def _preprocess_v4(self, img: Image.Image) -> bytes:
        """Otsu threshold"""
        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        gray = cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._to_bytes(Image.fromarray(thresh))

    def _preprocess_v5(self, img: Image.Image) -> bytes:
        """Color-based: extract dark text on light background"""
        arr = np.array(img.convert("RGB"))
        # Convert to LAB and use L channel
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        # Threshold to get dark regions
        _, dark = cv2.threshold(l_channel, 120, 255, cv2.THRESH_BINARY_INV)
        # Resize
        h, w = dark.shape
        dark = cv2.resize(dark, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        return self._to_bytes(Image.fromarray(dark))

    @staticmethod
    def _to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _click_cells(self, page: Page, grid_data: List[Dict], indices: List[int]) -> bool:
        """Click on the specified cell indices."""
        try:
            for idx in indices:
                if idx < len(grid_data):
                    cell = grid_data[idx]
                    x, y = cell["x"], cell["y"]
                    
                    # Add small random offset for human-like behavior
                    offset_x = np.random.randint(-5, 6)
                    offset_y = np.random.randint(-5, 6)
                    
                    click_x = x + offset_x
                    click_y = y + offset_y
                    
                    logger.info(f"  Clicking cell {idx} at ({click_x:.1f}, {click_y:.1f})")
                    page.mouse.click(click_x, click_y)
                    time.sleep(random.uniform(0.3, 0.7))
                    
            return True
        except Exception as exc:
            logger.error(f"Click cells error: {exc}")
            return False

    def _fill_password_reliable(self, page: Page, password: str) -> bool:
        """
        Fill the password field reliably, avoiding honeypot fields.
        The real password field is the visible one near the email display.
        """
        try:
            # Method 1: Find visible password input that's not a honeypot
            all_pwd_fields = page.locator('input[type="password"]').all()
            
            visible_field = None
            for field in all_pwd_fields:
                try:
                    if field.is_visible():
                        # Check if it has a reasonable size (honeypots might be hidden via CSS)
                        bbox = field.bounding_box()
                        if bbox and bbox["width"] > 100 and bbox["height"] > 20:
                            visible_field = field
                            break
                except Exception:
                    continue
            
            if not visible_field:
                logger.warning("No visible password field found")
                return False
            
            # Check if already filled correctly
            try:
                current_value = visible_field.input_value()
                if current_value == password:
                    logger.debug("Password already filled correctly")
                    return True
            except Exception:
                pass
            
            # Clear and fill
            visible_field.scroll_into_view_if_needed()
            time.sleep(0.3)
            visible_field.click()
            time.sleep(0.2)
            visible_field.fill("")
            time.sleep(0.2)
            
            # Type with human-like delay
            for char in password:
                visible_field.type(char, delay=random.uniform(30, 80))
            
            logger.info("✅ Password filled")
            return True
            
        except Exception as exc:
            logger.warning(f"Password fill error: {exc}")
            
            # Fallback: JavaScript fill
            try:
                result = page.evaluate("""
                    (password) => {
                        const inputs = document.querySelectorAll('input[type="password"]');
                        for (const input of inputs) {
                            const style = window.getComputedStyle(input);
                            if (style.display !== 'none' && style.visibility !== 'hidden') {
                                input.value = password;
                                input.dispatchEvent(new Event('input', { bubbles: true }));
                                input.dispatchEvent(new Event('change', { bubbles: true }));
                                return true;
                            }
                        }
                        return false;
                    }
                """, password)
                
                if result:
                    logger.info("✅ Password filled via JavaScript")
                    return True
            except Exception as js_exc:
                logger.debug(f"JS password fill failed: {js_exc}")
                
        return False

    def _ensure_password_filled_reliable(self, page: Page, password: str) -> bool:
        """Legacy method - ensure password field is filled."""
        return self._fill_password_reliable(page, password)

    def _verify_success(self, page: Page) -> bool:
        """Check if login was successful."""
        try:
            time.sleep(2)
            url = page.url.lower()
            
            # Success indicators
            if any(x in url for x in ['/home', '/dashboard', '/global/home']):
                return True
            
            # If we're not on captcha page anymore, likely success
            if 'captcha' not in url and 'logincaptcha' not in url:
                # Double-check by looking for login elements
                has_password = page.locator('input[type="password"]:visible').count() > 0
                has_captcha = page.locator('.captcha-img:visible').count() > 0
                
                if not has_password and not has_captcha:
                    return True
                    
            return False
        except Exception as exc:
            logger.debug(f"Verify error: {exc}")
            return False

    def _refresh_captcha(self, page: Page) -> None:
        """Refresh the CAPTCHA."""
        logger.info("🔄 Refreshing CAPTCHA")
        try:
            # Try clicking refresh link
            refresh_link = page.locator('a[href*="refresh"], a:has-text("Refresh")').first
            if refresh_link.count() > 0:
                refresh_link.click()
                time.sleep(3)
                return
        except Exception:
            pass
        
        # Fallback: reload page
        page.reload()
        time.sleep(3)

    def _log_grid(self, grid_data: List[Dict], target: int) -> None:
        """Log the grid for debugging."""
        logger.info("📊 CAPTCHA Grid (3×3):")
        for row in range(3):
            row_str = ""
            for col in range(3):
                idx = row * 3 + col
                cell = grid_data[idx] if idx < len(grid_data) else None
                if cell and cell["number"] is not None:
                    marker = " ◀" if cell["number"] == target else "  "
                    row_str += f"{cell['number']:>4}{marker} | "
                else:
                    row_str += " ???  | "
            logger.info(f"   {row_str}")

    def _navigate_to_login(self, page: Page) -> bool:
        """Navigate to login page with multiple strategies."""
        logger.info("🌐 Attempting to navigate to Spain Visa login...")
        
        urls_to_try = [
            Config.LOGIN_URL,
            "https://appointment.thespainvisa.com/Global/account/login",
            "https://appointment.thespainvisa.com/Global/Account/Login",
            "https://appointment.thespainvisa.com/",
            "http://appointment.thespainvisa.com/Global/account/login"
        ]
        
        for url in urls_to_try:
            try:
                logger.info(f"Trying URL: {url}")
                page.goto(url, wait_until="commit", timeout=30000)
                time.sleep(3)
                
                page_title = page.title()
                page_url = page.url
                
                logger.info(f"Page title: {page_title}")
                logger.info(f"Current URL: {page_url}")
                
                if "login" in page_url.lower() or "account" in page_url.lower() or "login" in page_title.lower():
                    logger.info(f"✅ Successfully loaded login page from {url}")
                    
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=10000)
                    except Exception:
                        pass
                    
                    return True
                else:
                    logger.warning(f"Page loaded but might not be login page: {page_url}")
                    return True
                    
            except Exception as e:
                logger.warning(f"Failed to load {url}: {e}")
                continue
        
        logger.error("❌ Could not load any login page")
        return False
    
    def _navigate_to_alternative(self, page: Page) -> bool:
        """Try alternative navigation methods."""
        try:
            page.goto("https://appointment.thespainvisa.com/", wait_until="commit", timeout=30000)
            time.sleep(3)
            
            login_links = page.locator('a:has-text("Login")').all()
            if login_links:
                login_links[0].click()
                time.sleep(3)
                return True
            
            return False
        except Exception:
            return False
    
    def _handle_email_page(self, page: Page, email: str) -> bool:
        """Handle first page - email entry."""
        logger.info("📧 Processing email page...")
        
        save_screenshot(page, f"before_email_{email.replace('@', '_')}")
        time.sleep(2)
        
        # Find email field
        email_field = None
        
        # Method 1: Use field detector
        result = self.field_detector.find_visible_email_field(page)
        if result and result[1]:
            email_field = result[1]
        
        # Method 2: Look for email input directly
        if not email_field:
            email_inputs = page.locator('input[type="email"]:visible').all()
            if email_inputs:
                email_field = email_inputs[0]
        
        # Method 3: Look for input with email-related attributes
        if not email_field:
            selectors = [
                'input[name*="email" i]:visible',
                'input[id*="email" i]:visible',
                'input[placeholder*="email" i]:visible'
            ]
            for selector in selectors:
                inputs = page.locator(selector).all()
                if inputs:
                    email_field = inputs[0]
                    break
        
        if not email_field:
            logger.error("❌ Could not find email field")
            return False
        
        logger.info("✅ Found email field")
        
        # Fill email
        try:
            email_field.scroll_into_view_if_needed()
            time.sleep(0.3)
            email_field.click()
            time.sleep(0.2)
            email_field.fill('')
            time.sleep(0.2)
            email_field.type(email, delay=random.uniform(50, 150))
            logger.info(f"✅ Email entered: {email}")
        except Exception as e:
            logger.error(f"Failed to fill email: {e}")
            return False
        
        time.sleep(random.uniform(1, 2))
        
        # Click submit button
        if not self._click_submit_button(page):
            logger.error("❌ Failed to click submit button")
            return False
        
        logger.info("✅ Email submitted, waiting for captcha page...")
        return True
    
    def _wait_for_captcha_page(self, page: Page) -> bool:
        """Wait for CAPTCHA page to load."""
        logger.info("⏳ Waiting for CAPTCHA page...")
        
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                # Check for CAPTCHA elements
                has_captcha = page.locator('.captcha-img').count() > 0
                has_target = page.locator('.box-label').count() > 0
                
                if has_captcha and has_target:
                    logger.info("✅ CAPTCHA page detected")
                    return True
                
                # Check for password field (fallback)
                if len(page.locator('input[type="password"]:visible').all()) > 0:
                    logger.info("✅ Password field detected (no captcha)")
                    return True
                
                # Check URL
                if 'captcha' in page.url.lower():
                    logger.info(f"✅ Navigated to: {page.url}")
                    return True
                    
            except Exception as e:
                logger.debug(f"Error while waiting: {e}")
            
            time.sleep(1)
        
        logger.error("❌ Timeout waiting for CAPTCHA page")
        save_screenshot(page, "captcha_timeout")
        return False
    
    def _wait_for_password_or_captcha_page(self, page: Page) -> bool:
        """Legacy method - wait for either password page or captcha page."""
        return self._wait_for_captcha_page(page)
    
    def _click_submit_button(self, page: Page) -> bool:
        """Click submit button with multiple fallbacks."""
        logger.info("🔍 Looking for submit button...")
        
        # Before clicking, ensure password field (if present) is not empty
        try:
            if self.current_password:
                pwd_locator = page.locator('input[type="password"]:visible').first
                if pwd_locator.count() > 0:
                    try:
                        current_val = pwd_locator.input_value()
                    except Exception:
                        current_val = ""
                    if not current_val:
                        logger.info("🔐 Password field empty before submit, refilling...")
                        self._fill_password_reliable(page, self.current_password)
        except Exception as e:
            logger.debug(f"Password pre-check before submit failed: {e}")

        selectors = [
            '#btnVerify',
            'button[type="submit"]',
            'input[type="submit"]',
            '.btn-success',
            'button:has-text("Submit")',
            'button:has-text("Login")',
            'button:has-text("Continue")',
            'button:has-text("Sign In")',
            'button:has-text("Verificar")'
        ]
        
        for selector in selectors:
            try:
                btn = page.locator(selector).first
                if btn.count() > 0:
                    # Ensure the button is visible before interacting
                    try:
                        btn.wait_for(state="visible", timeout=3000)
                    except Exception:
                        if not btn.is_visible():
                            continue

                    btn.scroll_into_view_if_needed()
                    time.sleep(0.5)
                    
                    try:
                        btn.click()
                        logger.info(f"✅ Clicked submit: {selector}")
                        return True
                    except Exception:
                        # Try JS click
                        page.evaluate(f"document.querySelector('{selector}').click()")
                        logger.info(f"✅ Clicked submit via JS: {selector}")
                        return True
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue
        
        # Generic JS fallback
        try:
            page.evaluate("""
                () => {
                    const btns = document.querySelectorAll('button, input[type="submit"]');
                    for (const b of btns) {
                        const t = (b.innerText || b.value || '').toLowerCase();
                        if (t.includes('submit') || t.includes('continue') || t.includes('login') || t.includes('verify') || b.type === 'submit') {
                            b.click();
                            return true;
                        }
                    }
                    return false;
                }
            """)
            logger.info("✅ Clicked submit via generic JS")
            return True
        except Exception as e:
            logger.error(f"Submit click error: {e}")
        
        logger.error("❌ Submit button not found")
        return False
    
    def _is_logged_in(self, page: Page) -> bool:
        """Check if user is logged in."""
        try:
            current_url = page.url.lower()
            
            # Check for successful login URLs
            success_urls = ['/home', '/dashboard', '/Global/home', '/account/dashboard']
            for url in success_urls:
                if url in current_url:
                    logger.debug(f"Logged in detected via URL: {current_url}")
                    return True
            
            # Check for absence of login elements
            has_login_elements = (
                len(page.locator('input[type="password"]:visible').all()) > 0 or
                page.locator('.captcha-img').count() > 0 or
                len(page.locator('input[type="email"]:visible').all()) > 0
            )
            
            return not has_login_elements and 'login' not in current_url and 'captcha' not in current_url
            
        except Exception as e:
            logger.error(f"Error checking login state: {e}")
            return False
    
    def _is_login_success_page(self, page: Page) -> bool:
        """Check if current page indicates login success."""
        try:
            current_url = page.url.lower()
            
            # Success URL indicators
            success_patterns = [
                '/home',
                '/dashboard',
                '/Global/home',
                '/account/dashboard',
                '/user/profile'
            ]
            
            for pattern in success_patterns:
                if pattern in current_url:
                    logger.info(f"✅ Success page detected via URL: {current_url}")
                    return True
            
            # Check page content for success indicators
            page_content = page.content().lower()
            success_texts = [
                'welcome',
                'dashboard',
                'my account',
                'logout',
                'sign out',
                'profile'
            ]
            
            # Only check content if we're not on a login/captcha page
            if 'login' not in current_url and 'captcha' not in current_url:
                for text in success_texts:
                    if text in page_content:
                        logger.info(f"✅ Success detected via content: '{text}'")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking login success: {e}")
            return False
    
    def _is_rate_limited(self, page: Page) -> bool:
        """Check for rate limiting."""
        try:
            content = page.content().lower()
            indicators = [
                "too many requests", 
                "rate limit", 
                "429", 
                "blocked",
                "access denied",
                "unusual traffic",
                "try again later"
            ]
            return any(indicator in content for indicator in indicators)
        except Exception:
            return False
    
    def _get_wait_time(self, attempt: int) -> float:
        """Calculate wait time between retries."""
        return (attempt + 1) * 20 + random.uniform(10, 20)
    
    def _launch_browser(self, playwright):
        """Launch browser with anti-detection."""
        return playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--window-size=1920,1080',
                '--start-maximized',
                '--disable-gpu',
                '--disable-software-rasterizer'
            ]
        )
    
    def _create_context(self, browser: Browser) -> BrowserContext:
        """Create browser context with anti-detection."""
        context = browser.new_context(
            user_agent=random.choice(Config.USER_AGENTS),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="Europe/Madrid",
            geolocation={"latitude": 40.4168, "longitude": -3.7038},
            permissions=["geolocation"],
            color_scheme="light",
            ignore_https_errors=True,
            accept_downloads=True
        )
        
        # Anti-detection script
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        """)
        
        return context
    
    def _print_progress(self):
        """Print current progress."""
        logger.info("-" * 50)
        logger.info(f"Progress: {self.stats['processed']}/{self.stats['total_applicants']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Success Rate: {(self.stats['successful']/max(self.stats['processed'],1)*100):.1f}%")
        logger.info("-" * 50)
    
    def _print_statistics(self):
        """Print final statistics."""
        logger.info(f"\n{'='*60}")
        logger.info("📊 FINAL STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Applicants: {self.stats['total_applicants']}")
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        if self.stats['processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['processed']) * 100
            logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Captcha stats
        captcha_stats = self.captcha_solver.get_stats()
        logger.info("\n🤖 Captcha Stats:")
        logger.info(f"  OCR Success: {captcha_stats.get('success', 0)}")
        logger.info(f"  Manual: {captcha_stats.get('manual', 0)}")
        logger.info(f"  Failed: {captcha_stats.get('failed', 0)}")
        logger.info(f"  Total: {captcha_stats.get('attempts', 0)}")
        logger.info(f"{'='*60}")
    
    def _save_final_results(self):
        """Save final results to files."""
        result_data = {
            'stats': self.stats,
            'results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        save_result(result_data, "final_results.json")
        self._save_summary_csv()
    
    def _save_summary_csv(self):
        """Save summary results as CSV."""
        import csv
        from pathlib import Path
        
        filename = f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = Path("results") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Email', 'Name', 'Passport', 'Status', 'Timestamp'])
            
            for result in self.results:
                applicant = result['applicant']
                writer.writerow([
                    applicant['email'],
                    f"{applicant.get('first_name', '')} {applicant.get('last_name', '')}",
                    applicant.get('passport_number', ''),
                    'SUCCESS' if result['success'] else 'FAILED',
                    result['timestamp']
                ])
        
        logger.info(f"📁 Summary saved to {filepath}")
    
    def _fill_field(self, page: Page, field_id: str, field, value: str, field_type: str) -> bool:
        """Legacy method - fill a form field with human-like typing."""
        try:
            logger.info(f"✍️ Filling {field_type} field...")
            
            # Enable field via JavaScript
            page.evaluate(f"""
                document.getElementById('{field_id}').removeAttribute('disabled');
                document.getElementById('{field_id}').removeAttribute('readonly');
            """)
            
            time.sleep(0.5)
            
            # Scroll into view
            field.scroll_into_view_if_needed()
            time.sleep(0.3)
            
            # Click and clear
            field.click()
            time.sleep(0.3)
            
            # Clear using JavaScript to be sure
            page.evaluate(f"document.getElementById('{field_id}').value = ''")
            time.sleep(0.2)
            
            # Type with human-like delay
            for char in value:
                field.type(char, delay=random.uniform(50, 150))
            
            # Verify
            time.sleep(0.3)
            field_value = field.input_value()
            if field_value == value:
                logger.info(f"✅ Successfully filled {field_type} field")
                return True
            else:
                logger.warning(f"⚠️ Value mismatch: got '{field_value}'")
                # Try one more time with JavaScript
                page.evaluate(f"document.getElementById('{field_id}').value = '{value}'")
                return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fill {field_type} field: {e}")
            return False
    
    def _handle_captcha(self, page: Page) -> bool:
        """Legacy method - handle captcha challenge."""
        return self._solve_captcha_and_handle_password(page, self.current_password if self.current_password else "")
    
    def _submit_final_form(self, page: Page) -> bool:
        """Legacy method - submit final form."""
        return self._click_submit_button(page)
    
    def _check_response(self, page: Page) -> Tuple[bool, str]:
        """Legacy method - check response after submission."""
        try:
            time.sleep(2)
            current_url = page.url
            logger.info(f"Current URL: {current_url}")
            
            if "/Global/home" in current_url or current_url.endswith("/"):
                return True, "Login successful - redirected to home"
            
            if "/Global/Account/ChangePassword" in current_url:
                return True, "Login successful - password change required"
            
            if "/Global/Account/UserVerification" in current_url:
                return True, "Login successful - verification required"
            
            if "/Global/account/bot" in current_url:
                return False, "Bot detected by server"
            
            page_content = page.content().lower()
            if "welcome" in page_content or "dashboard" in page_content:
                return True, "Login successful"
            
            error_div = page.locator('.validation-summary').first
            if error_div.is_visible():
                error_text = error_div.text_content()
                if error_text:
                    return False, f"Validation error: {error_text.strip()}"
            
            if "login" in current_url.lower():
                return False, "Still on login page"
            
            return False, "Unknown state"
            
        except Exception as e:
            return False, f"Error checking response: {e}"
    
    def _check_for_captcha(self, page: Page) -> bool:
        """Legacy method - check if captcha is present."""
        try:
            return page.locator('.captcha-img').count() > 0
        except Exception:
            return False
    
    def _wait_for_post_captcha(self, page: Page) -> bool:
        """Legacy method - wait for post-captcha page."""
        logger.info("⏳ Waiting for post-captcha page to load...")
        
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                current_url = page.url
                
                if "/Global/home" in current_url or "dashboard" in current_url:
                    logger.info("✅ Successfully navigated to dashboard")
                    return True
                
                if "login" not in current_url.lower() and "captcha" not in current_url.lower():
                    logger.info(f"✅ Navigated to: {current_url}")
                    return True
                
                password_field = page.locator('input[type="password"]').first
                if password_field.count() > 0 and password_field.is_visible():
                    logger.info("✅ Password field detected")
                    return True
                
                time.sleep(1)
                
            except Exception as e:
                logger.debug(f"Error while waiting: {e}")
                time.sleep(1)
        
        logger.warning("⚠️ Timeout waiting for post-captcha navigation")
        return False
    
    def _ensure_password_field_filled(self, page: Page, password: str) -> bool:
        """Legacy method - ensure password field is filled."""
        return self._fill_password_reliable(page, password)
    
    def _fill_field_with_js(self, page: Page, selector: str, value: str):
        """Legacy method - fill field with JavaScript."""
        try:
            page.evaluate("""
                (args) => {
                    const el = document.querySelector(args.selector);
                    if (el) {
                        el.removeAttribute('readonly');
                        el.removeAttribute('disabled');
                        el.value = args.value;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                        el.dispatchEvent(new Event('blur', { bubbles: true }));
                        return true;
                    }
                    return false;
                }
            """, {"selector": selector, "value": value})
        except Exception as e:
            logger.error(f"JS fill failed: {e}")
            try:
                page.fill(selector, value)
            except Exception:
                pass
    
    def _verify_password_filled(self, page: Page, expected_password: str) -> bool:
        """Legacy method - verify password is filled."""
        try:
            password_fields = page.locator('input[type="password"]:visible').all()
            if not password_fields:
                return False
            field_value = password_fields[0].input_value()
            return field_value == expected_password
        except Exception:
            return False
    
    # Keep these methods for backward compatibility
    def login_flow(self, page: Page, applicant: Applicant):
        """Legacy method - kept for compatibility."""
        return self.login(applicant)
    
    def _process_single_applicant(self, page: Page, applicant: Applicant) -> bool:
        """Legacy method - kept for compatibility."""
        return self.login(applicant)
    
    def handle_password_captcha_page(self, page: Page, applicant_data: Dict) -> bool:
        """Legacy method - kept for compatibility."""
        if 'password' in applicant_data:
            return self._solve_captcha_and_handle_password(page, applicant_data['password'])
        return False
