# mypy: ignore-errors
import time
import re
from typing import List, Optional, Tuple, Dict
from playwright.sync_api import Page, ElementHandle, TimeoutError as PlaywrightTimeout

# Assuming these are your internal modules
from utils.logger import get_logger
from src.config import Config

logger = get_logger(__name__)

class CaptchaSolver:
    """Handles captcha solving using OCR with manual fallback"""

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key
        self.stats = {
            'ocr_success': 0,
            'manual': 0,
            'failed': 0,
            'total': 0
        }
        self._captcha_cache: Dict[str, Tuple[str, float]] = {}
        self._warm_cache_from_screenshots()
        logger.info("CaptchaSolver initialized")

    def solve_captcha(self, page: Page, password: Optional[str] = None) -> bool:
        """
        Solve captcha by reading numbers from images and selecting matching cells
        """
        self.stats['total'] += 1

        max_attempts = max(6, Config.MAX_RETRIES * 3)
        attempt = 0
        while attempt < max_attempts:
            try:
                logger.info(f"🔍 Starting captcha solving process (Attempt {attempt + 1}/{max_attempts})...")
                if page.is_closed():
                    return False
                
                # Clear any existing stale selection
                self._clear_selection(page)

                # Get target number from instruction text
                target_number = self._get_visible_target_number_js(page)
                if not target_number:
                    logger.error("❌ Could not determine target number")
                    attempt += 1
                    continue

                logger.info(f"🎯 Target number: {target_number}")

                # === 🛠️ THE FIX IS APPLIED HERE ===
                # Find ONLY the physically visible captcha cells sorted by real X/Y coordinates
                cells = self._get_captcha_grid_cells(page)
                if not cells or len(cells) != 9:
                    logger.error(f"❌ Expected 9 visible captcha cells, found {len(cells) if cells else 0}")
                    self._refresh_captcha(page)
                    attempt += 1
                    continue

                logger.info(f"📦 Found {len(cells)} physically visible captcha cells in the 3x3 grid")

                # Extract numbers from each cell using OCR
                cell_numbers = self._extract_numbers_from_cells(page, cells)
                
                # Find cells that match the target number
                matching_indices = []
                low_confidence_count = 0
                for idx, (cell_id, number, confidence) in enumerate(cell_numbers):
                    if confidence < Config.OCR_CONFIDENCE_THRESHOLD:
                        low_confidence_count += 1
                    if number == target_number:
                        matching_indices.append(idx)
                        logger.debug(f"✅ Cell {idx} matches target: {number}")
                    else:
                        logger.debug(f"❌ Cell {idx} does not match target: {number}")

                # If missing target, do aggressive pass
                if not matching_indices or low_confidence_count >= 4:
                    logger.info("Retrying OCR with aggressive pass on all cells.")
                    cell_numbers = self._extract_numbers_from_cells(page, cells, force_aggressive=True)
                    matching_indices = []
                    for idx, (_, number, confidence) in enumerate(cell_numbers):
                        if number == target_number:
                            matching_indices.append(idx)

                if not matching_indices:
                    logger.warning(f"⚠️ Target number {target_number} not found in grid.")
                    self._refresh_captcha(page)
                    attempt += 1
                    continue

                logger.info(f"✅ Found {len(matching_indices)} cells with number {target_number} at indices: {matching_indices}")

                # Clear previous selections again before clicking
                self._clear_selection(page)

                # Select matching cells
                if not self._select_cells(page, cells, matching_indices):
                    logger.error("❌ Failed to select cells")
                    attempt += 1
                    continue

                # Fill password before submitting when available.
                if password:
                    self._fill_password(page, password)

                # Submit captcha
                if not self._submit_captcha(page):
                    logger.error("❌ Failed to submit captcha")
                    attempt += 1
                    continue

                # Wait for submission to process
                time.sleep(0.5)

                # Check if captcha was accepted
                if self._check_captcha_success(page):
                    logger.success("✅ Captcha solved successfully!")
                    self.stats['ocr_success'] += 1
                    return True
                else:
                    logger.warning("⚠️ Captcha submission failed. Website rejected the solution.")
                    self._refresh_captcha(page)
                    attempt += 1
                    continue

            except KeyboardInterrupt:
                raise
            except PlaywrightTimeout as e:
                logger.error(f"⏰ Timeout in captcha solving: {e}")
                attempt += 1
                continue
            except Exception as e:
                logger.error(f"❌ Error in captcha solving: {e}")
                attempt += 1
                continue

        self.stats['failed'] += 1
        return False


    def _get_captcha_grid_cells(self, page: Page) -> List[ElementHandle]:
        """
        Retrieves ONLY the visible captcha cells and sorts them by their physical
        screen coordinates to recreate the true 0-8 indexing.
        """
        try:
            # 1. Get all elements with the captcha image class
            all_cells = page.locator('.captcha-img').element_handles()
            visible_cells = []

            # 2. Filter out decoy/hidden elements
            for cell in all_cells:
                if cell.is_visible():
                    box = cell.bounding_box()
                    if box and box['width'] > 0 and box['height'] > 0:
                        visible_cells.append({
                            'element': cell,
                            'y': round(box['y']), 
                            'x': round(box['x'])
                        })

            # 3. Sort by Y (row) first, then X (column) to recreate the visual 3x3 grid
            # Adding a small tolerance (e.g., // 10) in case cells aren't perfectly aligned
            visible_cells.sort(key=lambda c: (c['y'] // 10, c['x']))

            # Return just the element handles in the correct order
            return [c['element'] for c in visible_cells]

        except Exception as e:
            logger.error(f"Error finding captcha cells: {e}")
            return []


    def _select_cells(self, page: Page, cells: List[ElementHandle], indices_to_click: List[int]) -> bool:
        """
        Clicks the specified indices safely, waiting between clicks to avoid race conditions.
        """
        try:
            for idx in indices_to_click:
                if idx >= len(cells):
                    continue
                    
                cell = cells[idx]
                
                # Check if it's ALREADY selected to prevent accidental de-selection (toggle)
                is_selected = page.evaluate("(element) => element.classList.contains('img-selected')", cell)
                
                if not is_selected:
                    cell.scroll_into_view_if_needed()
                    # Use force=True to bypass potential pointer-event traps
                    cell.click(force=True)
                    
                    # ⏳ CRITICAL WAIT: Give the site's JS time to register the click and update classes
                    time.sleep(0.4) 
                
            return True
        except Exception as e:
            logger.error(f"Error selecting cells: {e}")
            return False


    def _get_visible_target_number_js(self, page: Page) -> Optional[str]:
        """Extract strict 3-digit target number from instruction text using JavaScript."""
        try:
            visible_text = page.evaluate("""
                () => {
                    const boxLabels = document.querySelectorAll('.box-label');
                    for (const label of boxLabels) {
                        const style = window.getComputedStyle(label);
                        const rect = label.getBoundingClientRect();
                        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
                        if (rect.left < 0 || rect.top < 0 || rect.width < 10) continue;
                        return label.textContent;
                    }
                    return null;
                }
            """)
            if visible_text:
                match = re.search(r'\b(\d{3})\b', visible_text)
                if match:
                    return match.group(1)
            return None
        except Exception as e:
            logger.error(f"Error getting visible target number: {e}")
            return None


    def _extract_numbers_from_cells(self, page: Page, cells: List[ElementHandle], force_aggressive: bool = False):
        """Mock implementation - integrate with your actual OCR service"""
        results = []
        for i, cell in enumerate(cells):
            try:
                # Assuming ocr_service.extract takes an element/image and returns (text, confidence)
                # Replace this with your actual OCR call
                # text, conf = ocr_service.extract(cell.screenshot())
                text, conf = ("123", 0.99) # Placeholder
                results.append((f"cell_{i}", text, conf))
            except Exception:
                results.append((f"cell_{i}", None, 0.0))
        return results

    
    def _clear_selection(self, page: Page):
        """Clears all currently selected images by clicking them again."""
        try:
            selected_cells = page.locator('.captcha-img.img-selected').element_handles()
            for cell in selected_cells:
                if cell.is_visible():
                    cell.click(force=True)
                    time.sleep(0.2)
        except Exception as e:
            logger.debug(f"Clear selection passed: {e}")


    def _submit_captcha(self, page: Page) -> bool:
        """Clicks the verify/submit button specifically for the captcha frame"""
        try:
            # Add specific logic for the captcha submit button
            submit_btn = page.locator('#btnVerify, .captcha-verify-btn').first
            if submit_btn.is_visible():
                submit_btn.click()
                return True
            return False
        except Exception:
            return False

    def _check_captcha_success(self, page: Page) -> bool:
        """Check if captcha was accepted"""
        try:
            if page.is_closed():
                return False
            time.sleep(0.02)
            current_url = page.url
            
            if 'captcha' in current_url.lower():
                return False

            captcha_div = page.locator('.captcha-div').first
            if captcha_div.count() > 0 and captcha_div.is_visible():
                return False

            if "/Global/home" in current_url or "/" == current_url or "dashboard" in current_url:
                return True

            password_field = page.locator('input[type="password"]').first
            if password_field.count() > 0 and password_field.is_visible():
                return True

            success_msg = page.locator('text="success"').first
            if success_msg.count() > 0:
                return True

            return True
        except Exception as e:
            logger.error(f"Error checking captcha success: {e}")
            return False

    def _refresh_captcha(self, page: Page):
        """Clicks the refresh icon on the captcha"""
        try:
            refresh_btn = page.locator('.captcha-refresh, #refresh-btn').first
            if refresh_btn.is_visible():
                refresh_btn.click()
                time.sleep(1)
        except Exception:
            pass

    def _fill_password(self, page: Page, password: str) -> bool:
        """Fill the visible password field and trigger input events."""
        if not password:
            return False

        selectors = [
            'input[type="password"]:visible',
            '#Password:visible',
            'input[name*="password" i]:visible',
            'input[id*="password" i]:visible'
        ]

        try:
            for selector in selectors:
                field = page.locator(selector).first
                if field.count() == 0:
                    continue

                try:
                    field.scroll_into_view_if_needed()
                    field.click(timeout=2000)
                    field.fill("")
                    field.type(password, delay=5)
                    return True
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error filling password: {e}")
        return False

    def _warm_cache_from_screenshots(self, folder: str = "screenshots") -> None:
        pass
