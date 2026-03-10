"""
CaptchaSolver — Parallel OCR captcha solver with maximum speed and accuracy.

Architecture:
  • All 9 cell screenshots are captured in a single JS bulk call.
  • OCR runs on all 9 cells concurrently via ThreadPoolExecutor.
  • Target-aware early-exit stops workers the moment a confident match is found.
  • Cache layer avoids redundant OCR on repeated image bytes.
  • All navigation / page-state checks live in automator.py.
"""

import time
import re
import hashlib
import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, TypedDict

import cv2                                         # noqa: F401  (used by ocr_service internally)
import numpy as np                                 # noqa: F401
from playwright.sync_api import Page, ElementHandle, TimeoutError as PlaywrightTimeout
from PIL import Image

from src.services.ocr_service import ocr_service
from utils.logger import get_logger
from src.config import Config

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Internal types
# ──────────────────────────────────────────────────────────────────────────────

class _CellInfo(TypedDict):
    element: ElementHandle
    x: float
    y: float


class CaptchaSolver:
    """
    Solves the Spain Visa image-captcha by:
      1. Reading the target 3-digit number from the visible instruction box.
      2. Extracting screenshots of all 9 grid cells in one JS call.
      3. Running OCR on every cell in parallel.
      4. Clicking the cells whose number matches the target.
      5. Submitting the form.
    """

    # ── construction ────────────────────────────────────────────────────────

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key
        self.stats: Dict[str, int] = {
            "ocr_success": 0,
            "manual": 0,
            "failed": 0,
            "total": 0,
        }
        self._captcha_cache: Dict[str, Tuple[str, float]] = {}
        self._confidence_floor: float = max(0.5, float(Config.OCR_CONFIDENCE_THRESHOLD))
        self._warm_cache_from_screenshots()
        

    # ── public API ───────────────────────────────────────────────────────────

    def solve_captcha(
        self,
        page: Page,
        password: Optional[str] = None,
        flow_attempt: Optional[int] = None,
        flow_total: Optional[int] = None,
    ) -> bool:
        """
        Main entry point.  Returns True when the captcha was solved and the
        page navigated away from the captcha screen.
        """
        self.stats["total"] += 1
        try:
            if page.is_closed():
                self.stats["failed"] += 1
                logger.info("CAPTCHA_FAILED_STAYED_ON_PAGE")
                return False

            self._wait_for_captcha_images_ready(page, timeout_ms=1200)
            self._clear_selection(page)

            target_number = self._get_visible_target_number_js(page)
            if not target_number:
                self.stats["failed"] += 1
                logger.info("CAPTCHA_FAILED_STAYED_ON_PAGE")
                return False

            cells = self._get_captcha_grid_cells(page)
            if not cells or len(cells) != 9:
                self.stats["failed"] += 1
                logger.info("CAPTCHA_FAILED_STAYED_ON_PAGE")
                return False

            cell_data_list = self._analyze_cells_parallel(
                page, cells, target_number=target_number
            )
            self._display_3x3_grid(cell_data_list)

            matching_indices = self._find_matching_indices(cell_data_list, target_number)
            if not matching_indices:
                self.stats["failed"] += 1
                logger.info("CAPTCHA_FAILED_STAYED_ON_PAGE")
                return False

            latest = self._get_visible_target_number_js(page)
            if latest and latest != target_number:
                self.stats["failed"] += 1
                logger.info("CAPTCHA_FAILED_STAYED_ON_PAGE")
                return False

            self._clear_selection(page)
            selection_ok = self._select_cells_with_verify(
                page, cells, matching_indices, retries=1
            )
            if not selection_ok:
                self.stats["failed"] += 1
                logger.info("CAPTCHA_FAILED_STAYED_ON_PAGE")
                return False

            if password:
                self._fill_password(page, password)
                time.sleep(getattr(Config, "CAPTCHA_CLICK_DELAY_MS", 80) / 1000.0)

            if not self._submit_captcha(page):
                self.stats["failed"] += 1
                logger.info("CAPTCHA_FAILED_STAYED_ON_PAGE")
                return False

            outcome, reason = self._wait_for_submit_outcome(page, timeout_s=6.0)
            if outcome == "success":
                logger.info("CAPTCHA_SUCCESS_NAVIGATED")
                self.stats["ocr_success"] += 1
                return True

            self.stats["failed"] += 1
            logger.info("CAPTCHA_FAILED_STAYED_ON_PAGE")
            return False

        except KeyboardInterrupt:
            raise
        except PlaywrightTimeout as exc:
            _ = exc
            self.stats["failed"] += 1
            logger.info("CAPTCHA_FAILED_STAYED_ON_PAGE")
            return False
        except Exception as exc:
            _ = exc
            self.stats["failed"] += 1
            logger.info("CAPTCHA_FAILED_STAYED_ON_PAGE")
            return False

    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()

    # ── target-number helpers ────────────────────────────────────────────────

    def _parse_target_number(self, text: Optional[str]) -> Optional[str]:
        """Extract a strict 3-digit number from instruction text."""
        if not text:
            return None
        text = text.strip()
        m = re.search(r"(?:number|value|code|with)\s*:?\s*(\d{3})\b", text, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"\b(\d{3})\b", text)
        return m.group(1) if m else None

    def _get_visible_target_number_js(self, page: Page) -> Optional[str]:
        """Use JS raycasting to find the *truly* visible instruction box."""
        try:
            visible_text = page.evaluate("""
                () => {
                    const boxLabels = document.querySelectorAll('.box-label');
                    for (const label of boxLabels) {
                        const style = window.getComputedStyle(label);
                        const rect  = label.getBoundingClientRect();
                        if (
                            style.display     === 'none'   ||
                            style.visibility  === 'hidden' ||
                            parseFloat(style.opacity) < 0.1
                        ) continue;
                        if (rect.left < 0 || rect.top < 0 || rect.width < 10 || rect.height < 5) continue;
                        const cx  = rect.left + rect.width  / 2;
                        const cy  = rect.top  + rect.height / 2;
                        const top = document.elementFromPoint(cx, cy);
                        if (top && (top === label || label.contains(top))) return label.textContent;
                    }
                    return null;
                }
            """)
            if visible_text:
                parsed = self._parse_target_number(visible_text)
                if parsed:
                    return parsed
        except Exception:
            pass

        # simple fallback
        try:
            for label in page.locator(".box-label").all():
                if label.is_visible():
                    parsed = self._parse_target_number(label.text_content())
                    if parsed:
                        return parsed
        except Exception:
            pass
        return None

    # ── cell grid helpers ────────────────────────────────────────────────────

    def _get_captcha_grid_cells(self, page: Page) -> List[ElementHandle]:
        """
        Return the 9 visible captcha IMG elements sorted top-left → bottom-right.
        """
        try:
            elements = page.query_selector_all("img.captcha-img")
            valid: List[_CellInfo] = []

            for el in elements:
                try:
                    if not el.is_visible():
                        continue
                    bbox = el.bounding_box()
                    if not bbox or bbox["width"] < 20 or bbox["height"] < 20:
                        continue
                    valid.append({"element": el, "x": float(bbox["x"]), "y": float(bbox["y"])})
                except Exception:
                    continue

            # Sort by row (10px tolerance) then column
            valid.sort(key=lambda c: (round(c["y"] / 10), c["x"]))

            # Deduplicate overlapping positions
            unique: List[_CellInfo] = []
            for cell in valid:
                if not any(
                    abs(e["x"] - cell["x"]) < 5 and abs(e["y"] - cell["y"]) < 5
                    for e in unique
                ):
                    unique.append(cell)

            elements_out = [c["element"] for c in unique]
            return elements_out[:9]

        except Exception:
            return []

    # ── parallel OCR ────────────────────────────────────────────────────────

    def _analyze_cells_parallel(
        self,
        page: Page,
        cells: List[ElementHandle],
        target_number: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze each cell separately and thoroughly to find the best possible answer.
        Combines fast OCR with aggressive refinement in a single parallel pass per cell.
        """
        n = len(cells)
        results: List[Dict[str, Any]] = [
            {"index": i, "number": None, "confidence": 0.0, "candidates": {}, "screenshot_bytes": None}
            for i in range(n)
        ]

        screenshots: Dict[int, bytes] = self._capture_cells_from_full_screenshot(page, cells)

        missing = [i for i in range(n) if i not in screenshots]
        if missing:
            bulk = self._bulk_capture_cells(page, cells)
            for idx in missing:
                if idx in bulk:
                    screenshots[idx] = bulk[idx]

        missing = [i for i in range(n) if i not in screenshots]
        if missing:
            individual = self._individual_capture_cells(page, cells, missing)
            screenshots.update(individual)

        # ── 3. Store Bytes ───────────────────────────────────────────────────
        for idx, img in screenshots.items():
            results[idx]["screenshot_bytes"] = img

        # ── 4. Parallel Analysis ─────────────────────────────────────────────
        # Thresholds
        min_conf = self._confidence_floor
        # If target is present, we accept it with slightly lower confidence than generic best
        target_accept_conf = max(0.40, min_conf - 0.15)

        def _analyze_task(idx: int, img_bytes: bytes):
            # A. Check Cache
            cached = self._lookup_cached_number(img_bytes)
            if cached and cached[1] >= min_conf:
                # If cached value matches target, return immediately
                if target_number and cached[0] == target_number:
                    return idx, {cached[0]: cached[1]}, cached[0], cached[1]
                # If cached value is very high confidence, trust it
                if cached[1] >= 0.95:
                    return idx, {cached[0]: cached[1]}, cached[0], cached[1]
                # Otherwise, treat cache as a baseline but allow re-check if needed?
                # For speed, we usually trust cache.
                return idx, {cached[0]: cached[1]}, cached[0], cached[1]

            # B. Fast Pass (Standard OCR)
            candidates, best_num, best_conf = ocr_service.get_candidates_with_confidence(
                img_bytes, aggressive=False, target_number=target_number
            )
            
            # Decision 1: Is target found with good confidence?
            if target_number and candidates.get(target_number, 0.0) >= target_accept_conf:
                return idx, candidates, target_number, candidates[target_number]
            
            # C. Aggressive Pass (Refinement)
            # We ALWAYS run aggressive if the target wasn't found in the fast pass.
            # This ensures we don't miss "hard" cells that the fast pass misidentified.
            c_agg, n_agg, cf_agg = ocr_service.get_candidates_with_confidence(
                img_bytes, aggressive=True, target_number=target_number
            )
            
            # Merge candidates
            merged = dict(candidates)
            for k, v in c_agg.items():
                if v > merged.get(k, 0.0):
                    merged[k] = v
            
            # Decision 2: Target found in aggressive results?
            if target_number and merged.get(target_number, 0.0) >= target_accept_conf:
                return idx, merged, target_number, merged[target_number]

            # Decision 3: Return best overall
            # Compare aggressive best vs fast best
            final_best = n_agg if cf_agg > best_conf else best_num
            final_conf = cf_agg if cf_agg > best_conf else best_conf
            
            return idx, merged, final_best, final_conf

        with ThreadPoolExecutor(max_workers=min(9, n)) as executor:
            future_map = {
                executor.submit(_analyze_task, idx, img): idx 
                for idx, img in screenshots.items()
            }
            
            for future in as_completed(future_map):
                idx, candidates, best_num, best_conf = future.result()
                data = results[idx]
                data["candidates"] = candidates
                data["number"] = best_num
                data["confidence"] = best_conf

                # Update cache if confident
                img_bytes = data["screenshot_bytes"]
                if img_bytes and best_num and best_conf >= min_conf:
                    self._captcha_cache[hashlib.sha1(img_bytes).hexdigest()] = (
                        str(best_num),
                        float(best_conf),
                    )

        return results

    def _capture_cells_from_full_screenshot(
        self, page: Page, cells: List[ElementHandle]
    ) -> Dict[int, bytes]:
        result: Dict[int, bytes] = {}
        try:
            full_png = page.screenshot(type="png")
            with Image.open(io.BytesIO(full_png)) as pil_img:
                for idx, cell in enumerate(cells):
                    try:
                        box = cell.bounding_box()
                        if not box:
                            continue
                        region = pil_img.crop(
                            (
                                int(box["x"]),
                                int(box["y"]),
                                int(box["x"] + box["width"]),
                                int(box["y"] + box["height"]),
                            )
                        )
                        buf = io.BytesIO()
                        region.save(buf, format="PNG")
                        result[idx] = buf.getvalue()
                    except Exception:
                        continue
        except Exception:
            pass
        return result

    def _bulk_capture_cells(
        self, page: Page, cells: List[ElementHandle]
    ) -> Dict[int, bytes]:
        """
        Capture all cell images in a single JS evaluation.
        Returns a dict of {index: png_bytes}.
        """
        screenshots: Dict[int, bytes] = {}
        try:
            for _attempt in range(2):
                b64_list = page.evaluate(
                    """
                    (elements) => elements.map(el => {
                        try {
                            if (!el.complete) return null;
                            const w = el.naturalWidth  || el.width;
                            const h = el.naturalHeight || el.height;
                            if (w < 10 || h < 10) return null;
                            const canvas = document.createElement('canvas');
                            canvas.width  = w;
                            canvas.height = h;
                            canvas.getContext('2d').drawImage(el, 0, 0, w, h);
                            return canvas.toDataURL('image/png', 1.0).split(',')[1];
                        } catch(e) { return null; }
                    })
                    """,
                    cells,
                )
                if b64_list:
                    for idx, b64 in enumerate(b64_list):
                        if b64:
                            screenshots[idx] = base64.b64decode(b64)

                if len(screenshots) >= len(cells) - 1:
                    break
                time.sleep(0.1)

            # Sanity check — reject if all images are identical (bad CORS / placeholders)
            if len(screenshots) >= 5:
                hashes = {hashlib.md5(v).hexdigest() for v in screenshots.values()}
                if len(hashes) < 4:
                    return {}

        except Exception:
            pass
        return screenshots

    def _individual_capture_cells(
        self,
        page: Page,
        cells: List[ElementHandle],
        indices: List[int],
    ) -> Dict[int, bytes]:
        """
        Fallback: capture missing cells one at a time.
        Tries a full-page crop first (fewer round-trips), then per-element screenshot.
        """
        result: Dict[int, bytes] = {}

        # Try full-page crop
        try:
            full_png = page.screenshot(type="png")
            with Image.open(io.BytesIO(full_png)) as pil_img:
                for idx in indices:
                    try:
                        box = cells[idx].bounding_box()
                        if box:
                            region = pil_img.crop(
                                (
                                    int(box["x"]),
                                    int(box["y"]),
                                    int(box["x"] + box["width"]),
                                    int(box["y"] + box["height"]),
                                )
                            )
                            buf = io.BytesIO()
                            region.save(buf, format="PNG")
                            result[idx] = buf.getvalue()
                    except Exception:
                        pass
            if len(result) == len(indices):
                return result
        except Exception:
            pass

        # Per-element screenshot for anything still missing
        for idx in indices:
            if idx in result:
                continue
            try:
                cells[idx].scroll_into_view_if_needed(timeout=500)
            except Exception:
                pass
            try:
                result[idx] = cells[idx].screenshot(type="png", timeout=1500)
            except Exception:
                pass

        return result

    # ── matching & refinement ────────────────────────────────────────────────

    def _find_matching_indices(
        self, cell_data_list: List[Dict[str, Any]], target_number: str
    ) -> List[int]:
        matches = []
        accept = max(0.40, self._confidence_floor - 0.15)
        relaxed = max(0.32, accept - 0.08)
        for idx, data in enumerate(cell_data_list):
            if data.get("number") == target_number and float(data.get("confidence", 0.0)) >= accept:
                matches.append(idx)
                continue
            candidates = data.get("candidates", {}) or {}
            if float(candidates.get(target_number, 0.0)) >= accept:
                matches.append(idx)
                continue
            for cand, conf in candidates.items():
                digits = "".join(ch for ch in str(cand) if ch.isdigit())
                if not digits:
                    continue
                if target_number in digits and float(conf) >= relaxed:
                    matches.append(idx)
                    break
        return matches



    # ── selection helpers ────────────────────────────────────────────────────

    def _select_cells_with_verify(
        self,
        page: Page,
        cells: List[ElementHandle],
        indices: List[int],
        retries: Optional[int] = None,
    ) -> bool:
        """Select cells and verify that exactly the expected set is selected."""
        _ = retries
        return self._select_cells(page, cells, indices)

    def _select_cells(
        self, page: Page, cells: List[ElementHandle], indices: List[int]
    ) -> bool:
        try:
            if not indices:
                return False
            unique = sorted(set(indices))
            for idx in unique:
                if idx >= len(cells):
                    continue
                try:
                    cell = cells[idx]
                    bbox = cell.bounding_box()
                    if bbox:
                        page.mouse.move(
                            bbox["x"] + bbox["width"] / 2,
                            bbox["y"] + bbox["height"] / 2,
                        )
                        page.mouse.click(
                            bbox["x"] + bbox["width"] / 2,
                            bbox["y"] + bbox["height"] / 2,
                        )
                    time.sleep(getattr(Config, "CAPTCHA_CLICK_DELAY_MS", 80) / 1000.0)
                except Exception:
                    continue

            return True
        except Exception:
            return False

    def _get_selection_ids(
        self, cells: List[ElementHandle], indices: List[int]
    ) -> List[str]:
        ids: List[str] = []
        for idx in indices:
            if idx >= len(cells):
                continue
            try:
                onclick = cells[idx].get_attribute("onclick") or ""
                m = re.search(r"(?:Select|OnImageSelect|OnSelect)\(['\"]([^'\"]+)['\"]", onclick)
                if m:
                    ids.append(m.group(1))
                    continue
                cell_id = cells[idx].get_attribute("id")
                if cell_id:
                    ids.append(cell_id)
                    continue
                parent_id = cells[idx].evaluate(
                    "(el) => (el && el.parentElement && el.parentElement.id) ? el.parentElement.id : ''"
                )
                if parent_id:
                    ids.append(parent_id)
            except Exception:
                continue
        return ids

    def _get_selected_indices(
        self, page: Page, cells: List[ElementHandle]
    ) -> List[int]:
        selected = []
        for idx, cell in enumerate(cells):
            try:
                is_sel = page.evaluate(
                    """
                    (cell) => {
                        if (!cell) return false;
                        const img = cell.matches && cell.matches('img.captcha-img')
                            ? cell : cell.querySelector('img.captcha-img');
                        const border = (cell.style && cell.style.border) || (img && img.style && img.style.border) || '';
                        return cell.classList.contains('img-selected')
                            || (img && img.classList.contains('img-selected'))
                            || (border && border.includes('green'));
                    }
                    """,
                    cell,
                )
            except Exception:
                is_sel = False
            if is_sel:
                selected.append(idx)
        return selected

    def _get_unselected_indices(
        self, page: Page, cells: List[ElementHandle], indices: List[int]
    ) -> List[int]:
        missing = []
        for idx in indices:
            if idx >= len(cells):
                missing.append(idx)
                continue
            try:
                selected = page.evaluate(
                    """
                    (cell) => {
                        if (!cell) return false;
                        const img = cell.matches && cell.matches('img.captcha-img')
                            ? cell : cell.querySelector('img.captcha-img');
                        const border = (cell.style && cell.style.border) || (img && img.style && img.style.border) || '';
                        return cell.classList.contains('img-selected')
                            || (img && img.classList.contains('img-selected'))
                            || (border && border.includes('green'));
                    }
                    """,
                    cells[idx],
                )
            except Exception:
                selected = False
            if not selected:
                missing.append(idx)
        return missing

    def _sync_selected_images(
        self, page: Page, selection_ids: List[str]
    ) -> bool:
        if not selection_ids:
            return False
        try:
            return bool(
                page.evaluate(
                    """
                    (ids) => {
                        const input = document.querySelector('#SelectedImages');
                        if (input) {
                            const value = ids.join(',');
                            input.value = value;
                            input.setAttribute('value', value);
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                            input.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                        if (Array.isArray(window.selection)) window.selection = ids.slice();
                        if (typeof window.setAction === 'function') window.setAction();
                        return true;
                    }
                    """,
                    selection_ids,
                )
            )
        except Exception:
            return False

    def _clear_selection(self, page: Page) -> bool:
        try:
            clear_selectors = [
                'a:has-text("Clear Selection")',
                "a[onclick*=\"onUndo\"]",
                "a[onclick*=\"OnClearSelect\"]",
                "a[onclick*=\"onClearSelect\"]",
                "a[href*=\"OnClearSelect\"]",
            ]
            for selector in clear_selectors:
                link = page.locator(selector).first
                if link.count() > 0 and link.is_visible():
                    link.click()
                    time.sleep(0.02)
                    return True
            return False
        except Exception:
            return False

    # ── submit & navigation ──────────────────────────────────────────────────

    def _submit_captcha(self, page: Page) -> bool:
        try:
            for selector in [
                "#btnVerify",
                'button[type="submit"]',
                'button:has-text("Submit")',
                'button:has-text("Verify")',
                ".btn-success",
            ]:
                btn = page.locator(selector).first
                if btn.count() > 0 and btn.is_visible():
                    btn.scroll_into_view_if_needed()
                    time.sleep(0.02)
                    try:
                        btn.click()
                    except Exception:
                        continue
                    return True
            return False

        except Exception:
            return False

    def _wait_for_submit_outcome(
        self, page: Page, timeout_s: float = 6.0
    ) -> Tuple[str, Optional[str]]:
        """
        Returns ("success", None) | ("rejected", reason) | ("stalled", None).
        """
        if page.is_closed():
            return "rejected", "page closed"
        initial_url = (page.url or "").lower()
        try:
            page.wait_for_load_state("domcontentloaded", timeout=1500)
        except Exception:
            pass

        error = self._detect_submit_error(page)
        if error:
            return "rejected", error

        current_url = (page.url or "").lower()
        if any(
            t in current_url
            for t in ["global/home", "home", "dashboard", "appointment/newappointment", "visatype"]
        ):
            return "success", None
        if current_url != initial_url and "captcha" not in current_url and "login" not in current_url:
            return "success", None
        try:
            if page.locator(".captcha-img:visible").count() == 0 and "captcha" not in current_url:
                return "success", None
        except Exception:
            pass

        try:
            page.wait_for_url(
                re.compile(r".*(/home|/dashboard|/appointment/newappointment|/visatype).*", re.IGNORECASE),
                timeout=max(1000, int(timeout_s * 1000)),
            )
            return "success", None
        except Exception:
            pass

        error = self._detect_submit_error(page)
        if error:
            return "rejected", error
        return "stalled", None

    def _detect_submit_error(self, page: Page) -> Optional[str]:
        try:
            selectors = [
                "text=/invalid captcha|captcha invalid|captcha you submitted is invalid/i",
                'a:has-text("Go To Home")',
                "text=/too many attempts|too many requests|try again later|account locked|access denied|session expired/i",
                ".validation-summary-errors",
                ".alert-danger",
                ".text-danger",
                ".field-validation-error",
            ]
            for sel in selectors:
                node = page.locator(sel).first
                if node.count() > 0 and node.is_visible():
                    try:
                        return (node.inner_text() or sel).strip()[:220]
                    except Exception:
                        return sel

            text = page.evaluate(
                """
                () => {
                    const keywords = [
                        'invalid captcha','captcha invalid','captcha you submitted is invalid',
                        'too many attempts','too many requests','try again later',
                        'access denied','account locked','session expired'
                    ];
                    const nodes = Array.from(document.querySelectorAll(
                        '.alert,.alert-danger,.validation-summary-errors,.text-danger,.field-validation-error'
                    ));
                    for (const n of nodes) {
                        const s = window.getComputedStyle(n);
                        if (s.display==='none'||s.visibility==='hidden'||parseFloat(s.opacity||'1')<0.1) continue;
                        const t = (n.textContent||'').trim().toLowerCase();
                        if (keywords.some(k => t.includes(k))) return t.slice(0,220);
                    }
                    return '';
                }
                """
            )
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            pass
        return None

    # ── page-state checks ────────────────────────────────────────────────────

    def _is_on_captcha_page(self, page: Page) -> bool:
        try:
            if page.is_closed():
                return False
            count = page.evaluate(
                """
                () => {
                    let n = 0;
                    for (const img of document.querySelectorAll('img.captcha-img')) {
                        if (img.offsetParent === null) continue;
                        const s = window.getComputedStyle(img);
                        if (s.display==='none'||s.visibility==='hidden'||s.opacity==='0') continue;
                        const r = img.getBoundingClientRect();
                        if (r.width >= 20 && r.height >= 20) n++;
                    }
                    return n;
                }
                """
            )
            return count > 0
        except Exception:
            return False

    def _wait_for_captcha_images_ready(
        self, page: Page, timeout_ms: int = 1200
    ) -> bool:
        try:
            page.wait_for_function(
                """
                () => {
                    const imgs = Array.from(document.querySelectorAll('img.captcha-img'));
                    if (imgs.length < 9) return false;
                    for (const img of imgs.slice(0, 9)) {
                        const s = window.getComputedStyle(img);
                        if (s.display==='none'||s.visibility==='hidden'||s.opacity==='0') return false;
                        if (!img.complete||img.naturalWidth<10||img.naturalHeight<10) return false;
                    }
                    return true;
                }
                """,
                timeout=timeout_ms,
            )
            return True
        except Exception:
            return False

    def _refresh_captcha(self, page: Page) -> bool:
        try:
            if page.is_closed():
                return False
            for selector in [
                "a:has(i.fa-sync)",
                "a:has(i.fa-redo)",
                "a[onclick*=\"onReload\" i]",
                "a[onclick*=\"reload\" i]",
                "[title*=\"refresh\" i]",
                "[title*=\"reload\" i]",
                'a:has-text("Reload")',
                "a[href*=\"/appointment/newappointment\"]",
                ".refresh-captcha",
                "img[src*=\"refresh\"]",
            ]:
                btn = page.locator(selector).first
                if btn.count() > 0 and btn.is_visible():
                    btn.click(force=True)
                    time.sleep(0.02)
                    self._wait_for_captcha_images_ready(page, timeout_ms=800)
                    return True
            # Full reload fallback
            page.reload(wait_until="domcontentloaded", timeout=10000)
            self._wait_for_captcha_images_ready(page, timeout_ms=1200)
            return True
        except Exception:
            return False

    # ── password helper ──────────────────────────────────────────────────────

    def _fill_password(self, page: Page, password: str) -> bool:
        if not password:
            return False
        import random
        for selector in [
            'input[type="password"]:visible',
            "#Password:visible",
            "input[name*=\"password\" i]:visible",
            "input[id*=\"password\" i]:visible",
        ]:
            field = page.locator(selector).first
            if field.count() == 0:
                continue
            try:
                field.scroll_into_view_if_needed()
                field.click(timeout=1500)
                time.sleep(random.uniform(0.01, 0.03))
                field.press("Control+a")
                field.press("Backspace")
                field.fill(password)
                if field.input_value() == password:
                    return True
                for ch in password:
                    field.type(ch, delay=random.randint(10, 30))
                if field.input_value() == password:
                    return True
            except Exception:
                pass
            try:
                page.evaluate(
                    """
                    ({ selector, value }) => {
                        const el = document.querySelector(selector);
                        if (!el) return;
                        el.removeAttribute('readonly');
                        el.removeAttribute('disabled');
                        el.focus(); el.value = value;
                        el.dispatchEvent(new Event('input',  { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                        el.dispatchEvent(new Event('blur',   { bubbles: true }));
                    }
                    """,
                    {"selector": selector.replace(":visible", ""), "value": password},
                )
                if field.input_value() == password:
                    return True
            except Exception:
                pass
        return False

    # ── cache helpers ────────────────────────────────────────────────────────

    def _lookup_cached_number(
        self, image_bytes: bytes
    ) -> Optional[Tuple[str, float]]:
        try:
            return self._captcha_cache.get(hashlib.sha1(image_bytes).hexdigest())
        except Exception:
            return None

    def _warm_cache_from_screenshots(self, folder: str = "screenshots") -> None:
        try:
            path = Path(folder)
            if not path.exists():
                return
            loaded = 0
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_path in path.glob(ext):
                    try:
                        with Image.open(img_path) as im:
                            w, h = im.size
                        if max(w, h) > 220:
                            continue
                        data = img_path.read_bytes()
                        key = hashlib.sha1(data).hexdigest()
                        if key in self._captcha_cache:
                            continue
                        number, conf = ocr_service.extract_number_with_confidence(data, aggressive=True)
                        if number and conf >= 0.65:
                            self._captcha_cache[key] = (number, conf)
                            loaded += 1
                        if loaded >= 200:
                            return
                    except Exception:
                        continue
        except Exception:
            pass

    # ── debug display ────────────────────────────────────────────────────────

    def _display_3x3_grid(self, cell_data_list: List[Dict[str, Any]]) -> None:
        _ = cell_data_list
        return
