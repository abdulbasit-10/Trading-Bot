"""
SpainVisaAutomator — All browser navigation & page-state logic lives here.
=========================================================================

Design principles:
  • Every navigate_*, _is_*, _wait_for_* method belongs in this file.
  • CaptchaSolver owns only OCR + cell-selection logic.
  • No time.sleep(N) > 0.5s except in clearly labelled back-off points.
  • All page-state helpers are private methods of this class.
"""

import random
import time
import re
import csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from playwright.sync_api import (
    Playwright,
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeout,
)

from src.config import Config
from src.models.captcha_solver import CaptchaSolver
from src.models.field_detector import FieldDetector
from src.models.applicant import Applicant
from src.services.email_service import EmailService
from utils.logger import get_logger, setup_logging
from utils.helpers import save_screenshot, save_result
from src.pages.appointment_page import handle_appointment_booking, wait_for_appointment_page

logger = get_logger(__name__)


class SpainVisaAutomator:
    """
    Drives the full Spain Visa appointment booking flow:
      Login → Captcha → Home → Book New Appointment → Captcha → Visa-Type → Booking
    """

    # ── construction ────────────────────────────────────────────────────────

    def __init__(
        self,
        headless: bool = Config.HEADLESS,
        max_retries: int = Config.MAX_RETRIES,
        gemini_api_key: Optional[str] = Config.GEMINI_API_KEY,
    ):
        self.headless = headless
        self.max_retries = max_retries
        self.timeout = Config.TIMEOUT

        self.captcha_solver = CaptchaSolver(gemini_api_key)
        self.field_detector = FieldDetector()

        self._current_password: Optional[str] = None

        self.stats = {
            "total_applicants": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0,
        }
        self.results: List[Dict[str, Any]] = []
        setup_logging()

    # ── public API ───────────────────────────────────────────────────────────

    def book_appointment(self, data: Applicant, browser: Browser) -> bool:
        applicant = data if isinstance(data, Applicant) else Applicant.from_dict(data)
        return self._run_booking_flow(browser, applicant)

    # ── main booking flow ────────────────────────────────────────────────────

    def _run_booking_flow(self, browser: Browser, applicant: Applicant) -> bool:
        context: Optional[BrowserContext] = None
        page: Optional[Page] = None
        should_close_on_exit = True

        try:
            context = self._create_context(browser)

            for attempt in range(self.max_retries):
                logger.info(f"\n{'='*50}")
                logger.info(f"Attempt {attempt + 1}/{self.max_retries} — {applicant.email}")
                logger.info(f"{'='*50}")

                # Close previous page if any
                if page:
                    try:
                        page.close()
                    except Exception:
                        pass

                page = context.new_page()
                page.set_default_timeout(self.timeout)
                self._attach_dialog_handler(page)
                self._current_password = applicant.password

                try:
                    # ── Step 1: navigate to login ─────────────────────────
                    if not self._navigate_to_login(page):
                        raise RuntimeError("Failed to reach login page")

                    # ── Step 2: enter email ───────────────────────────────
                    if not self._handle_email_page(page, applicant.email):
                        logger.error("Email page failed — retrying")
                        continue

                    # ── Step 3: optional verification step ───────────────
                    if not self._handle_optional_verify_step(page):
                        logger.error("Verification step failed — retrying")
                        continue

                    # ── Step 4: wait for captcha page ─────────────────────
                    if not self._wait_for_captcha_page(page):
                        logger.error("Captcha page not reached — retrying")
                        continue

                    # ── Step 5: solve login captcha ───────────────────────
                    if not self._solve_captcha_and_handle_password(page, applicant.password):
                        logger.error("Login captcha solving failed. Keeping page open as requested.")
                        should_close_on_exit = False
                        return False

                    logger.info(f"✅ Login captcha submitted — {applicant.email}")
                    save_screenshot(page, f"success_{applicant.email.replace('@','_')}")

                    # ── Step 6: navigate to index page ────────────────────
                    if not self._is_index_page(page) and not self._has_book_new_appointment_link(page):
                        try:
                            page.goto(
                                "https://appointment.thespainvisa.com/",
                                wait_until="domcontentloaded",
                                timeout=15000,
                            )
                        except Exception:
                            pass
                    if not self._is_index_page(page) and not self._has_book_new_appointment_link(page):
                        raise RuntimeError("Failed to reach index page after captcha")

                    # ── Step 7: ensure we're on Book New Appointment ──────
                    # Skip navigation when already there; avoid redundant logs.
                    if not self._ensure_book_new_appointment_page(page):
                        logger.warning(
                            "Proceeding without confirmed 'Book New Appointment' navigation; "
                            "downstream checks will validate the current page."
                        )

                    # ── Step 8: captcha on new-appointment page (if any) ───
                    if not self._handle_new_appointment_captcha(page):
                        raise RuntimeError("New-appointment captcha failed")

                    # ── Step 9: visa-type form ────────────────────────────
                    visa_ok = self._handle_visa_type_page(page, applicant)
                    # In some cases the visa-type handler returns False even
                    # though the site has already navigated to the slots page.
                    # Before aborting, explicitly wait for the appointment page.
                    if not visa_ok and not wait_for_appointment_page(page):
                        raise RuntimeError("Visa-type form handling failed")

                    # ── Step 10: booking (Slots + OTP + Verification) ─────
                    if not handle_appointment_booking(page, applicant):
                        raise RuntimeError("Appointment booking flow failed")

                    return True

                except KeyboardInterrupt:
                    raise
                except PlaywrightTimeout as exc:
                    logger.error(f"⏰ Timeout: {exc}")
                    save_screenshot(page, f"timeout_{applicant.email.replace('@','_')}")
                except Exception as exc:
                    logger.error(f"❌ Attempt {attempt + 1} failed: {str(exc)[:200]}")
                    save_screenshot(page, f"error_{applicant.email.replace('@','_')}")

                if attempt < self.max_retries - 1:
                    wait = self._get_wait_time(attempt)
                    logger.info(f"⏳ Waiting {wait:.1f}s before retry…")
                    time.sleep(wait)

            logger.error(f"❌ All attempts failed — {applicant.email}")
            return False

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.error(f"Fatal booking error: {exc}")
            return False
        finally:
            if should_close_on_exit:
                for obj in (page, context):
                    if obj:
                        try:
                            obj.close()
                        except Exception:
                            pass

    # ══════════════════════════════════════════════════════════════════════════
    # NAVIGATION METHODS  (all navigate_* / _navigate_* live here)
    # ══════════════════════════════════════════════════════════════════════════

    def _navigate_to_login(self, page: Page) -> bool:
        """Navigate to the Spain Visa login page."""
        logger.info("🌐 Navigating to Spain Visa login…")

        urls = [
            Config.LOGIN_URL,
            "https://appointment.thespainvisa.com/Global/account/login",
            "https://appointment.thespainvisa.com/",
        ]

        for url in urls:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                
                # Check if we are already logged in or on the login page
                if self._is_logged_in(page):
                    logger.info("✅ Already logged in")
                    return True

                lower_url = (page.url or "").lower()
                if (
                    "login" in lower_url
                    or "account" in lower_url
                    or self._is_captcha_visible(page)
                    or page.locator('input[type="email"]:visible').count() > 0
                    or page.locator('input[type="password"]:visible').count() > 0
                ):
                    logger.info(f"✅ Login page loaded from {url}")
                    return True

            except Exception as exc:
                logger.warning(f"Failed to load {url}")

        logger.error("❌ Could not load any login page")
        return False

    def _navigate_to_new_appointment(self, page: Page) -> bool:
        """Navigate to the Book New Appointment page after login."""
        
        target = re.compile(r"/global/appointment/newappointment", re.IGNORECASE)
        if target.search(page.url or ""):
            logger.info("✅ Already on Book New Appointment page")
            return True
            
        logger.info("🧭 Navigating to Book New Appointment…")

        # Try direct navigation first as it's fastest
        try:
            page.goto(
                "https://appointment.thespainvisa.com/Global/appointment/newappointment",
                wait_until="domcontentloaded",
                timeout=10000,
            )
            if target.search(page.url or ""):
                logger.info("✅ Opened via direct URL")
                return True
        except Exception:
            pass

        # Fallback to clicking links
        selectors = [
            'a[href*="/Global/appointment/newappointment"]:has-text("Book Now")',
            'a[href*="/Global/appointment/newappointment"]:has-text("Book New Appointment")',
            'a:has-text("Book New Appointment")',
            'a:has-text("Book Now")',
        ]

        for selector in selectors:
            try:
                links = page.locator(selector)
                if links.count() > 0 and links.first.is_visible():
                    links.first.click(timeout=3000)
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass
                    if target.search(page.url or ""):
                        logger.info("✅ Opened Book New Appointment page")
                        return True
            except Exception:
                continue

        logger.warning("⚠️ Could not reach Book New Appointment page")
        return False

    def _navigate_back_to_home_after_captcha_failure(self, page: Page) -> bool:
        """Handle the 'Go To Home' page that appears after a failed captcha."""
        try:
            go_home = page.locator('a:has-text("Go To Home")').first
            if go_home.count() > 0 and go_home.is_visible():
                try:
                    go_home.scroll_into_view_if_needed()
                except Exception:
                    pass
                try:
                    go_home.click()
                except Exception:
                    page.evaluate(
                        """
                        () => {
                            const a = Array.from(document.querySelectorAll('a'))
                                .find(el => (el.textContent||'').includes('Go To Home'));
                            if (a) a.click();
                        }
                        """
                    )
                try:
                    page.wait_for_url(
                        re.compile(r".*/$|.*/index.*", re.IGNORECASE), timeout=10000
                    )
                except Exception:
                    pass
                return True
            # Fallback: navigate to root
            page.goto(
                "https://appointment.thespainvisa.com/",
                wait_until="domcontentloaded",
                timeout=15000,
            )
            return True
        except Exception:
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # CAPTCHA FLOW ORCHESTRATION
    # ══════════════════════════════════════════════════════════════════════════

    def _solve_captcha_and_handle_password(self, page: Page, password: str) -> bool:
        """Orchestrate captcha solving for the login page."""
        logger.info("🔐 Starting login captcha flow…")
        if page.is_closed():
            return False
        attempt = 0
        while not page.is_closed():
            attempt += 1
            solved = self.captcha_solver.solve_captcha(
                page,
                password=password,
                flow_attempt=attempt,
                flow_total=999,
            )
            if solved:
                deadline = time.time() + 4
                while time.time() < deadline:
                    if page.is_closed():
                        return False
                    if self._is_go_home_page(page) or self._is_captcha_invalid_banner(page):
                        break
                    if not self._is_password_captcha_page(page) and not self._is_captcha_visible(page):
                        break
                    time.sleep(0.02)

                current_url = (page.url or "").lower()
                if self._is_password_captcha_page(page) and self._is_captcha_visible(page):
                    # We might just need to refresh and try again
                    pass
                elif "captcha" in current_url:
                    pass
                else:
                    self.captcha_solver.stats["ocr_success"] += 1
                    return True

            if self._is_password_captcha_page(page) or self._is_captcha_visible(page):
                self._refresh_captcha(page)
                continue

            if page.locator('input[type="email"]:visible').count() > 0:
                try:
                    page.go_back(wait_until="domcontentloaded", timeout=5000)
                except Exception:
                    pass
                if self._is_password_captcha_page(page) or self._is_captcha_visible(page):
                    self._refresh_captcha(page)
                    continue

            if self._is_go_home_page(page):
                try:
                    page.go_back(wait_until="domcontentloaded", timeout=5000)
                except Exception:
                    pass
                if self._is_password_captcha_page(page) or self._is_captcha_visible(page):
                    self._refresh_captcha(page)
                    continue

            time.sleep(0.1)
            if attempt >= max(3, getattr(Config, "CAPTCHA_SELECTION_RETRIES", 3)):
                attempt = 0
        return False

    def _handle_new_appointment_captcha(self, page: Page) -> bool:
        """Solve the captcha on the Book New Appointment page (no password)."""
        url = (page.url or "").lower()
        # If already past captcha (visa form or slots page visible), proceed
        if "appointment/newappointment" in url:
            if self._is_visa_type_form_visible(page):
                logger.info("✅ Visa Type form visible — skipping captcha")
                return True
            slot_label = page.locator('label.form-label:has-text("Appointment Date")').first
            if slot_label.count() > 0 and slot_label.is_visible():
                logger.info("✅ Slots page visible — skipping captcha")
                return True

        if not self._is_captcha_visible(page):
            if self._is_post_captcha_destination(page):
                logger.info("✅ No captcha on new-appointment page — already past it")
                return True

        logger.info("🔐 Handling Book New Appointment captcha…")

        try:
            solved = self.captcha_solver.solve_captcha(
                page, flow_attempt=1, flow_total=1
            )
        except Exception as exc:
            logger.error(f"New-appt captcha error: {exc}")
            return False

        if not solved:
            # Captcha solver failed — check if we can proceed anyway
            if self._is_visa_type_form_visible(page):
                logger.info("✅ Visa Type form visible after captcha attempt — proceeding")
                return True
            if page.locator('label.form-label:has-text("Appointment Date")').first.count() > 0:
                logger.info("✅ Slots page visible after captcha attempt — proceeding")
                return True
            return False

        self.captcha_solver.stats["ocr_success"] += 1
        if self._is_go_home_page(page):
            self._navigate_back_to_home_after_captcha_failure(page)
            return False
            
        return self._wait_for_post_captcha_result(page)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE-STATE CHECKS  (private)
    # ══════════════════════════════════════════════════════════════════════════

    def _is_captcha_visible(self, page: Page) -> bool:
        try:
            if page.is_closed():
                return False
            if page.locator(".captcha-div").first.count() > 0 and page.locator(".captcha-div").first.is_visible():
                return True
            return (
                page.locator(".captcha-img:visible").count() > 0
                and page.locator(".box-label:visible").count() > 0
            )
        except Exception:
            return False

    def _is_password_captcha_page(self, page: Page) -> bool:
        try:
            if page.is_closed():
                return False
            if re.search(r"(captcha|logincaptcha|newcaptcha)", page.url or "", re.IGNORECASE):
                return True
            if page.locator("#captchaForm").first.count() > 0 and page.locator("#captchaForm").first.is_visible():
                return True
            if page.locator("#SelectedImages").count() > 0 and page.locator(".captcha-img").count() > 0:
                return True
            if page.locator('input[type="password"]').count() > 0 and page.locator(".box-label").count() > 0:
                return True
            return False
        except Exception:
            return False

    def _is_logged_in(self, page: Page) -> bool:
        try:
            if page.is_closed():
                return False
            if self._is_go_home_page(page) or self._is_captcha_invalid_banner(page):
                return False
            if self._is_captcha_visible(page):
                return False
            url = page.url.lower()
            if any(p in url for p in ["/home", "/dashboard", "/global/home", "/account/dashboard"]):
                return True
            if self._has_book_new_appointment_link(page) or self._has_logout_link(page):
                return True
            has_login_els = (
                page.locator('input[type="password"]:visible').count() > 0
                or page.locator(".captcha-img:visible").count() > 0
                or page.locator('input[type="email"]:visible').count() > 0
            )
            return not has_login_els and "login" not in url and "captcha" not in url
        except Exception:
            return False

    def _is_go_home_page(self, page: Page) -> bool:
        try:
            go_home = page.locator('a:has-text("Go To Home")').first
            alert = page.locator('text="The captcha you submitted is invalid"').first
            return (go_home.count() > 0 and go_home.is_visible()) or (
                alert.count() > 0 and alert.is_visible()
            )
        except Exception:
            return False

    def _is_index_page(self, page: Page) -> bool:
        try:
            url = page.url or ""
            if re.search(r"/$|/index|/global/home", url, re.IGNORECASE):
                return True
            if page.locator('a:has-text("Book Now")').first.count() > 0 and page.locator('a:has-text("Book Now")').first.is_visible():
                return True
            h = page.locator('text="Apply for VISA to Spain In Pakistan"').first
            return h.count() > 0 and h.is_visible()
        except Exception:
            return False

    def _is_post_captcha_destination(self, page: Page) -> bool:
        try:
            url = (page.url or "").lower()
            if self._is_captcha_invalid_banner(page):
                return False
            if self._is_password_captcha_page(page) and self._is_captcha_visible(page):
                return False
            if any(t in url for t in ["/global/home", "/home", "/dashboard", "/account/dashboard"]):
                return True
            if "appointment/newappointment" in url:
                return True
            if self._is_visa_type_form_visible(page):
                return True
            return self._is_index_page(page) or self._has_book_new_appointment_link(page)
        except Exception:
            return False

    def _is_captcha_invalid_banner(self, page: Page) -> bool:
        try:
            selectors = [
                'text="The captcha you submitted is invalid"',
                "text=/invalid captcha|captcha invalid|captcha you submitted is invalid/i",
                "text=/too many attempts|too many requests|try again later|account locked|access denied|session expired/i",
                ".validation-summary-errors",
                ".alert-danger",
                ".text-danger",
                ".field-validation-error",
            ]
            for selector in selectors:
                alert = page.locator(selector).first
                if alert.count() > 0 and alert.is_visible():
                    try:
                        text = (alert.inner_text() or "").strip().lower()
                    except Exception:
                        text = ""
                    if not text:
                        return True
                    if any(
                        t in text
                        for t in [
                            "captcha", "invalid", "too many", "locked",
                            "access denied", "session expired",
                        ]
                    ):
                        return True
            return bool(
                page.evaluate(
                    """
                    () => {
                        const kw = ['captcha','invalid','too many','locked','access denied','session expired'];
                        const nodes = Array.from(document.querySelectorAll(
                            '.alert,.alert-danger,.validation-summary-errors,.text-danger,.field-validation-error'
                        ));
                        for (const n of nodes) {
                            const s = window.getComputedStyle(n);
                            if (s.display==='none'||s.visibility==='hidden'||parseFloat(s.opacity||'1')<0.1) continue;
                            const t = (n.textContent||'').trim().toLowerCase();
                            if (t && kw.some(k => t.includes(k))) return true;
                        }
                        return false;
                    }
                    """
                )
            )
        except Exception:
            return False

    def _is_verification_step(self, page: Page) -> bool:
        try:
            if self._is_password_captcha_page(page) or self._is_captcha_visible(page):
                return False
            url = (page.url or "").lower()
            if "verification" in url or "dataprotectionemailsent" in url:
                return True
            btn = page.locator('button:has-text("Verify"), button:has-text("Continue")').first
            if btn.count() > 0 and btn.is_visible():
                return True
            h = page.locator("text=/verification|data protection/i").first
            return h.count() > 0 and h.is_visible()
        except Exception:
            return False

    def _is_visa_type_form_visible(self, page: Page) -> bool:
        try:
            if page.locator('form[action*="/Global/Appointment/VisaType"]').first.count() > 0 and page.locator('form[action*="/Global/Appointment/VisaType"]').first.is_visible():
                return True
            lbl = page.locator('label.form-label:has-text("Visa Type")').first
            return lbl.count() > 0 and lbl.is_visible()
        except Exception:
            return False

    def _has_logout_link(self, page: Page) -> bool:
        try:
            for selector in [
                'a:has-text("Logout")',
                'a:has-text("Sign Out")',
                'button:has-text("Logout")',
                'button:has-text("Sign Out")',
            ]:
                if page.locator(selector).first.count() > 0 and page.locator(selector).first.is_visible():
                    return True
            return False
        except Exception:
            return False

    def _has_book_new_appointment_link(self, page: Page) -> bool:
        try:
            lnk = page.locator('a[href*="/Global/appointment/newappointment"]').first
            return lnk.count() > 0 and lnk.is_visible()
        except Exception:
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # WAIT HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _wait_for_captcha_page(self, page: Page) -> bool:
        logger.info("⏳ Waiting for CAPTCHA page…")
        deadline = time.time() + Config.CAPTCHA_PAGE_WAIT_MS / 1000.0

        while time.time() < deadline:
            try:
                if self._is_password_captcha_page(page):
                    logger.info("✅ CAPTCHA page detected")
                    return True
                if page.locator(".captcha-img:visible").count() > 0 and page.locator(".box-label:visible").count() > 0:
                    logger.info("✅ CAPTCHA elements detected")
                    return True
                if "captcha" in page.url.lower():
                    return True
            except Exception as exc:
                logger.debug(f"Wait loop error: {exc}")
            time.sleep(0.05)

        logger.error("❌ Timeout waiting for CAPTCHA page")
        save_screenshot(page, "captcha_timeout")
        return False

    def _wait_for_captcha_clear(self, page: Page, timeout_s: float = 8) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if page.is_closed():
                return False
            if not self._is_captcha_visible(page):
                return True
            time.sleep(0.05)
        return not self._is_captcha_visible(page)

    def _wait_for_post_captcha_result(self, page: Page) -> bool:
        deadline = time.time() + 1.8
        while time.time() < deadline:
            if self._is_go_home_page(page):
                return False
            if self._is_visa_type_form_visible(page):
                logger.info("✅ Visa Type page detected")
                return True
            if "visatype" in (page.url or "").lower():
                logger.info("✅ Visa Type URL detected")
                return True
            time.sleep(0.05)
        return self._is_visa_type_form_visible(page) and not self._is_go_home_page(page)

    def _wait_for_login_navigation(self, page: Page, timeout_s: float = 6) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if page.is_closed():
                return False
            try:
                page.wait_for_load_state("domcontentloaded", timeout=400)
            except Exception:
                pass
            if self._is_captcha_invalid_banner(page):
                return False
            if self._is_post_captcha_destination(page):
                return True
            if self._is_go_home_page(page):
                return False
            if not self._is_captcha_visible(page):
                url = (page.url or "").lower()
                if "login" not in url and "captcha" not in url:
                    return True
            time.sleep(0.03)
        return self._is_post_captcha_destination(page)

    def _ensure_post_captcha_destination(self, page: Page) -> bool:
        deadline = time.time() + 6

        while time.time() < deadline:
            if page.is_closed():
                return False
            if self._is_go_home_page(page) or self._is_captcha_invalid_banner(page):
                return False
            if self._is_post_captcha_destination(page):
                return True

            elapsed = time.time() - (deadline - 18)
            if elapsed > 10 and self._is_password_captcha_page(page) and self._is_captcha_visible(page) and not self._is_logged_in(page):
                self._submit_captcha_fallback(page)
                if self._wait_for_login_navigation(page, timeout_s=3.0) and self._is_post_captcha_destination(page):
                    return True
                if self._is_go_home_page(page) or self._is_captcha_invalid_banner(page):
                    return False

            if self._is_verification_step(page):
                self._handle_optional_verify_step(page)
                if self._wait_for_login_navigation(page, timeout_s=5.0) and self._is_post_captcha_destination(page):
                    return True
                time.sleep(0.3)
                continue

            if self._is_logged_in(page):
                try:
                    page.goto(
                        "https://appointment.thespainvisa.com/",
                        wait_until="domcontentloaded",
                        timeout=7000,
                    )
                except Exception:
                    pass
                if self._is_post_captcha_destination(page):
                    return True

            time.sleep(0.2)

        return False

    # ══════════════════════════════════════════════════════════════════════════
    # FORM INTERACTION HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_email_page(self, page: Page, email: str) -> bool:
        logger.info("📧 Processing email page…")
        save_screenshot(page, f"before_email_{email.replace('@','_')}")
        time.sleep(0.2)

        # Give the page a bit more time to fully render the email field.
        # On slower connections the form controls can appear slightly late,
        # which previously caused a false "Email field not found" error.
        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                # Prefer a real form control becoming visible over a blind wait.
                if (
                    page.locator('input[type="email"]:visible').count() > 0
                    or page.locator('input[name*="email" i]:visible').count() > 0
                    or page.locator('input[id*="email" i]:visible').count() > 0
                    or page.locator('input[placeholder*="email" i]:visible').count() > 0
                ):
                    break
            except Exception:
                pass
            try:
                page.wait_for_load_state("domcontentloaded", timeout=1000)
            except Exception:
                pass
            time.sleep(0.2)

        email_field = None

        # Method 1: FieldDetector
        result = self.field_detector.find_visible_email_field(page)
        if result and result[1]:
            email_field = result[1]

        # Method 2: type=email
        if not email_field:
            inputs = page.locator('input[type="email"]:visible').all()
            if inputs:
                email_field = inputs[0]

        # Method 3: heuristic selectors
        if not email_field:
            for sel in [
                "input[name*=\"email\" i]:visible",
                "input[id*=\"email\" i]:visible",
                "input[placeholder*=\"email\" i]:visible",
                'input.entry-disabled[type="text"]:visible',
            ]:
                items = page.locator(sel).all()
                if items:
                    email_field = items[0]
                    break

        if not email_field:
            logger.error("❌ Email field not found")
            return False

        try:
            email_field.scroll_into_view_if_needed()
            time.sleep(0.02)
            email_field.click()
            email_field.fill("")
            time.sleep(0.02)
            email_field.type(email, delay=random.uniform(5, 15))
            logger.info(f"✅ Email entered: {email}")
        except Exception as exc:
            logger.error(f"Failed to fill email: {exc}")
            return False

        time.sleep(0.05)

        if not self._submit_email_form(page):
            logger.error("❌ Submit button click failed")
            return False

        try:
            page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass

        # Poll for captcha page
        deadline = time.time() + 10
        while time.time() < deadline:
            if self._is_password_captcha_page(page):
                logger.info("✅ Password captcha page detected")
                return True
            time.sleep(0.5)

        # JS form-submit fallback
        try:
            page.evaluate(
                """
                () => {
                    const form = document.querySelector('form[action*="LoginSubmit"]') || document.querySelector('form');
                    if (form) form.submit();
                }
                """
            )
        except Exception:
            pass

        deadline = time.time() + 10
        while time.time() < deadline:
            if self._is_password_captcha_page(page):
                return True
            time.sleep(0.5)

        logger.error("❌ Did not reach password captcha page after email submit")
        return False

    def _submit_email_form(self, page: Page) -> bool:
        """Specific submit logic for the email page."""
        logger.info("🖱️ Attempting to click email submit button...")
        
        # 1. Try common button selectors first (Physical click)
        selectors = [
            "#btnVerify",
            'button:has-text("Continue")',
            'button:has-text("Next")',
            'button:has-text("Submit")',
            'input[type="submit"]',
            '.btn-primary',
            'button[type="submit"]'
        ]
        
        for selector in selectors:
            try:
                btn = page.locator(selector).first
                if btn.count() > 0 and btn.is_visible():
                    logger.info(f"👉 Clicking button: {selector}")
                    btn.scroll_into_view_if_needed()
                    time.sleep(0.1)
                    btn.click()
                    return True
            except Exception:
                pass
                
        # 2. Try JS click on those selectors
        for selector in selectors:
            try:
                if page.evaluate(f"document.querySelector('{selector}') !== null"):
                    page.evaluate(f"document.querySelector('{selector}').click()")
                    logger.info(f"👉 JS Clicked button: {selector}")
                    return True
            except Exception:
                pass

        logger.warning("⚠️ No visible submit button found for email form")
        return False

    def _handle_optional_verify_step(self, page: Page) -> bool:
        try:
            if page.is_closed():
                return False
            if self._is_password_captcha_page(page) or self._is_captcha_visible(page):
                return True
            url = (page.url or "").lower()
            if "verification" not in url:
                return True
            for selector in [
                'button:has-text("Verify")',
                'button:has-text("Continue")',
                'input[type="submit"][value*="Verify" i]',
                'input[type="submit"][value*="Continue" i]',
            ]:
                btn = page.locator(selector).first
                if btn.count() > 0 and btn.is_visible():
                    btn.scroll_into_view_if_needed()
                    time.sleep(0.05)
                    try:
                        btn.click()
                    except Exception:
                        page.evaluate(f"document.querySelector('{selector}').click()")
                    logger.info("✅ Verification submitted")
                    return True
            return True
        except Exception:
            return False

    def _click_submit_button(self, page: Page) -> bool:
        if page.is_closed():
            return False

        # Pre-fill password if empty
        try:
            if self._current_password:
                pwd = page.locator('input[type="password"]:visible').first
                if pwd.count() > 0:
                    try:
                        val = pwd.input_value()
                    except Exception:
                        val = ""
                    if not val:
                        self._fill_password_reliable(page, self._current_password)
        except Exception:
            pass

        # Prefer page-native JS submit handlers for reliability on obfuscated forms
        try:
            js_ok = page.evaluate(
                """
                () => {
                    if (typeof OnCaptchaSubmit === 'function') {
                        const r = OnCaptchaSubmit();
                        if (r === false) return false;
                        return true;
                    }
                    if (typeof onSubmit === 'function') {
                        const r = onSubmit();
                        if (r === false) return false;
                        return true;
                    }
                    if (typeof OnSubmitVerify === 'function') {
                        const r = OnSubmitVerify();
                        if (r === false) return false;
                        const btn = document.querySelector('#btnVerify');
                        if (btn) { btn.click(); return true; }
                        return true;
                    }
                    return null;
                }
                """
            )
            if js_ok is True:
                logger.info("✅ Submitted via page JS handler")
                return True
            if js_ok is False:
                return False
        except Exception:
            pass

        for selector in [
            "#btnVerify",
            'button[type="submit"]',
            'input[type="submit"]',
            ".btn-success",
            'button:has-text("Submit")',
            'button:has-text("Login")',
            'button:has-text("Continue")',
            'button:has-text("Sign In")',
            'button:has-text("Verificar")',
        ]:
            try:
                btn = page.locator(selector).first
                if btn.count() == 0 or not btn.is_visible():
                    continue
                btn.scroll_into_view_if_needed()
                time.sleep(0.1)
                try:
                    btn.click()
                except Exception:
                    page.evaluate(f"document.querySelector('{selector}').click()")
                logger.info(f"✅ Clicked submit: {selector}")
                return True
            except Exception:
                continue

        try:
            page.evaluate(
                """
                () => {
                    for (const b of document.querySelectorAll('button, input[type="submit"]')) {
                        const t = (b.innerText||b.value||'').toLowerCase();
                        if (t.includes('submit')||t.includes('continue')||t.includes('login')||
                            t.includes('verify')||b.type==='submit') { b.click(); return true; }
                    }
                    return false;
                }
                """
            )
            return True
        except Exception:
            pass

        logger.error("❌ Submit button not found")
        return False

    def _fill_password_reliable(self, page: Page, password: str) -> bool:
        if not password:
            return False
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
            except Exception:
                pass
            try:
                field.click(timeout=2000)
            except Exception:
                pass
            try:
                time.sleep(random.uniform(0.07, 0.18))
                field.press("Control+a")
                time.sleep(random.uniform(0.03, 0.09))
                field.press("Backspace")
            except Exception:
                try:
                    field.fill("")
                except Exception:
                    pass

            for ch in password:
                try:
                    field.type(ch, delay=random.randint(60, 140))
                except Exception:
                    break
                if random.random() < 0.25:
                    time.sleep(random.uniform(0.03, 0.12))

            try:
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
                        el.removeAttribute('readonly'); el.removeAttribute('disabled');
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

    def _apply_captcha_solution(self, page: Page, solution: Dict[str, Any], password: Optional[str]) -> bool:
        """Apply a pre-computed captcha solution dict (called from tests or external solvers)."""
        cells = solution["cells"]
        matching_indices = solution["matching_indices"]
        selection_ids = solution["selection_ids"]

        self._clear_captcha_selection(page)

        selection_ok = False
        for _ in range(getattr(Config, "CAPTCHA_SELECTION_RETRIES", 5)):
            if not self._select_captcha_cells(page, cells, matching_indices):
                continue
            time.sleep(getattr(Config, "CAPTCHA_AFTER_SELECTION_WAIT_MS", 200) / 1000.0)
            try:
                selected_indices = set(self._get_selected_indices(page, cells))
            except Exception:
                selected_indices = set()
            if selected_indices == set(matching_indices):
                selection_ok = True
                break
            self._clear_captcha_selection(page)
            time.sleep(getattr(Config, "CAPTCHA_CLICK_DELAY_MS", 140) / 1000.0)

        if not selection_ok:
            return False
        if password:
            self._fill_password_reliable(page, password)
            time.sleep(getattr(Config, "CAPTCHA_CLICK_DELAY_MS", 140) / 1000.0)
        if selection_ids:
            self._sync_selected_images(page, selection_ids)
        return self._submit_captcha_form(page, selection_ids=selection_ids)

    def _clear_captcha_selection(self, page: Page) -> bool:
        try:
            for selector in [
                'a:has-text("Clear Selection")',
                "a[onclick*=\"onUndo\"]",
                "a[onclick*=\"OnClearSelect\"]",
                "a[onclick*=\"onClearSelect\"]",
                "a[href*=\"OnClearSelect\"]",
            ]:
                link = page.locator(selector).first
                if link.count() > 0 and link.is_visible():
                    link.click()
                    time.sleep(0.02)
                    return True
            return bool(
                page.evaluate(
                    """
                    () => {
                        if (typeof OnClearSelect === 'function') { OnClearSelect(); return true; }
                        if (typeof onUndo       === 'function') { onUndo();        return true; }
                        return false;
                    }
                    """
                )
            )
        except Exception:
            return False

    def _select_captcha_cells(self, page: Page, cells: List[Any], indices: List[int]) -> bool:
        """Physical cell selection (used by _apply_captcha_solution)."""
        try:
            if not indices:
                return False
            unique = sorted(set(indices))

            def is_selected(cell) -> bool:
                try:
                    return page.evaluate(
                        """
                        (cell) => {
                            if (!cell) return false;
                            return cell.classList.contains('img-selected') ||
                                   (cell.style.border && cell.style.border.includes('5px solid green'));
                        }
                        """, cell
                    )
                except Exception:
                    return False

            def physical_click(cell) -> bool:
                try:
                    box = cell.bounding_box()
                    if not box:
                        return False
                    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                    return True
                except Exception:
                    return False

            for _ in range(max(3, getattr(Config, "CAPTCHA_SELECTION_RETRIES", 5))):
                missing = [i for i in unique if i < len(cells) and not is_selected(cells[i])]
                if not missing:
                    return True
                for idx in missing:
                    if idx >= len(cells):
                        continue
                    cell = cells[idx]
                    if is_selected(cell):
                        continue
                    try:
                        cell.scroll_into_view_if_needed()
                    except Exception:
                        pass
                    physical_click(cell)
                    time.sleep(getattr(Config, "CAPTCHA_CLICK_DELAY_MS", 300) / 1000.0)
                    if not is_selected(cell):
                        try:
                            cell.click(force=True, timeout=1000)
                        except Exception:
                            pass
                        time.sleep(getattr(Config, "CAPTCHA_CLICK_DELAY_MS", 300) / 1000.0)

            remaining = [i for i in unique if i < len(cells) and not is_selected(cells[i])]
            return len(remaining) == 0
        except Exception:
            return False

    def _get_selected_indices(self, page: Page, cells: List[Any]) -> List[int]:
        selected = []
        for idx, cell in enumerate(cells):
            try:
                if page.evaluate(
                    """
                    (cell) => {
                        if (!cell) return false;
                        const img = cell.matches && cell.matches('img.captcha-img')
                            ? cell : cell.querySelector('img.captcha-img');
                        return cell.classList.contains('img-selected')
                            || (img && img.classList.contains('img-selected'));
                    }
                    """, cell
                ):
                    selected.append(idx)
            except Exception:
                pass
        return selected

    def _get_selection_ids(self, cells: List[Any], indices: List[int]) -> List[str]:
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
                cid = cells[idx].get_attribute("id")
                if cid:
                    ids.append(cid)
            except Exception:
                continue
        return ids

    def _sync_selected_images(self, page: Page, selection_ids: List[str]) -> bool:
        if not selection_ids:
            return False
        try:
            return bool(
                page.evaluate(
                    """
                    (ids) => {
                        const input = document.querySelector('#SelectedImages');
                        if (input) input.value = ids.join(',');
                        if (Array.isArray(window.selection)) window.selection = ids.slice();
                        if (typeof window.setAction === 'function') window.setAction();
                        return true;
                    }
                    """, selection_ids
                )
            )
        except Exception:
            return False

    def _submit_captcha_form(self, page: Page, selection_ids: Optional[List[str]] = None) -> bool:
        try:
            if selection_ids:
                self._sync_selected_images(page, selection_ids)
            try:
                ok = page.evaluate(
                    """
                    () => {
                        if (typeof OnCaptchaSubmit === 'function') return OnCaptchaSubmit();
                        if (typeof onSubmit       === 'function') return onSubmit();
                        return null;
                    }
                    """
                )
                if ok is True:
                    return True
                if ok is False:
                    return False
            except Exception:
                pass
            for selector in ["#btnVerify", 'button[type="submit"]', 'button:has-text("Submit")', 'button:has-text("Verify")', ".btn-success"]:
                btn = page.locator(selector).first
                if btn.count() > 0 and btn.is_visible():
                    btn.scroll_into_view_if_needed()
                    time.sleep(0.05)
                    try:
                        btn.click(force=True)
                    except Exception:
                        try:
                            btn.click()
                        except Exception:
                            pass
                    return True
            return False
        except Exception:
            return False

    def _submit_captcha_fallback(self, page: Page) -> None:
        try:
            page.evaluate(
                """
                () => {
                    if (typeof OnCaptchaSubmit === 'function') { OnCaptchaSubmit(); return; }
                    if (typeof onSubmit       === 'function') { onSubmit();        return; }
                    const form = document.getElementById('captchaForm');
                    if (form) form.submit();
                }
                """
            )
        except Exception:
            pass
        try:
            page.wait_for_load_state("domcontentloaded", timeout=2500)
        except Exception:
            pass

    def _refresh_captcha(self, page: Page) -> bool:
        try:
            if page.is_closed():
                return False
            logger.info("🔄 Refreshing captcha…")
            for selector in [
                "a:has(i.fa-sync)", "a:has(i.fa-redo)",
                "a[onclick*=\"onReload\" i]", "a[onclick*=\"reload\" i]",
                "[title*=\"refresh\" i]", "[title*=\"reload\" i]",
                'a:has-text("Reload")', 'a:has-text("Refresh")',
                ".refresh-captcha", "img[src*=\"refresh\"]",
                'button:has-text("Refresh")', 'button:has-text("Reload")',
            ]:
                try:
                    btn = page.locator(selector).first
                    if btn.count() > 0 and btn.is_visible():
                        btn.scroll_into_view_if_needed()
                        time.sleep(0.2)
                        btn.click(force=True)
                        time.sleep(0.6)
                        if hasattr(self.captcha_solver, "_wait_for_captcha_images_ready"):
                            self.captcha_solver._wait_for_captcha_images_ready(page, timeout_ms=3000)
                        return True
                except Exception:
                    continue
            logger.warning("No refresh button — reloading page")
            current_url = page.url
            try:
                page.reload(wait_until="domcontentloaded", timeout=8000)
                time.sleep(0.8)
                if self._is_password_captcha_page(page):
                    return True
                page.goto(current_url, wait_until="domcontentloaded", timeout=6000)
                time.sleep(0.3)
                return True
            except Exception as exc:
                logger.error(f"Page reload failed: {exc}")
            return False
        except Exception as exc:
            logger.error(f"Refresh error: {exc}")
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # VISA TYPE & APPOINTMENT BOOKING
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_visa_type_page(self, page: Page, applicant: Applicant) -> bool:
        logger.info("🧾 Handling Visa Type page…")

        deadline = time.time() + 1.0
        while time.time() < deadline:
            if self._is_visa_type_form_visible(page):
                break
            time.sleep(0.02)

        if not self._is_visa_type_form_visible(page):
            logger.warning("Visa Type form not detected")
            return False

        location_value = (applicant.location or applicant.center or "").strip()
        explicit_type = (applicant.visa_type_option or "").strip()
        explicit_sub_type = (applicant.visa_sub_type_option or "").strip()
        visa_type_value, visa_sub_type_value = self._resolve_visa_type_values(applicant)
        if explicit_type:
            visa_type_value = explicit_type
        if explicit_sub_type:
            visa_sub_type_value = explicit_sub_type
        category_value = (applicant.category or "").strip() or "Normal"
        appointment_for_value = (applicant.appointment_for or "").strip() or "Individual"

        self._select_appointment_for(page, appointment_for_value)

        fields: List[Dict[str, str]] = []
        if location_value:
            fields.append({"label": "Location",     "value": location_value})
        if visa_type_value:
            fields.append({"label": "Visa Type",    "value": visa_type_value})
        if visa_sub_type_value:
            fields.append({"label": "Visa Sub Type","value": visa_sub_type_value})
        if category_value:
            fields.append({"label": "Category",     "value": category_value})
        if "family" in appointment_for_value.lower():
            members_value = (getattr(applicant, "members_count", "") or "").strip() or "2 Members"
            fields.append({"label": "Number Of Members", "value": members_value})

        selection_results = self._select_kendo_fields(page, fields) if fields else {}

        if location_value and not selection_results.get("Location"):
            logger.warning("Location selection failed")
            return False
        if visa_type_value and not selection_results.get("Visa Type"):
            logger.warning("Visa Type selection failed")
            return False

        if not self._submit_visa_type_form(page):
            logger.error("Visa Type form submit failed")
            return False

        if not self._wait_for_visa_type_navigation(page):
            return False
        return self._wait_for_appointment_page(page)

    def _wait_for_visa_type_navigation(self, page: Page) -> bool:
        start_url = page.url
        deadline = time.time() + 1.6
        while time.time() < deadline:
            url = (page.url or "").lower()
            if "login" in url or "account/login" in url:
                return False
            if "appointment/newappointment" in url:
                return True
            if page.url != start_url and not self._is_visa_type_form_visible(page):
                return True
            time.sleep(0.03)
        return "appointment/newappointment" in (page.url or "").lower()

    def _handle_liveness_and_data_protection(self, page: Page, email: str) -> bool:
        try:
            liveness_modal = page.locator("#alertModal").first
            if liveness_modal.count() > 0 and liveness_modal.is_visible():
                agree_btn = liveness_modal.locator('button:has-text("I agree")').first
                if agree_btn.count() > 0:
                    agree_btn.click()
                    time.sleep(0.3)
            dp_modal = page.locator("#dpModal").first
            if dp_modal.count() > 0 and dp_modal.is_visible():
                dp_btn = dp_modal.locator('button:has-text("I have read and understood")').first
                if dp_btn.count() > 0:
                    dp_btn.click()
                    time.sleep(0.4)
            if not self._wait_for_verification_email_sent(page):
                return False
            logger.info(f"📧 Live verification link sent to {email}")
            return self._wait_for_live_verification_complete(page)
        except Exception as exc:
            logger.error(f"Liveness error: {exc}")
            return False

    def _wait_for_verification_email_sent(self, page: Page) -> bool:
        deadline = time.time() + 2
        while time.time() < deadline:
            url = (page.url or "").lower()
            if "dataprotectionemailsent" in url:
                return True
            if page.locator('text="Data Protection"').first.count() > 0 and page.locator('text="Data Protection"').first.is_visible():
                return True
            time.sleep(0.05)
        return True

    def _wait_for_live_verification_complete(self, page: Page) -> bool:
        deadline = time.time() + 12
        while time.time() < deadline:
            url = (page.url or "").lower()
            if "rejection" in url:
                return False
            if "appointment" in url and "dataprotectionemailsent" not in url:
                if self._booking_form_present(page):
                    return True
            if self._booking_form_present(page):
                return True
            time.sleep(0.2)
        return False

    def _booking_form_present(self, page: Page) -> bool:
        try:
            if page.locator('form[action*="appointment"]').first.count() > 0 and page.locator('form[action*="appointment"]').first.is_visible():
                return True
            slot_inputs = page.locator(
                "input[id*=\"slot\" i], select[id*=\"slot\" i], "
                "input[id*=\"date\" i], select[id*=\"date\" i]"
            )
            for i in range(slot_inputs.count()):
                if slot_inputs.nth(i).is_visible():
                    return True
            return False
        except Exception:
            return False

    def _wait_for_booking_form_ready(self, page: Page) -> bool:
        deadline = time.time() + 4.5
        while time.time() < deadline:
            if self._booking_form_present(page):
                return True
            time.sleep(0.03)
        return False

    def _start_booking_process(self, page: Page) -> bool:
        try:
            for selector in [
                'button:has-text("Continue")',
                'button:has-text("Book")',
                'button:has-text("Submit")',
                "#btnSubmit",
                'button[type="submit"]',
            ]:
                btn = page.locator(selector).first
                if btn.count() > 0 and btn.is_visible():
                    btn.scroll_into_view_if_needed()
                    time.sleep(0.02)
                    btn.click()
                    return True
            return True
        except Exception:
            return False

    def _restart_new_appointment_flow(self, page: Page) -> bool:
        if self._is_go_home_page(page):
            self._navigate_back_to_home_after_captcha_failure(page)
        if not self._is_index_page(page) and not self._has_book_new_appointment_link(page):
            try:
                page.goto(
                    "https://appointment.thespainvisa.com/",
                    wait_until="domcontentloaded",
                    timeout=8000,
                )
            except Exception:
                return False
            if not self._is_index_page(page) and not self._has_book_new_appointment_link(page):
                return False
        return self._navigate_to_new_appointment(page)

    # ── Kendo dropdown helpers ───────────────────────────────────────────────

    def _select_kendo_fields(self, page: Page, fields: List[Dict[str, str]]) -> Dict[str, bool]:
        try:
            return page.evaluate(
                """
                (fields) => {
                    const norm = v => (v||'').toString().toLowerCase().replace(/[^a-z0-9]+/g,' ').replace(/\\s+/g,' ').trim();
                    const labels = Array.from(document.querySelectorAll('label.form-label'));
                    const byText = {};
                    for (const lbl of labels) {
                        const text = norm((lbl.textContent||'').replace('*',''));
                        if (!text || byText[text]) continue;
                        const s = window.getComputedStyle(lbl), r = lbl.getBoundingClientRect();
                        if (s.display==='none'||s.visibility==='hidden'||s.opacity==='0') continue;
                        if (r.width<5||r.height<5) continue;
                        const forId = lbl.getAttribute('for');
                        if (forId) byText[text] = forId;
                    }
                    const pick = (data, desired) => {
                        if (!data || !data.length) return null;
                        let best = null;
                        for (const item of data) {
                            const name = norm(item.Name||item.Text||'');
                            if (!name) continue;
                            if (name === desired) return item;
                            if (!best && (name.includes(desired) || desired.includes(name))) best = item;
                        }
                        return best;
                    };
                    const ensureData = (widget, fallback) => {
                        if (!widget || !widget.dataSource || !widget.dataSource.data) return [];
                        let data = widget.dataSource.data() || [];
                        if ((!data || !data.length) && fallback && fallback.length) {
                            try { widget.dataSource.data(fallback); } catch (e) {}
                            data = widget.dataSource.data() || [];
                        }
                        return data || [];
                    };
                    const out = {};
                    let lastLocationId = null;
                    let lastVisaTypeId = null;
                    for (const field of fields) {
                        const key = norm(field.label), desired = norm(field.value);
                        const fieldId = byText[key];
                        if (!fieldId || !desired) { out[field.label]=false; continue; }
                        const el = document.getElementById(fieldId);
                        if (!el) { out[field.label]=false; continue; }
                        const widget = window.jQuery ? window.jQuery(el).data('kendoDropDownList') : null;
                        if (!widget) { out[field.label]=false; continue; }
                        if (widget.dataSource && widget.dataSource.read) {
                            try { widget.dataSource.read(); } catch (e) {}
                        }
                        let fallback = null;
                        if (key === 'location' && Array.isArray(window.locationData)) fallback = window.locationData;
                        if (key === 'category' && Array.isArray(window.categoryData)) fallback = window.categoryData;
                        if (key === 'number of members' && Array.isArray(window.applicantsNoData)) fallback = window.applicantsNoData;
                        if (key === 'visa type' && Array.isArray(window.visaIdData)) {
                            fallback = lastLocationId ? window.visaIdData.filter(v => String(v.LegalEntityId) === String(lastLocationId)) : window.visaIdData;
                        }
                        if (key === 'visa sub type' && Array.isArray(window.visasubIdData)) {
                            fallback = lastVisaTypeId ? window.visasubIdData.filter(v => String(v.Value) === String(lastVisaTypeId)) : window.visasubIdData;
                        }
                        const data = ensureData(widget, fallback);
                        const match = pick(data, desired);
                        if (!match) { out[field.label]=false; continue; }
                        const value = match.Id ?? match.Value ?? match.value ?? match.id;
                        if (value===undefined || value===null || value==='') { out[field.label]=false; continue; }
                        if (String(widget.value()) !== String(value)) {
                            widget.value(value);
                            widget.trigger('change');
                        }
                        if (key === 'location') lastLocationId = value;
                        if (key === 'visa type') lastVisaTypeId = value;
                        out[field.label]=true;
                    }
                    return out;
                }
                """,
                fields,
            ) or {}
        except Exception:
            return {}

    def _select_appointment_for(self, page: Page, appointment_for: str) -> bool:
        target = "Family" if "family" in (appointment_for or "").lower() else "Individual"
        try:
            radios = page.locator(f'input[type="radio"][value="{target}"]')
            for i in range(radios.count()):
                radio = radios.nth(i)
                if radio.is_visible():
                    radio.click()
                    if target == "Family":
                        self._accept_family_disclaimer(page)
                    return True
            return False
        except Exception:
            return False

    def _accept_family_disclaimer(self, page: Page) -> bool:
        try:
            modal = page.locator("#familyDisclaimer").first
            if modal.count() > 0 and modal.is_visible():
                btn = modal.locator('button:has-text("Accept")').first
                if btn.count() > 0 and btn.is_visible():
                    btn.click()
                    return True
            return False
        except Exception:
            return False

    def _submit_visa_type_form(self, page: Page) -> bool:
        for selector in ["#btnSubmit", 'button#btnSubmit', 'button:has-text("Submit")', 'button[type="submit"]']:
            try:
                btn = page.locator(selector).first
                if btn.count() > 0 and btn.is_visible():
                    btn.scroll_into_view_if_needed()
                    time.sleep(0.05)
                    btn.click()
                    return True
            except Exception:
                continue
        return False

    def _resolve_visa_type_values(
        self, applicant: Applicant
    ) -> Tuple[Optional[str], Optional[str]]:
        raw = (applicant.visa_type or "").strip().lower()
        purpose = (applicant.purpose or "").strip().lower()
        combined = f"{raw} {purpose}".strip()

        if "tourist" in combined:
            return "Schengen Visa/ Short Term Visa", "Tourist Visa"
        if "business" in combined:
            return "Schengen Visa/ Short Term Visa", "Business Visa"
        if "conference" in combined:
            return "Schengen Visa/ Short Term Visa", "Conference Visa"
        if "medical" in combined:
            return "Schengen Visa/ Short Term Visa", "Medical  Visa"
        if "study" in combined or "student" in combined:
            return "National Visa/ Long Term Visa", "National Visas (Study, Work & Other National Visas)"
        if "family" in combined and "eu" in combined:
            return "Family of EEA/EU citizens", "Family of EEA/EU citizens"
        if "family" in combined:
            return "National Visa/ Long Term Visa", "Family Reunification Visa"
        if "national" in combined:
            return "National Visa/ Long Term Visa", "National Visa"
        if "schengen" in combined or "short" in combined:
            return "Schengen Visa/ Short Term Visa", None
        return applicant.visa_type, None

    # ── browser / context setup ──────────────────────────────────────────────

    def _launch_browser(self, playwright: Playwright) -> Browser:
        return playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--window-size=1920,1080",
                "--start-maximized",
                "--disable-gpu",
                "--disable-software-rasterizer",
            ],
        )

    def _create_context(self, browser: Browser) -> BrowserContext:
        context = browser.new_context(
            user_agent=random.choice(Config.USER_AGENTS),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="Europe/Madrid",
            geolocation={"latitude": 40.4168, "longitude": -3.7038},
            permissions=["geolocation"],
            color_scheme="light",
            ignore_https_errors=True,
            accept_downloads=True,
        )
        context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins',   { get: () => [1,2,3,4,5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
            """
        )
        return context

    def _attach_dialog_handler(self, page: Page) -> None:
        try:
            def handle(dialog):
                try:
                    dialog.dismiss()
                except Exception:
                    try:
                        dialog.accept()
                    except Exception:
                        pass
            page.on("dialog", handle)
        except Exception:
            pass

    # ── statistics / reporting ───────────────────────────────────────────────

    def _get_wait_time(self, attempt: int) -> float:
        return 0.1 + random.uniform(0.02, 0.08)

    def _print_progress(self) -> None:
        p = self.stats
        logger.info("-" * 50)
        logger.info(f"Progress: {p['processed']}/{p['total_applicants']}")
        logger.info(f"Successful: {p['successful']}  Failed: {p['failed']}")
        rate = p["successful"] / max(p["processed"], 1) * 100
        logger.info(f"Success Rate: {rate:.1f}%")
        logger.info("-" * 50)

    def _print_statistics(self) -> None:
        p = self.stats
        logger.info(f"\n{'='*60}")
        logger.info("📊 FINAL STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total: {p['total_applicants']}  Processed: {p['processed']}")
        logger.info(f"Successful: {p['successful']}  Failed: {p['failed']}")
        if p["processed"] > 0:
            logger.info(f"Success Rate: {p['successful']/p['processed']*100:.1f}%")
        stats = self.captcha_solver.get_stats()
        logger.info("\n🤖 Captcha Stats:")
        logger.info(f"  OCR Success: {stats.get('ocr_success', 0)}")
        logger.info(f"  Failed:      {stats.get('failed', 0)}")
        logger.info(f"  Total:       {stats.get('total', 0)}")
        logger.info(f"{'='*60}")

    def _save_final_results(self) -> None:
        save_result(
            {"stats": self.stats, "results": self.results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "final_results.json",
        )
        self._save_summary_csv()

    def _save_summary_csv(self) -> None:
        filename = f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = Path("results") / filename
        filepath.parent.mkdir(exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Email", "Name", "Passport", "Status", "Timestamp"])
            for result in self.results:
                appl = result["applicant"]
                writer.writerow([
                    appl["email"],
                    f"{appl.get('first_name','')} {appl.get('last_name','')}",
                    appl.get("passport_number", ""),
                    "SUCCESS" if result["success"] else "FAILED",
                    result["timestamp"],
                ])
        logger.info(f"📁 Summary saved: {filepath}")
