from pathlib import Path
import time
from playwright.sync_api import Page

from src.models.applicant import Applicant
from src.services.email_service import EmailService
from utils.logger import get_logger

logger = get_logger(__name__)


def handle_appointment_booking(page: Page, applicant: Applicant) -> bool:
    """
    Main appointment booking handler.
    Orchestrates the flow: slots page -> second slots page -> live verification.
    """

    if not _wait_for_appointment_page(page):
        logger.warning("Appointment page not detected")
        return False

    # Step 1: Handle slots selection page (slots.html)
    if not _handle_slots_page(page):
        logger.error("Failed to handle slots page")
        return False

    # Step 2: Handle second slots page (second_slots.html) - photo, OTP, date
    if not _handle_second_slots_page(page, applicant):
        logger.error("Failed to handle second slots page")
        return False

    # Step 3: Handle live verification page (live_verification.html)
    if not _handle_live_verification_page(page, applicant):
        logger.error("Failed to handle live verification page")
        return False

    logger.info("Appointment booking flow completed successfully")
    return True


def wait_for_appointment_page(page: Page) -> bool:
    """
    Public helper so the main automator can confirm that we really are on the
    appointment/slots page after the visa-type step.
    """
    return _wait_for_appointment_page(page)


# ----------------------------------------------------------
# WAIT FOR APPOINTMENT PAGE
# ----------------------------------------------------------

def _wait_for_appointment_page(page: Page) -> bool:
    """Wait for the appointment page to load by checking URL or heading."""
    deadline = time.time() + 10

    while time.time() < deadline:
        url = (page.url or "").lower()
        if "appointment/newappointment" in url:
            return True

        # Check for the page heading
        try:
            heading = page.locator('h5:has-text("Book New Appointment")').first
            if heading.is_visible():
                return True
        except:
            pass

        time.sleep(0.2)

    return False


# ----------------------------------------------------------
# SLOT DETECTION
# ----------------------------------------------------------

def _slots_available(page: Page) -> bool:
    """
    Detects whether slots are available based on page HTML.
    Checks for 'no slots' alert or visible appointment slot dropdowns.
    """
    try:
        # Check for "no slots available" alert
        alert = page.locator(".alert-danger").first
        if alert.count() > 0 and alert.is_visible():
            alert_text = alert.inner_text().lower()
            if "no slots" in alert_text or "not available" in alert_text:
                logger.info("No slots available alert detected")
                return False

        # Check for any visible date picker (Kendo datepicker)
        date_pickers = page.locator('input[data-role="datepicker"]').all()
        visible_pickers = [p for p in date_pickers if p.is_visible()]

        if visible_pickers:
            logger.info(f"Found {len(visible_pickers)} visible date picker(s)")
            return True

        return False
    except Exception as e:
        logger.error(f"Error checking slot availability: {e}")
        return False


# ----------------------------------------------------------
# CLICK TRY AGAIN
# ----------------------------------------------------------

def _click_try_again(page: Page) -> bool:
    """Click the Try Again button if slots are not available."""
    try:
        retry_btn = page.locator('a:has-text("Try Again")').first
        if retry_btn.count() == 0:
            return False

        if retry_btn.is_visible():
            logger.info("Clicking Try Again")
            with page.expect_navigation(wait_until="domcontentloaded"):
                retry_btn.click()
            page.wait_for_load_state("networkidle")
            return True

        return False
    except Exception as e:
        logger.error(f"Try Again click error: {e}")
        return False


# ----------------------------------------------------------
# SLOTS PAGE HANDLER (slots.html)
# ----------------------------------------------------------

def _handle_slots_page(page: Page) -> bool:
    """
    Handle the slot selection page (slots.html).
    Selects first available appointment date and slot from visible fields.
    """
    try:
        logger.info("Handling slots page - looking for available dates and slots")

        # Wait for the page to fully load
        page.wait_for_load_state("networkidle")
        time.sleep(1)  # Give JavaScript time to initialize

        # Find all visible date pickers
        date_pickers = page.locator('input[data-role="datepicker"]').all()
        visible_pickers = []

        for picker in date_pickers:
            try:
                if picker.is_visible():
                    picker_id = picker.get_attribute("id")
                    if picker_id:
                        visible_pickers.append(picker_id)
            except:
                continue

        if not visible_pickers:
            logger.error("No visible date pickers found")
            return False

        logger.info(f"Found visible date pickers: {visible_pickers}")

        # Use the first visible date picker
        target_date_id = visible_pickers[0]
        logger.info(f"Using date picker: {target_date_id}")

        # Click the calendar icon to open the date picker
        calendar_icon = page.locator(f'#{target_date_id}').locator('..').locator('span.k-select').first
        if calendar_icon.count() == 0 or not calendar_icon.is_visible():
            # Try alternative selector
            calendar_icon = page.locator(f'[aria-controls="{target_date_id}_dateview"]').first

        if calendar_icon.count() == 0 or not calendar_icon.is_visible():
            logger.error(f"Could not find calendar icon for {target_date_id}")
            return False

        logger.info("Opening calendar popup")
        calendar_icon.click()

        # Wait for calendar popup
        page.wait_for_selector(f"#{target_date_id}_dateview", timeout=10000)
        time.sleep(0.5)

        # Find and click the first available (green/success) date
        date_clicked = page.evaluate(f"""
            () => {{
                const calendar = document.querySelector('#{target_date_id}_dateview');
                if (!calendar) return {{success: false, error: 'Calendar not found'}};

                const links = Array.from(calendar.querySelectorAll('td:not(.k-state-disabled) a'));

                for (const link of links) {{
                    const style = window.getComputedStyle(link);
                    const bg = style.backgroundColor || '';

                    // Check for green background (available dates: rgb(25, 135, 84))
                    if (bg.includes('25, 135, 84') || bg.includes('success')) {{
                        link.click();
                        return {{success: true, date: link.getAttribute('data-value') || link.textContent}};
                    }}
                }}

                // Fallback: click first non-disabled date
                const firstAvailable = calendar.querySelector('td:not(.k-state-disabled) a');
                if (firstAvailable) {{
                    firstAvailable.click();
                    return {{success: true, date: firstAvailable.getAttribute('data-value') || firstAvailable.textContent, fallback: true}};
                }}

                return {{success: false, error: 'No available dates found'}};
            }}
        """)

        if not date_clicked or not date_clicked.get('success'):
            logger.warning(f"Could not select date: {date_clicked}")
            return False

        logger.info(f"Selected date: {date_clicked}")

        # Wait for date picker to close and slots to load
        time.sleep(1.5)

        # Find the corresponding slot dropdown (any visible dropdown that is not the date picker)
        slot_dropdown = None

        # Try to find slot dropdown by looking at parent container
        try:
            parent = page.locator(f'#{target_date_id}').locator('xpath=../../..')
            slot_in_same_container = parent.locator('input[data-role="dropdownlist"]').first
            if slot_in_same_container.count() > 0 and slot_in_same_container.is_visible():
                slot_dropdown = slot_in_same_container
        except Exception:
            pass

        # If not found, look for any visible slot dropdown
        if not slot_dropdown:
            all_slots = page.locator('input[data-role="dropdownlist"]').all()
            for slot in all_slots:
                try:
                    if slot.is_visible() and slot.get_attribute("id") != target_date_id:
                        slot_dropdown = slot
                        break
                except Exception:
                    continue

        if not slot_dropdown:
            logger.error("No visible slot dropdown found")
            return False

        slot_id = slot_dropdown.get_attribute("id")
        logger.info(f"Using slot dropdown: {slot_id}")

        # Open slot dropdown by clicking the input or the dropdown arrow
        slot_dropdown.click()
        time.sleep(0.8)  # Allow listbox to render and options to load

        # Try Kendo API first (if jQuery and widget exist)
        slot_selected = page.evaluate(f"""
            () => {{
                const slotInput = document.getElementById('{slot_id}');
                if (!slotInput) return {{success: false, error: 'Slot input not found'}};

                const $ = window.jQuery || window.$;
                if ($) {{
                    const dropdown = $(slotInput).data('kendoDropDownList');
                    if (dropdown && dropdown.dataSource) {{
                        const data = dropdown.dataSource.data();
                        if (data && data.length > 0) {{
                            // Prefer first item with Count > 0; else take first item
                            const item = data.find(function(i) {{ return i && (i.Count === undefined || i.Count > 0); }}) || data[0];
                            const value = item.Id !== undefined ? item.Id : (item.Value !== undefined ? item.Value : item.value);
                            if (value !== undefined && value !== null && value !== '') {{
                                dropdown.value(value);
                                dropdown.trigger('change');
                                return {{success: true, slot: item.Name || item.Text || item.name, id: value}};
                            }}
                        }}
                    }}
                }}
                return {{success: false, error: 'Kendo API not available or no data'}};
            }}
        """)

        # Fallback: select first slot by clicking the first listbox option
        if not slot_selected or not slot_selected.get('success'):
            logger.info("Kendo slot selection failed, trying listbox click fallback")
            listbox_id = f"{slot_id}_listbox"
            try:
                # Wait for listbox to appear (Kendo creates id_listbox when dropdown opens)
                page.wait_for_selector(f"#{listbox_id}", timeout=5000)
                time.sleep(0.3)
                # Kendo listbox: ul#id_listbox with li.k-item (skip header/disabled)
                for selector in [
                    f"#{listbox_id} li.k-item:not(.k-state-disabled)",
                    f"#{listbox_id} li.k-item",
                    f"#{listbox_id} [role='option']",
                    f"#{listbox_id} li",
                ]:
                    first_option = page.locator(selector).first
                    if first_option.count() > 0 and first_option.is_visible():
                        first_option.click()
                        slot_selected = {"success": True, "slot": "first option (click)"}
                        break
            except Exception as e:
                logger.warning(f"Listbox click fallback failed: {e}")

        if not slot_selected or not slot_selected.get('success'):
            logger.warning(f"Could not select slot: {slot_selected}")
            return False

        logger.info(f"Selected slot: {slot_selected}")
        time.sleep(0.3)  # Brief pause after selection before submit

        # Click Submit button
        submit = page.locator("#btnSubmit").first
        if not submit.is_visible():
            logger.error("Submit button not visible")
            return False

        logger.info("Clicking Submit to proceed to next page")
        with page.expect_navigation(wait_until="domcontentloaded", timeout=30000):
            submit.click()

        # Wait for next page to load
        page.wait_for_load_state("networkidle")
        time.sleep(1)

        return True

    except Exception as e:
        logger.error(f"Slots page error: {e}")
        return False


# ----------------------------------------------------------
# SECOND SLOT PAGE HANDLER (second_slots.html)
# ----------------------------------------------------------

def _handle_second_slots_page(page: Page, applicant: Applicant) -> bool:
    """
    Handle the applicant selection page (second_slots.html).
    Uploads photo, enters OTP, sets travel date, selects applicant.
    """
    try:
        logger.info("Handling second slots page (applicant selection)")

        # Wait for page to load
        page.wait_for_load_state("networkidle")
        time.sleep(1)

        # Handle Terms of Service modal if present
        try:
            terms_modal = page.locator("#termsmodal").first
            if terms_modal.count() > 0 and terms_modal.is_visible():
                logger.info("Accepting terms of service")
                agree_btn = terms_modal.locator('button:has-text("I agree")').first
                if agree_btn.count() > 0:
                    agree_btn.click()
                    time.sleep(0.5)
        except:
            pass

        # 1. Upload photo
        photo_path = "assets/photo.jpg"
        if Path(photo_path).exists():
            logger.info("Uploading photo")
            file_input = page.locator("#uploadfile-1").first
            if file_input.count() > 0:
                file_input.set_input_files(photo_path)
                time.sleep(1)  # Wait for upload

                # Handle photo confirmation modal if it appears
                try:
                    photo_modal = page.locator("#photoUploadModal").first
                    if photo_modal.count() > 0 and photo_modal.is_visible():
                        logger.info("Confirming photo upload")
                        understood_btn = photo_modal.locator('button:has-text("Understood")').first
                        if understood_btn.count() > 0:
                            understood_btn.click()
                            time.sleep(0.5)
                except:
                    pass
        else:
            logger.warning(f"Photo not found at {photo_path}")

        # 2. Fetch and enter OTP
        logger.info("Fetching OTP from email")
        email_service = EmailService(applicant.email, applicant.password)
        otp = email_service.fetch_otp()

        if not otp:
            logger.error("OTP not received")
            return False

        logger.info(f"OTP received: {otp}")
        page.fill("#EmailCode", otp)

        # 3. Set travel date
        travel_date = applicant.travel_date or "2026-03-20"
        logger.info(f"Setting travel date: {travel_date}")

        # Use JavaScript to set the Kendo date picker
        page.evaluate(f"""
            () => {{
                const datePicker = $('#TravelDate').data('kendoDatePicker');
                if (datePicker) {{
                    datePicker.value('{travel_date}');
                    datePicker.trigger('change');
                    return true;
                }}
                return false;
            }}
        """)

        # Also fill the input directly as fallback
        page.fill("#TravelDate", travel_date)

        # 4. Select applicant (radio button)
        logger.info("Selecting applicant")
        radio = page.locator(".rdo-applicant").first
        if radio.count() > 0 and radio.is_visible():
            radio.click()
            time.sleep(0.5)

        # 5. Click Submit
        logger.info("Submitting applicant selection")
        submit = page.locator("#btnSubmit").first
        if not submit.is_visible():
            logger.error("Submit button not visible on second slots page")
            return False

        with page.expect_navigation(wait_until="domcontentloaded", timeout=30000):
            submit.click()

        page.wait_for_load_state("networkidle")
        time.sleep(1)

        return True

    except Exception as e:
        logger.error(f"Second slots page error: {e}")
        return False


# ----------------------------------------------------------
# LIVE VERIFICATION PAGE HANDLER (live_verification.html)
# ----------------------------------------------------------

def _handle_live_verification_page(page: Page, applicant: Applicant) -> bool:
    """
    Handle the live verification page (live_verification.html).
    Sends email with verification link and cookies for manual completion.
    """
    try:
        logger.info("Handling live verification page")

        # Wait for page to load
        page.wait_for_load_state("networkidle")
        time.sleep(1)

        # Get current URL and cookies
        link = page.url
        cookies = page.context.cookies()
        cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])

        logger.info(f"Live verification URL: {link}")

        # Send email with verification details
        email_service = EmailService(applicant.email, applicant.password)
        subject = "Live Verification Required - Complete Your Appointment"
        body = f"""Dear Applicant,

Your appointment booking requires live verification. Please complete the following steps:

1. Open this link in your browser:
{link}

2. Use the following session cookies if needed:
{cookie_str}

3. Complete the liveness detection process by clicking "Accept" on the page.

Note: This link is time-sensitive. Please complete the verification as soon as possible.

Best regards,
Appointment Booking System
"""

        if email_service.send_email(applicant.email, subject, body):
            logger.info("Verification email sent successfully")
            return True
        else:
            logger.error("Failed to send verification email")
            return False

    except Exception as e:
        logger.error(f"Verification page error: {e}")
        return False