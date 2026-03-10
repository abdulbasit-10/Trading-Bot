#!/usr/bin/env python3
"""
Spain Visa Login Bot with OCR Captcha Solving
Main entry point for the application
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, TypedDict

# Windows-specific: Silence "Event loop is closed" errors on exit
if sys.platform == 'win32':
    from asyncio.proactor_events import _ProactorBasePipeTransport
    
    def _silence_proactor_pipe_del(self):
        try:
            self.close()
        except (RuntimeError, ValueError, OSError):
            pass
            
    setattr(_ProactorBasePipeTransport, "__del__", _silence_proactor_pipe_del)

    # Also silence BaseSubprocessTransport
    from asyncio.base_subprocess import BaseSubprocessTransport
    
    def _silence_subprocess_del(self):
        try:
            self.close()
        except (RuntimeError, ValueError, OSError):
            pass
            
    setattr(BaseSubprocessTransport, "__del__", _silence_subprocess_del)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.automator import SpainVisaAutomator
from src.models.applicant import Applicant
from utils.logger import get_logger
from utils.helpers import save_result
from playwright.sync_api import sync_playwright, Browser

logger = get_logger(__name__)


class _RunStats(TypedDict):
    total_applicants: int
    successful: int
    failed: int


class _RunResults(TypedDict):
    stats: _RunStats
    details: List[Dict[str, Any]]


def _book_with_retries(automator: SpainVisaAutomator, browser: Browser, applicant: Applicant) -> bool:
    for attempt in range(3):
        try:
            logger.info(f"🔁 Attempt {attempt + 1}/3 for {applicant.email}")
            return automator.book_appointment(applicant, browser=browser)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"❌ Attempt {attempt + 1} failed for {applicant.email}: {e}")
    return False


def run_sequential_bookings(automator: SpainVisaAutomator, applicants: List[Applicant]) -> _RunResults:
    # Process applicants sequentially
    results: _RunResults = {
        'stats': {'total_applicants': len(applicants), 'successful': 0, 'failed': 0},
        'details': []
    }
    
    with sync_playwright() as p:
        # Launch browser once for sequential processing
        browser = p.chromium.launch(
            headless=Config.HEADLESS,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--window-size=1920,1080',
                '--start-maximized'
            ]
        )

        def worker(applicant: Applicant) -> Dict[str, Any]:
            try:
                success = _book_with_retries(automator, browser, applicant)
                return {
                    'applicant': applicant.to_dict(),
                    'success': bool(success),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            except Exception as e:
                return {
                    'applicant': applicant.to_dict(),
                    'success': False,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'error': str(e)
                }

        try:
            for i, applicant in enumerate(applicants):
                logger.info(f"\nProcessing applicant {i+1}/{len(applicants)}: {applicant.email}")
                
                # Process single applicant
                result = worker(applicant)
                
                if result['success']:
                    results['stats']['successful'] += 1
                    logger.info(f"✅ Successfully processed {applicant.email}")
                else:
                    results['stats']['failed'] += 1
                    logger.error(f"❌ Failed to process {applicant.email}")
                    
                results['details'].append(result)
                
                # Small delay between applicants
                if i < len(applicants) - 1:
                    time.sleep(2)
        finally:
            try:
                browser.close()
            except Exception:
                pass

    return results


def main():
    """Main execution function"""
    
    print("\n" + "!"*70)
    print("Spain Visa Login Bot with OCR Captcha Solving")
    print("!"*70 + "\n")
    
    try:
        # Validate configuration
        Config.validate()
        
        # Load applicants from CSV
        print(f"\nLoading applicants from {Config.APPLICANTS_FILE}...")
        applicants_data = Config.load_applicants()
        
        if not applicants_data:
            print("❌ No applicants found in CSV file")
            return 1
        
        print(f"Loaded {len(applicants_data)} applicants")
        
        # Convert to Applicant objects
        applicants = [Applicant.from_dict(data) for data in applicants_data]
        
        # Display first applicant (for verification)
        if applicants:
            first = applicants[0]
            print("\nFirst applicant:")
            print(f"   Email: {first.email}")
            print(f"   Name: {first.get_full_name()}")
            print(f"   Passport: {first.passport_number}")
            print(f"   Visa Type: {first.visa_type}")
            print(f"   Travel Date: {first.travel_date}")
        
        print(f"\nReady to process {len(applicants)} applicants")
        
        # Initialize automator
        automator = SpainVisaAutomator(
            headless=Config.HEADLESS,
            max_retries=Config.MAX_RETRIES,
            gemini_api_key=Config.GEMINI_API_KEY
        )
        
        # Process all applicants sequentially (single browser)
        start_time = time.time()
        results = run_sequential_bookings(automator, applicants)
        end_time = time.time()
        
        # Print final results
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n{'='*60}")
        print("✅ BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total Time: {hours}h {minutes}m {seconds}s")
        print(f"Total Applicants: {results['stats']['total_applicants']}")
        print(f"Successful: {results['stats']['successful']}")
        print(f"Failed: {results['stats']['failed']}")
        print(f"Success Rate: {(results['stats']['successful']/max(results['stats']['total_applicants'],1)*100):.1f}%")
        
        # Save final results
        result_file = save_result(results, "batch_results")
        print(f"\nDetailed results saved to: {result_file}")
        print(f"{'='*60}")
        
        return 0 if results['stats']['failed'] == 0 else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user")
        return 130
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("\nPlease make sure applicants.csv is in the current directory")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
