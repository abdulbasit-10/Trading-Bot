import re
import time
import json
import base64
from pathlib import Path
from playwright.sync_api import Page
from typing import Optional, Any

# Create directories
screenshot_dir = Path("screenshots")
screenshot_dir.mkdir(exist_ok=True)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


def save_screenshot(page: Page, name: str) -> str:
    """Save a screenshot with timestamp"""
    try:
        if page.is_closed():
            return ""
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.png"
        filepath = screenshot_dir / filename
        page.screenshot(path=str(filepath))
        print(f"📸 Screenshot saved: {filepath}")
        return str(filepath)
    except Exception:
        return ""


def save_base64_image(base64_data: str, filename: str) -> str:
    """Save base64 image to file"""
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        image_data = base64.b64decode(base64_data)
        filepath = screenshot_dir / filename
        with open(filepath, 'wb') as f:
            f.write(image_data)
        return str(filepath)
    except Exception as e:
        print(f"Error saving base64 image: {e}")
        return ""


def save_result(data: Any, filename: str) -> str:
    """Save result data to JSON file"""
    timestamp = int(time.time())
    if filename.endswith('.json'):
        base_name = filename[:-5]
        final_filename = f"{base_name}_{timestamp}.json"
    else:
        final_filename = f"{filename}_{timestamp}.json"
    
    filepath = results_dir / final_filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Results saved: {filepath}")
    return str(filepath)


def extract_number_from_classes(class_string: str) -> Optional[str]:
    """Extract number from class names"""
    # Look for 3-digit numbers first
    match = re.search(r'(\d{3})', class_string)
    if match:
        return match.group(1)
    
    # Fallback to any digits
    match = re.search(r'(\d+)', class_string)
    if match:
        return match.group(1)
    
    return None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def chunks(lst, n):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_csv_file(filepath: str) -> list:
    """Read CSV file and return list of dictionaries"""
    import csv
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
