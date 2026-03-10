import cv2
import numpy as np
from src.services.ocr_service import ocr_service

def test_ocr_service():
    print("Testing OCRService optimization...")
    
    # Create a dummy grayscale image (100x100)
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(img, "123", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    
    # Encode to bytes
    success, encoded_img = cv2.imencode('.png', img)
    image_bytes = encoded_img.tobytes()
    
    # Test aggressive=False (Fast Path)
    print("Testing aggressive=False...")
    variants_fast = ocr_service._prepare_variants(image_bytes, aggressive=False)
    print(f"Fast path variants: {len(variants_fast)}")
    assert len(variants_fast) > 0, "Fast path should return variants"
    
    # Test aggressive=True (Slow Path)
    print("Testing aggressive=True...")
    variants_slow = ocr_service._prepare_variants(image_bytes, aggressive=True)
    print(f"Slow path variants: {len(variants_slow)}")
    assert len(variants_slow) > len(variants_fast), "Slow path should return more variants"
    
    print("OCRService optimization verified successfully!")

if __name__ == "__main__":
    test_ocr_service()
