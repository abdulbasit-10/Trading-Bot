"""
OCRService — Fast, accurate digit extraction using RapidOCR (PaddleOCR).

Key optimisations vs. original:
  • Fast path skips expensive colour-channel analysis unless in aggressive mode.
  • Ensemble stops early the moment a candidate exceeds the confidence threshold.
  • Target-aware early exit: if the caller supplies a target number, the loop
    exits the moment that target is confirmed with high confidence.
  • Variant count kept small (4 in fast mode, 6 in aggressive) to minimise
    redundant OCR calls while covering the most effective preprocessing paths.
"""

import importlib
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import Config
from utils.logger import get_logger

RapidOCREngine = Any
logger = get_logger(__name__)


class OCRService:
    """
    Digit extraction from captcha cell images.

    Public interface
    ----------------
    extract_number_with_confidence(image_bytes, aggressive, target_number)
        → (number_str | None, confidence_float)

    get_candidates_with_confidence(image_bytes, aggressive, target_number)
        → (scores_dict, best_number | None, best_confidence)

    get_all_candidates(image_bytes, aggressive, target_number)
        → scores_dict

    extract_number_from_image(image_bytes)
        → number_str | None   (backward-compat wrapper)

    extract_text_from_image(image_bytes)
        → str                 (full text, no digit filtering)
    """

    # ── construction ────────────────────────────────────────────────────────

    def __init__(self):
        self.engine: Optional[RapidOCREngine] = None
        self._initialize_engine()

    def _initialize_engine(self):
        try:
            logger.info("Initialising PaddleOCR (RapidOCR)…")
            module = importlib.import_module("rapidocr_onnxruntime")
            self.engine = module.RapidOCR()
            logger.info("PaddleOCR initialised")
        except Exception as exc:
            logger.error(f"Failed to initialise PaddleOCR: {exc}")
            self.engine = None

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def extract_number_with_confidence(
        self,
        image_bytes: bytes,
        aggressive: bool = False,
        target_number: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        try:
            scores, max_conf = self._run_ensemble(image_bytes, aggressive, target_number)
            best, score, conf = self._best_candidate(scores, max_conf)
            if best:
                logger.debug(f"OCR winner: {best} (score={score:.2f}, conf={conf:.2f})")
                return best, conf
            return None, 0.0
        except Exception as exc:
            logger.error(f"OCR extraction failed: {exc}")
            return None, 0.0

    def extract_number_from_image(self, image_bytes: bytes) -> Optional[str]:
        """Backward-compatible wrapper — returns only the number string."""
        number, _ = self.extract_number_with_confidence(image_bytes)
        return number

    def get_candidates_with_confidence(
        self,
        image_bytes: bytes,
        aggressive: bool = False,
        target_number: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Optional[str], float]:
        scores, max_conf = self._run_ensemble(image_bytes, aggressive, target_number)
        best, _, conf = self._best_candidate(scores, max_conf)
        return scores, best, conf

    def get_candidates_with_confidence_from_array(
        self,
        img: np.ndarray,
        aggressive: bool = False,
    ) -> Tuple[Dict[str, float], Optional[str], float]:
        scores: Dict[str, float] = {}
        max_conf: Dict[str, float] = {}
        variants = self._prepare_variants_from_array(img, aggressive=aggressive)
        if not variants:
            return scores, None, 0.0

        stop_conf = min(0.98, max(Config.OCR_CONFIDENCE_THRESHOLD + 0.15, 0.8))
        weights = [1.0] + [0.9] * max(0, len(variants) - 1)

        for idx, variant in enumerate(variants):
            weight = weights[idx] if idx < len(weights) else 0.8
            for text, conf in self._run_ocr(variant):
                self._update_candidate_scores(scores, max_conf, text, conf, weight)
            best, _, conf = self._best_candidate(scores, max_conf)
            if best and conf >= stop_conf:
                break

        best, _, conf = self._best_candidate(scores, max_conf)
        return scores, best, conf

    def get_all_candidates(
        self,
        image_bytes: bytes,
        aggressive: bool = False,
        target_number: Optional[str] = None,
    ) -> Dict[str, float]:
        scores, _ = self._run_ensemble(image_bytes, aggressive, target_number)
        return scores

    def extract_number(self, image_bytes: bytes) -> Optional[int]:
        img = self.preprocess_image(image_bytes)
        for text, _ in self._run_ocr(img):
            cleaned = "".join(filter(str.isdigit, text))
            if cleaned:
                return int(cleaned)
        return None

    def extract_all_numbers(self, image_bytes: bytes) -> List[int]:
        numbers: List[int] = []
        try:
            img = self.preprocess_image(image_bytes)
            for text, _ in self._run_ocr(img):
                for num_str in re.findall(r"\b\d+\b", text):
                    if num_str.isdigit():
                        numbers.append(int(num_str))
        except Exception as exc:
            logger.error(f"extract_all_numbers failed: {exc}")
        return numbers

    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """Extract all text without digit-specific filtering."""
        try:
            gray = self._decode_grayscale(image_bytes)
            if gray is None:
                return ""
            results = self._run_ocr(gray, use_det=True)
            if not results:
                results = self._run_ocr(gray, use_det=False)
            return " ".join(text for text, _ in results)
        except Exception as exc:
            logger.error(f"Text extraction failed: {exc}")
            return ""

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        variants = self._prepare_variants(image_bytes, aggressive=False)
        return variants[0] if variants else np.zeros((1, 1), dtype=np.uint8)

    # ══════════════════════════════════════════════════════════════════════════
    # CORE ENSEMBLE
    # ══════════════════════════════════════════════════════════════════════════

    def _run_ensemble(
        self,
        image_bytes: bytes,
        aggressive: bool = False,
        target_number: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Run multiple preprocessing variants through OCR and accumulate scores.

        Early-exit conditions
        ─────────────────────
        1. If target_number is supplied and found with confidence ≥ 0.85 → stop.
        2. If best candidate confidence ≥ stop_conf (0.72–0.98) → stop.
        3. In fast mode, stop as soon as confidence ≥ OCR_CONFIDENCE_THRESHOLD.
        """
        scores: Dict[str, float] = {}
        max_conf: Dict[str, float] = {}

        if not self.engine:
            return scores, max_conf

        variants = self._prepare_variants(image_bytes, aggressive=aggressive)
        if not variants:
            return scores, max_conf

        # Limit variants in fast mode
        if not aggressive:
            variants = variants[:4]

        stop_conf = min(0.98, max(Config.OCR_CONFIDENCE_THRESHOLD + 0.1, 0.72))
        target_stop_conf = 0.85
        weights = [1.0] + [0.9] * max(0, len(variants) - 1)
        digit_whitelist = "0123456789lI|iOQoDZSsz$BbAGgqT"

        try:
            for idx, variant in enumerate(variants):
                weight = weights[idx] if idx < len(weights) else 0.8
                ocr_results: List[Tuple[str, float]] = []

                # Light pass (no detection) for the first variant in fast mode
                if idx == 0 and not aggressive:
                    try:
                        res = self._run_ocr(variant, use_det=False, whitelist=digit_whitelist)
                        if any(
                            any(c.isdigit() for c in txt) and c_conf > 0.5
                            for txt, c_conf in res
                        ):
                            ocr_results.extend(res)
                    except Exception:
                        pass

                if not ocr_results:
                    ocr_results = self._run_ocr(variant, use_det=True, whitelist=digit_whitelist)

                # If detection found nothing useful, try without detection
                if not any(
                    len(self._extract_digit_candidates(txt, 3, 4)) > 0 and conf > 0.6
                    for txt, conf in ocr_results
                ):
                    try:
                        extra = self._run_ocr(variant, use_det=False, whitelist=digit_whitelist)
                        ocr_results.extend(extra)
                    except Exception:
                        pass

                # Accumulate per-block and merged
                full_digits = ""
                total_conf = 0.0
                count = 0

                for text, conf in ocr_results:
                    self._update_candidate_scores(scores, max_conf, text, conf, weight)
                    cleaned = self._fix_common_ocr_errors(text)
                    digits = "".join(filter(str.isdigit, cleaned))
                    if digits:
                        full_digits += digits
                        total_conf += conf
                        count += 1

                if count > 1 and len(full_digits) >= 3:
                    self._update_candidate_scores(
                        scores, max_conf, full_digits, total_conf / count, weight * 0.9
                    )

                # Early-exit checks
                if scores:
                    if target_number:
                        if max_conf.get(target_number, 0.0) >= target_stop_conf:
                            logger.debug(
                                f"Target {target_number} found early "
                                f"(conf={max_conf[target_number]:.2f})"
                            )
                            return scores, max_conf

                    best, _, best_c = self._best_candidate(scores, max_conf)
                    if best and best_c >= stop_conf:
                        break
                    if not aggressive and best_c >= Config.OCR_CONFIDENCE_THRESHOLD:
                        break

        except Exception as exc:
            logger.error(f"Ensemble failed: {exc}")

        return scores, max_conf

    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE PREPROCESSING
    # ══════════════════════════════════════════════════════════════════════════

    def _prepare_variants(
        self, image_bytes: bytes, aggressive: bool
    ) -> List[np.ndarray]:
        gray = self._decode_grayscale(image_bytes)
        if gray is None:
            return []

        # Fast path: skip colour analysis
        if not aggressive:
            return self._prepare_variants_from_array(gray, aggressive=False)

        # Aggressive path: pick the channel with the highest contrast
        color = self._decode_color(image_bytes)
        if color is None:
            return self._prepare_variants_from_array(gray, aggressive=True)

        maxc = np.max(color, axis=2)
        minc = np.min(color, axis=2)
        lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        candidates = sorted(
            [
                (float(np.std(gray)), gray),
                (float(np.std(maxc)), maxc),
                (float(np.std(minc)), minc),
                (float(np.std(lab[:, :, 0])), lab[:, :, 0]),
                (float(np.std(hsv[:, :, 2])), hsv[:, :, 2]),
            ],
            key=lambda x: x[0],
            reverse=True,
        )

        base = candidates[0][1]
        variants = self._prepare_variants_from_array(base, aggressive=True)
        for _, alt in candidates[1:3]:
            variants.extend(self._prepare_variants_from_array(alt, aggressive=False))
        return variants

    def _prepare_variants_from_array(
        self, img: np.ndarray, aggressive: bool
    ) -> List[np.ndarray]:
        original = img.copy() if img is not None else None

        img = self._crop_to_content(img)
        if img is None or img.size == 0:
            img = original if original is not None else np.zeros((10, 10), dtype=np.uint8)

        h, w = img.shape[:2]
        scale = 2.5 if max(h, w) < 60 else 1.6
        interp = cv2.INTER_CUBIC if aggressive else cv2.INTER_LINEAR
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)

        _, bin_img = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        norm = np.empty_like(resized)
        cv2.normalize(resized, norm, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(norm)
        bin_inv = cv2.bitwise_not(bin_img)

        if not aggressive:
            # Fast mode: 4 variants
            return [bin_img, bin_inv, clahe, resized]

        # Aggressive mode: 6 most effective variants
        denoise = cv2.fastNlMeansDenoising(clahe, None, 10, 7, 21)
        blur = cv2.bilateralFilter(denoise, 5, 40, 40)
        adapt = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return [bin_img, bin_inv, clahe, denoise, adapt, resized]

    def _crop_to_content(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return img

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(gray, 30, 150)
        coords = np.column_stack(np.where(edges > 0))

        if coords.size == 0:
            coords = np.column_stack(np.where(thresh < 250))
        if coords.size == 0:
            return img

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        pad = 8
        y_min = max(0, y_min - pad)
        y_max = min(h, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(w, x_max + pad)

        cropped_area = (x_max - x_min) * (y_max - y_min)
        total_area = w * h
        if cropped_area < total_area * 0.05:
            return img
        is_small = w < 120 and h < 120
        if is_small and cropped_area < total_area * 0.25:
            return img

        orig_aspect = w / h if h > 0 else 0
        cw, ch = x_max - x_min, y_max - y_min
        crop_aspect = cw / ch if ch > 0 else 0
        if crop_aspect > 4 * orig_aspect or (orig_aspect > 0 and crop_aspect < 0.25 * orig_aspect):
            return img

        return img[y_min:y_max, x_min:x_max]

    # ══════════════════════════════════════════════════════════════════════════
    # OCR ENGINE WRAPPER
    # ══════════════════════════════════════════════════════════════════════════

    def _run_ocr(
        self,
        img: np.ndarray,
        use_det: bool = True,
        whitelist: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        if self.engine is None:
            return []
        if img is None or img.size == 0:
            return []

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        def _parse(result) -> List[Tuple[str, float]]:
            if not result:
                return []
            out = []
            for item in result:
                text = str(item[1])
                conf = float(item[2])
                if whitelist:
                    text = "".join(c for c in text if c in whitelist)
                if text:
                    out.append((text, conf))
            return out

        try:
            result, _ = self.engine(img, use_det=use_det)
            return _parse(result)
        except Exception:
            pass

        # Fallback without use_det kwarg
        try:
            result, _ = self.engine(img) if use_det else (None, None)
            return _parse(result)
        except Exception:
            return []

    # ══════════════════════════════════════════════════════════════════════════
    # CANDIDATE SCORING
    # ══════════════════════════════════════════════════════════════════════════

    def _update_candidate_scores(
        self,
        scores: Dict[str, float],
        max_conf: Dict[str, float],
        text: str,
        confidence: float,
        pass_weight: float = 1.0,
    ) -> None:
        confidence = max(0.0, min(1.0, float(confidence)))
        cleaned_base = self._fix_common_ocr_errors(text)
        if not cleaned_base:
            return

        base_candidates = self._extract_digit_candidates(cleaned_base, 3, 4)
        to_process: List[Tuple[str, float]] = []

        for cand in base_candidates:
            w = 1.0 if cand == cleaned_base else 0.6
            to_process.append((cand, w))
            for var in self._generate_confusable_variants(cand):
                if var != cand:
                    to_process.append((var, w * 0.8))

        for candidate, quality_weight in to_process:
            score = confidence * pass_weight * quality_weight
            scores[candidate] = scores.get(candidate, 0.0) + score
            var_conf = confidence * (1.0 if candidate == cleaned_base else 0.9)
            max_conf[candidate] = max(max_conf.get(candidate, 0.0), var_conf)

    def _best_candidate(
        self,
        scores: Dict[str, float],
        max_conf: Dict[str, float],
    ) -> Tuple[Optional[str], float, float]:
        if not scores:
            return None, 0.0, 0.0
        candidate, score = max(scores.items(), key=lambda x: x[1])
        return candidate, score, max_conf.get(candidate, 0.0)

    # ══════════════════════════════════════════════════════════════════════════
    # TEXT PROCESSING HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    _OCR_REPLACEMENTS: Dict[str, str] = {
        "l": "1", "I": "1", "i": "1", "|": "1",
        "O": "0", "o": "0", "D": "0", "Q": "0",
        "S": "5", "s": "5", "$": "5",
        "Z": "2", "z": "2",
        "B": "8", "b": "8",
        "A": "4",
        "G": "6",
        "T": "7",
        "g": "9", "q": "9",
    }

    def _fix_common_ocr_errors(self, text: str) -> str:
        if not text:
            return ""
        return "".join(
            ch if ch.isdigit() else self._OCR_REPLACEMENTS.get(ch, "")
            for ch in text
        )

    def _extract_digit_candidates(
        self, text: str, min_len: int = 3, max_len: int = 4
    ) -> List[str]:
        cleaned = self._fix_common_ocr_errors(text)
        if not cleaned:
            return []
        length = len(cleaned)
        candidates = set()
        if min_len <= length <= max_len:
            candidates.add(cleaned)
        for chunk_len in range(min_len, max_len + 1):
            if length >= chunk_len:
                for i in range(length - chunk_len + 1):
                    candidates.add(cleaned[i : i + chunk_len])
        return list(candidates)

    _CONFUSIONS: Dict[str, List[str]] = {
        "0": ["6", "8", "9", "D", "O", "Q"],
        "1": ["7", "I", "l", "|"],
        "2": ["Z", "7"],
        "3": ["8", "9"],
        "4": ["A"],
        "5": ["6", "S"],
        "6": ["0", "5", "8", "G"],
        "7": ["1", "T", "Z"],
        "8": ["0", "3", "6", "B"],
        "9": ["0", "3", "6", "g", "q"],
    }

    def _generate_confusable_variants(self, text: str) -> List[str]:
        cleaned = "".join(filter(str.isdigit, text))
        if not cleaned:
            return []
        variants = {cleaned}
        for i, ch in enumerate(cleaned):
            for alt in self._CONFUSIONS.get(ch, []):
                v = list(cleaned)
                v[i] = alt
                variants.add("".join(v))
        return list(variants)

    # ── raw image decoding ───────────────────────────────────────────────────

    def _decode_grayscale(self, image_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    def _decode_color(self, image_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# Singleton
ocr_service = OCRService()
