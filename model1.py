import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import easyocr
from collections import Counter
import statistics
from ultralytics import YOLO

class YOLODigitExtractor:
    """
    Final, unified digit extractor.
    Combines a high-speed YOLOv8 cropping frontend with the advanced, high-quality
    image processing and candidate filtering pipeline for maximum accuracy.
    Includes a final outlier check to remove noise based on size variance.
    """
    def __init__(self, yolo_model_path='cels.pt', digit_model_path='digit_recognition_model.h5'):
        self.yolo_model_path = yolo_model_path
        self.digit_model_path = digit_model_path
        self.yolo_model = None
        self.digit_model = None
        self.ocr_reader = None
        
        self.load_yolo_model()
        self.load_digit_model()
        self.initialize_ocr()
        
        self.min_confidence = 0.4
        self.valid_grade_range = (0, 150)
        self.yolo_confidence_threshold = 0.9

    def load_yolo_model(self):
        """Loads the trained YOLO model."""
        if os.path.exists(self.yolo_model_path):
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
                print("✅ YOLO model loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading YOLO model: {e}")
        else:
            print(f"❌ YOLO model not found at '{self.yolo_model_path}'.")

    def load_digit_model(self):
        """Loads the trained TensorFlow digit recognition model."""
        if os.path.exists(self.digit_model_path):
            try:
                self.digit_model = load_model(self.digit_model_path)
                print("✅ Digit recognition model loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading digit model: {e}")
        else:
            print(f"❌ Digit model not found at '{self.digit_model_path}'.")

    def initialize_ocr(self):
        """Initializes the EasyOCR reader."""
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("✅ EasyOCR initialized successfully.")
        except Exception as e:
            print(f"❌ Error loading EasyOCR: {e}")

    def get_yolo_crop(self, image_array, debug=False):
        """Accepts an image ARRAY, runs YOLO, and returns a cropped image ARRAY."""
        if self.yolo_model is None:
            if debug: print("⚠️ YOLO model not loaded. Skipping crop.")
            return image_array
        if debug: print("🔍 Applying YOLOv8 filter...")

        results = self.yolo_model(image_array, conf=self.yolo_confidence_threshold, verbose=False)
        
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            if debug: print("  ⚠️ YOLOv8 found no number region. Using original image.")
            return image_array
            
        best_box = max(results[0].boxes, key=lambda box: box.conf[0])
        coords = best_box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        
        if debug: print(f"  ✅ YOLOv8 detected box with confidence {best_box.conf.item():.2f}")

        cropped_image = image_array[y1:y2, x1:x2]
        
        if cropped_image.size == 0:
            if debug: print("⚠️ Cropped image is empty, returning original.")
            return image_array
            
        return cropped_image

    def extract_digits_smart(self, image_array_gray, debug=False):
        """Main extraction pipeline with advanced image processing."""
        if image_array_gray is None: return "", []

        # --- Advanced Image Processing Pipeline (Restored) ---
        enhanced_image = cv2.bilateralFilter(image_array_gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(enhanced_image)
        _, best_binary = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        cleaned_binary = self.advanced_noise_removal(best_binary, debug)
        main_region = self.detect_main_digit_region(cleaned_binary, debug)
        
        contours, _ = cv2.findContours(main_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = [cv2.boundingRect(c) for c in contours if self.is_valid_digit_candidate(*cv2.boundingRect(c), cv2.contourArea(c), image_array_gray.shape)[0]]

        candidates_with_predictions = []
        for (x, y, w, h) in sorted(candidates, key=lambda c: c[0]):
            roi = main_region[y:y+h, x:x+w]
            processed_digit = self.preprocess_digit(roi)
            if processed_digit is not None and self.digit_model is not None:
                prediction = self.digit_model.predict(processed_digit, verbose=0)
                digit, confidence = np.argmax(prediction), np.max(prediction)
                if confidence >= self.min_confidence:
                    candidates_with_predictions.append(((x, y, w, h), digit, confidence))

        if debug: print(f"\nFound {len(candidates_with_predictions)} initial predictions.")
        
        # --- Final Filtering Stages ---
        candidates_with_predictions = self.filter_nested_candidates(candidates_with_predictions, debug)
        candidates_with_predictions = self.smart_grouping_detection(candidates_with_predictions, debug)
        candidates_with_predictions = self.filter_outlier_digits(candidates_with_predictions, debug) # NEW: Outlier filter
        candidates_with_predictions.sort(key=lambda c: c[0][0])
        
        result = "".join([str(d) for (b, d, c) in candidates_with_predictions])
        confidences = [c for (b, d, c) in candidates_with_predictions]
        areas = [b[2] * b[3] for (b, d, c) in candidates_with_predictions]
        
        validated_result, warnings = self.validate_grade_logic(result, confidences, areas, debug)
        
        if debug: print(f"📝 Custom Model Result: {validated_result}")
        return validated_result, warnings

    def extract_with_easyocr_alternative(self, image_array_bgr, debug=False):
        """Accepts a BGR image array and passes it directly to EasyOCR."""
        if self.ocr_reader is None: return "", []
        if debug: print("🔄 Using EasyOCR for extraction...")
        
        try:
            results = self.ocr_reader.readtext(image_array_bgr, detail=1, paragraph=False, allowlist='0123456789')
            if not results: return "", []
            
            best_result = max(results, key=lambda r: r[2])
            final_text = ''.join(filter(str.isdigit, best_result[1]))
            if debug: print(f"📝 EasyOCR Result: {final_text} (Confidence: {best_result[2]:.3f})")
            return final_text, []
        except Exception as e:
            print(f"❌ Error in EasyOCR: {e}")
            return "", []

    def extract_with_multiple_methods(self, image_path, debug=False):
        """Top-level method with smart failover logic."""
        print(f"🔍 Analyzing image: {image_path}")
        original_image_bgr = cv2.imread(image_path)
        if original_image_bgr is None:
            print(f"❌ Cannot read image: {image_path}")
            return "", []

        cropped_bgr_image = self.get_yolo_crop(original_image_bgr, debug=debug)
        cropped_gray_image = cv2.cvtColor(cropped_bgr_image, cv2.COLOR_BGR2GRAY)

        print("\n--- Attempting Custom Model ---")
        result1, warnings1 = self.extract_digits_smart(cropped_gray_image, debug)

        is_failed = not result1 or len(warnings1) > 0 or len(result1) <= 1
        
        if not is_failed:
            print("✅ Custom Model succeeded with high-quality result.")
            return result1, warnings1
        else:
            print("⚠️ Custom Model result is low-quality or failed. Engaging EasyOCR fallback.")
            print("\n--- Attempting EasyOCR Model ---")
            result2, warnings2 = self.extract_with_easyocr_alternative(cropped_bgr_image, debug)
            
            if not result2:
                print("⚠️ EasyOCR also failed. Returning original custom model result.")
                return result1, warnings1
            
            return result2, warnings2

    # --- FULL HELPER METHODS (RESTORED FROM SmartDigitExtractor) ---
    def advanced_noise_removal(self, binary_img, debug=False):
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_small)
        contours, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 5]
        if areas:
            avg_area = np.mean(areas)
            median_area = np.median(areas)
            noise_threshold = max(50, median_area * 0.1) if avg_area > 500 else max(20, median_area * 0.05)
            if debug: print(f"🧹 Noise removal threshold: {noise_threshold:.2f}")
            mask = np.zeros_like(cleaned)
            for contour in contours:
                if cv2.contourArea(contour) >= noise_threshold:
                    cv2.fillPoly(mask, [contour], 255)
            cleaned = cv2.bitwise_and(cleaned, mask)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(cleaned)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if not (aspect_ratio > 10 or aspect_ratio < 0.1) and cv2.contourArea(contour) > 30:
                cv2.fillPoly(final_mask, [contour], 255)
        return cv2.bitwise_and(cleaned, final_mask)

    def separate_connected_digits(self, binary_img, debug=False):
        contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        separated_img = binary_img.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 1000 and w > h * 0.8:
                roi = binary_img[y:y+h, x:x+w]
                horizontal_profile = np.sum(roi, axis=0)
                smoothed = np.convolve(horizontal_profile, np.ones(3)/3, mode='same')
                threshold = np.mean(smoothed) * 0.3
                valleys = [i for i in range(1, len(smoothed) - 1) if smoothed[i] < threshold and smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]]
                for valley_x in valleys:
                    if 5 < valley_x < w - 5:
                        cv2.line(separated_img, (x + valley_x, y), (x + valley_x, y + h), 0, 2)
        return separated_img

    def detect_main_digit_region(self, binary_img, debug=False):
        separated_img = self.separate_connected_digits(binary_img, debug)
        contours, _ = cv2.findContours(separated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return binary_img
        contour_data = [{'contour': c, 'area': cv2.contourArea(c), 'bbox': cv2.boundingRect(c), 'aspect_ratio': cv2.boundingRect(c)[2] / (cv2.boundingRect(c)[3] if cv2.boundingRect(c)[3] > 0 else 1)} for c in contours if cv2.contourArea(c) > 30]
        if not contour_data: return binary_img
        filtered_contours = [data for data in contour_data if not (data['aspect_ratio'] > 15 or data['aspect_ratio'] < 0.05)]
        if not filtered_contours: filtered_contours = [data for data in contour_data if data['area'] > 100]
        filtered_contours.sort(key=lambda x: x['bbox'][0])
        main_contours = filtered_contours if len(filtered_contours) <= 2 else sorted(filtered_contours, key=lambda x: x['area'], reverse=True)[:3]
        main_contours.sort(key=lambda x: x['bbox'][0])
        mask = np.zeros_like(separated_img)
        for data in main_contours:
            cv2.fillPoly(mask, [data['contour']], 255)
        if debug: print(f"✅ Selected {len(main_contours)} main contours")
        return cv2.bitwise_and(separated_img, mask)

    def is_valid_digit_candidate(self, x, y, w, h, area, image_shape):
        if area < 100 or w < 5 or h < 8: return False, "Too small"
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 15 or (h > 10 and aspect_ratio < 0.08): return False, "Line-like"
        return True, "Valid"

    def preprocess_digit(self, roi):
        if roi.size == 0: return None
        h, w = roi.shape
        size = max(h, w) + 6
        square = np.zeros((size, size), dtype=np.uint8)
        y_offset, x_offset = (size - h) // 2, (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = roi
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        return normalized.reshape(1, 28, 28, 1)

    def filter_nested_candidates(self, candidates, debug=False):
        if len(candidates) <= 1: return candidates
        if debug: print("\n🔍 Executing robust nested candidate filtering...")
        to_remove_indices = set()
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if i in to_remove_indices or j in to_remove_indices: continue
                box_i, box_j = candidates[i][0], candidates[j][0]
                area_i, area_j = box_i[2] * box_i[3], box_j[2] * box_j[3]
                i_contains_j = (box_i[0] <= box_j[0] and box_i[1] <= box_j[1] and (box_i[0] + box_i[2]) >= (box_j[0] + box_j[2]) and (box_i[1] + box_i[3]) >= (box_j[1] + box_j[3]))
                j_contains_i = (box_j[0] <= box_i[0] and box_j[1] <= box_i[1] and (box_j[0] + box_j[2]) >= (box_i[0] + box_i[2]) and (box_j[1] + box_j[3]) >= (box_i[1] + box_i[3]))
                if i_contains_j and area_i > area_j: to_remove_indices.add(i)
                elif j_contains_i and area_j > area_i: to_remove_indices.add(j)
        final_candidates = [cand for i, cand in enumerate(candidates) if i not in to_remove_indices]
        if debug: print(f"  ➡️ Nested filtering complete. {len(final_candidates)} candidates remain.")
        return final_candidates
    
    def filter_outlier_digits(self, candidates_with_predictions, debug=False):
        """
        Filters out outlier digits using a two-factor (height then area) statistical approach.
        """
        if len(candidates_with_predictions) < 2:
            return candidates_with_predictions

        if debug:
            print("\n🔍 Executing advanced outlier digit filtering...")

        # --- Stage 1: Filter by Height Consistency ---
        heights = [box[3] for (box, _, _) in candidates_with_predictions]
        median_height = np.median(heights)
        
        if median_height == 0: return candidates_with_predictions

        height_filtered_candidates = []
        for candidate in candidates_with_predictions:
            box = candidate[0]
            height = box[3]
            # Keep candidates whose height is reasonably close to the median height
            if 0.5 * median_height <= height <= 2.0 * median_height:
                height_filtered_candidates.append(candidate)
            else:
                if debug:
                    print(f"  🗑️ Discarding outlier by height: Digit '{candidate[1]}' with height {height} (Median: {median_height:.0f})")

        # If filtering by height removed everything, something is wrong, return original
        if not height_filtered_candidates:
            return candidates_with_predictions
            
        # --- Stage 2: Filter by Area Consistency on the remaining candidates ---
        if len(height_filtered_candidates) < 2:
            return height_filtered_candidates
            
        areas = [box[2] * box[3] for (box, _, _) in height_filtered_candidates]
        median_area = np.median(areas)

        if median_area == 0:
            return height_filtered_candidates

        final_candidates = []
        for candidate in height_filtered_candidates:
            box, digit, conf = candidate
            area = box[2] * box[3]
            ratio_to_median = area / median_area if median_area > 0 else 0

            # This ratio is more tolerant because major outliers were already removed by height.
            # It allows for digits like '1' (small area) to coexist with '8' (large area).
            if 0.15 <= ratio_to_median <= 4.0:
                final_candidates.append(candidate)
            else:
                if debug:
                    print(f"  🗑️ Discarding outlier by area: Digit '{digit}' with area {area} (Median Area: {median_area:.0f}, Ratio: {ratio_to_median:.2f})")
        
        return final_candidates

    def smart_grouping_detection(self, candidates, debug=False): return candidates
    def validate_grade_logic(self, result_str, confidences, areas, debug=False):
        if not result_str: return result_str, []
        warnings = []
        try:
            grade = int(result_str)
            if not (self.valid_grade_range[0] <= grade <= self.valid_grade_range[1]):
                warnings.append(f"Grade out of range: {grade}")
        except ValueError:
            warnings.append(f"Result is not an integer: {result_str}")
        return result_str, warnings

# Example usage
if __name__ == "__main__":
    # IMPORTANT: Update this path to an image that produces an outlier digit
    test_images = [r"D:\smarcard\sorc\asa\asa\py\DAGREE\output_pipeline\4_extracted_cells\111_row_18\cleaned_cells\grade_3_4_0_00.png"] 
    
    extractor = YOLODigitExtractor('cels.pt', 'digit_recognition_model.h5')
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            result, warnings = extractor.extract_with_multiple_methods(image_path, debug=True)
            print(f"✅ Final Result: {result}")
            if warnings:
                print("⚠️ Warnings:", warnings)
            print("="*60)
        else:
            print(f"❌ Image not found: {image_path}")
