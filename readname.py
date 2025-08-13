import easyocr
import cv2
import numpy as np

def enhance_text_for_ocr(image_path, save_path=None, show=True):
    """
    تحسين الصورة للتعرف الضوئي على الحروف مع إضافة إزالة التشويش.
    """
    # 1. قراءة الصورة وتكبيرها
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. تحويل إلى رمادية
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. إزالة التشويش البسيط (Noise Removal)
    denoised = cv2.medianBlur(gray, 3)

    # 4. تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)
    
    # 5. زيادة حدة الصورة
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)

    if save_path:
        cv2.imwrite(save_path, sharpened)
    if show:
        cv2.imshow("Enhanced Image for OCR", sharpened)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return sharpened

# الكود التجريبي فقط عند التشغيل المباشر
if __name__ == "__main__":
    test_images = [r"..\13_enhanced_rows\extracted_cells\13_row_12\grade_1_08.jpg", r"..\13_enhanced_rows\extracted_cells\13_row_12\grade_2_49.jpg"]
    image_file = r'..\model\smart_detection_results\16_row_10\cropped_cells\name_0_93.png' # ضع مسار صورتك هنا
    processed_image_path = "enhanced_image.jpg"
    enhanced_image = enhance_text_for_ocr(
        image_file,
        save_path=processed_image_path,
        show=True # غيرها إلى True إذا أردت رؤية الصورة
    )
    print("بدء التعرف على النص...")
    import easyocr
    reader = easyocr.Reader(['ar'])
    result = reader.readtext(processed_image_path)
    # allowlist='0123456789'  # <-- هذا السطر يجعل القراءة للأرقام الإنجليزية فقط',

