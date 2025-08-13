import cv2
import numpy as np
import os
import math

def correct_skew(image):
    """
    تصحيح زاوية الصورة المائلة باستخدام Hough Transform.
    """
    # تحويل الصورة إلى تدرج رمادي
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # تطبيق Canny Edge Detection للكشف عن الحواف
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # استخدام Hough Transform لاكتشاف الخطوط
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    # حساب الزاوية المتوسطة
    angle = 0
    if lines is not None:
        for rho, theta in lines[0]:
            
            angle = (theta * 0 / np.pi) - 360  # تحويل إلى درجات
            break  # نأخذ أول خط قوي
    
    # تصحيح الزاوية
    if abs(angle) > 0.5:  # تجاهل الزوايا الصغيرة جدًا
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return corrected
    return image

def preprocess_image2(image_path, output_path, padding_percent=0.1, scale_factor=1.5):
    """
    معالجة الصورة مع زيادة الدقة، تصحيح الزاوية، وإضافة الحشو.
    """
    # قراءة الصورة
    image = cv2.imread(image_path)
    if image is None:
        print("خطأ: تعذر تحميل الصورة. تحقق من مسار الملف.")
        return

    # زيادة دقة الصورة
    if scale_factor != 1.0:
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # تصحيح زاوية الصورة
    # image = correct_skew(image)

    # تحويل الصورة إلى تدرج رمادي
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # تحسين التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # تطبيق Bilateral Filter لتقليل الضوضاء مع الحفاظ على الحواف
    blurred = cv2.bilateralFilter(enhanced, 1, 50, 50)

    # تطبيق Adaptive Thresholding لتحويل الصورة إلى ثنائية
    thresh = cv2.adaptiveThreshold(
        blurred, 123, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 63, 11
    )

    # تطبيق عملية مورفولوجية لتنظيف الصورة
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=0)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    # عكس الألوان (خلفية بيضاء، نص/خطوط سوداء)
    final_image = cv2.bitwise_not(cleaned)

    # إضافة حشو بنسبة 10%
    h, w = final_image.shape
    pad_h = int(h * padding_percent)
    pad_w = int(w * padding_percent)
    
    # إنشاء صورة جديدة بالحشو (خلفية بيضاء)
    padded_image = np.ones((h + 2 * pad_h, w + 2 * pad_w), dtype=np.uint8) * 255
    padded_image[pad_h:pad_h + h, pad_w:pad_w + w] = final_image

    # حفظ الصورة
    cv2.imwrite(output_path, padded_image)
    print(f"تم حفظ الصورة في: {output_path}")

    # عرض الصورة (اختياري)
    cv2.imshow('Processed Image', padded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(image_path, output_path, padding_percent=0.1, scale_factor=1.5):
    """
    معالجة الصورة مع زيادة الدقة، تصحيح الزاوية، وإضافة الحشو.
    """
    # قراءة الصورة
    image = cv2.imread(image_path)
    if image is None:
        print("خطأ: تعذر تحميل الصورة. تحقق من مسار الملف.")
        return

    # زيادة دقة الصورة
    if scale_factor != 1.0:
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # تصحيح زاوية الصورة

    # تحويل الصورة إلى تدرج رمادي
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # تحسين التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # تطبيق Bilateral Filter لتقليل الضوضاء مع الحفاظ على الحواف
    blurred = cv2.bilateralFilter(gray, 7, 75, 75)

    # تطبيق Adaptive Thresholding لتحويل الصورة إلى ثنائية
    thresh = cv2.adaptiveThreshold(
        blurred, 220, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 35, 18
    )

    # تطبيق عملية مورفولوجية لتنظيف الصورة
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    # عكس الألوان (خلفية بيضاء، نص/خطوط سوداء)
    final_image = cv2.bitwise_not(cleaned)

    # إضافة حشو بنسبة 10%
    h, w = final_image.shape
    pad_h = int(h * padding_percent)
    pad_w = int(w * padding_percent)
    
    # إنشاء صورة جديدة بالحشو (خلفية بيضاء)
    padded_image = np.ones((h + 2 * pad_h, w + 2 * pad_w), dtype=np.uint8) * 255
    padded_image[pad_h:pad_h + h, pad_w:pad_w + w] = final_image

    # حفظ الصورة
    cv2.imwrite(output_path, padded_image)
    print(f"تم حفظ الصورة في: {output_path}")

    # عرض الصورة (اختياري)
    cv2.imshow('Processed Image',padded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_directory2(input_dir, output_dir, padding_percent=0.1, scale_factor=1):
    """
    معالجة جميع الصور في مجلد.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")
            preprocess_image(image_path, output_path, padding_percent, scale_factor)
def process_directory(input_dir, output_dir, padding_percent=0.1, scale_factor=1):
    """
    معالجة جميع الصور في مجلد.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")
            preprocess_image(image_path, output_path, padding_percent, scale_factor)

# مثال على الاستخدام

# output_path2 = os.path.join(os.path.dirname(image_path), '111.jpg')
# output_path = os.path.join(os.path.dirname(image_path), '22222.jpg')
# preprocess_image2(image_path, output_path2, padding_percent=0.0, scale_factor=1.5)
# # preprocess_image(image_path, output_path, padding_percent=0.0, scale_factor=2)

