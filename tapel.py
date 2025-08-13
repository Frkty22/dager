import cv2
import os
import numpy as np
from ultralytics import YOLO

class SmartRowExtractor:
    """
    كلاس نهائي يقوم باكتشاف الصفوف وقصها بذكاء عبر الخطوات التالية:
    1. يختبر عدة أحجام وإعدادات لاختيار أفضل نتيجة تنبؤ من النموذج.
    2. يستخدم نتيجة التنبؤ الأفضل لقص الصفوف بناءً على مربعات 'name' المكتشفة.
    3. يضيف هامشًا رأسيًا (padding) لكل صف مقصوص.
    """
    def __init__(self, model_path):
        """
        تهيئة الكلاس مع مسار نموذج YOLO.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ملف النموذج غير موجود في المسار: {model_path}")
        self.model = YOLO(model_path)
        print("✅ تم تهيئة النموذج بنجاح.")

    def _find_best_prediction(self, image_path, sizes_to_test, conf, iou):
        """
        وظيفة داخلية خاصة لاختبار أحجام متعددة واختيار أفضل نتيجة بناءً على نظام التقييم المتقدم.
        """
        best_result_obj = None
        best_score = -1
        best_size = 0
        
        print(f"\n🔍 جاري اختبار الأحجام {sizes_to_test} لاختيار الأفضل...")
        
        for imgsz in sizes_to_test:
            try:
                results = self.model.predict(source=image_path, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
                result = results[0]
                boxes = result.boxes
                if len(boxes) == 0:
                    print(f"  - حجم {imgsz}px: لم يتم اكتشاف أي كائنات.")
                    continue
                
                all_boxes = [{'label': self.model.names[int(b.cls)], 'center_y': (b.xyxy[0][1] + b.xyxy[0][3]) / 2} for b in boxes]
                num_detections = len(all_boxes)
                avg_confidence = np.mean([b.conf.item() for b in boxes])
                
                name_boxes = [b for b in all_boxes if b['label'] == 'name']
                grade_boxes = [b for b in all_boxes if b['label'] != 'name']
                
                balance_score = 0
                if name_boxes and grade_boxes:
                    balance_ratio = min(len(name_boxes), len(grade_boxes)) / max(len(name_boxes), len(grade_boxes), 1)
                    balance_score = balance_ratio * 20
                
                y_positions = [b['center_y'] for b in all_boxes]
                y_std = np.std(y_positions) if len(y_positions) > 1 else 0
                distribution_score = 10 / (1 + y_std * 0.05)
                
                score = (min(num_detections, 100) + avg_confidence * 30 + balance_score + distribution_score)
                
                print(f"  - حجم {imgsz}px: الكشوفات={num_detections}, الثقة={avg_confidence:.2f}, النقاط={score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_result_obj = result
                    best_size = imgsz
            except Exception as e:
                print(f"  - ⚠️ حدث خطأ عند اختبار حجم {imgsz}: {e}")
                continue
        
        if best_result_obj:
            print(f"\n🏆 تم اختيار الحجم {best_size}px كأفضل حجم (النقاط: {best_score:.2f}).")
        else:
            print("\n❌ لم تنجح عملية الاكتشاف لأي حجم.")
            
        return best_result_obj

    def crop_and_save(self, image_path, output_folder, sizes_to_test=[640, 960, 1280], conf=0.7, iou=0.4, padding=5):
        """
        الوظيفة الرئيسية التي تقوم بتشغيل العملية الكاملة: البحث عن أفضل تنبؤ ثم القص والحفظ.
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"❌ خطأ: لم يتم العثور على الصورة في {image_path}")
            return
        
        height, width, _ = original_image.shape

        # الخطوة 1: البحث عن أفضل نتيجة تنبؤ
        best_prediction = self._find_best_prediction(image_path, sizes_to_test, conf, iou)
        
        if best_prediction is None:
            print("لم يتم العثور على اكتشافات جيدة، لا يمكن المتابعة للقص.")
            return

        # الخطوة 2: استخلاص مربعات 'name' من النتيجة الأفضل
        name_boxes = []
        for box in best_prediction.boxes:
            if self.model.names[int(box.cls)] == 'name':
                name_boxes.append(box.xyxy[0].tolist())
        
        if not name_boxes:
            print("❌ لم يتم العثور على أي خلايا 'name' في أفضل نتيجة، لا يمكن قص الصفوف.")
            return
            
        # فرز المربعات من الأعلى للأسفل لضمان ترتيب الصفوف
        name_boxes.sort(key=lambda b: b[1])
        
        print(f"\n✂️ تم العثور على {len(name_boxes)} اسم للقص، سيتم إضافة هامش {padding}px لكل صف...")
        
        os.makedirs(output_folder, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # الخطوة 3: قص الصفوف بناءً على مربعات الأسماء مع إضافة الهامش
        for i, box in enumerate(name_boxes):
            y1 = int(box[1])
            y2 = int(box[3])
            
            # تحديد حدود القص مع إضافة الهامش والتأكد من عدم تجاوز أبعاد الصورة
            y_start = max(0, y1 - padding)
            y_end = min(height, y2 + padding)
            
            # قص صورة الصف
            row_image = original_image[y_start:y_end, :]
            
            # حفظ الصف المقصوص
            filename = f"{base_filename}_row_{i+1:02d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, row_image)
            
        print(f"\n🎉 تمت العملية بنجاح! تم حفظ {len(name_boxes)} صفًا في المجلد: '{output_folder}'")


# --- طريقة الاستخدام ---
if __name__ == "__main__":
    MODEL_PATH = r'C:\Users\Owner\Documents\test\mm\models\best2.pt'        # <--- ضع هنا مسار النموذج
    INPUT_IMAGE_PATH = '111.jpg'     # <--- ضع هنا مسار الصورة التي تريد معالجتها
    OUTPUT_FOLDER = 'final_rows'     # <--- اسم المجلد الذي سيتم حفظ الصفوف فيه

    # إنشاء الكائن من الكلاس
    extractor = SmartRowExtractor(model_path=MODEL_PATH)
    
    # تشغيل عملية الاستخراج والقص الكاملة
    # يمكنك تعديل الإعدادات هنا إذا أردت
    extractor.crop_and_save(
        image_path=INPUT_IMAGE_PATH,
        output_folder=OUTPUT_FOLDER,
        sizes_to_test=[640, 960], # قائمة الأحجام للاختبار
        conf=0.7,                        # عتبة الثقة التي أعطت أفضل النتائج
        iou=0.4,                         # عتبة التداخل
        padding=25                    # الهامش بالبكسل من الأعلى والأسفل
    )