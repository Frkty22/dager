import cv2
import numpy as np
import os
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import warnings

# تجاهل التحذيرات غير الضرورية
warnings.filterwarnings("ignore")

class AdvancedTableExtractor:
    """
    يقوم هذا الكلاس المتقدم باكتشاف وقص الخلايا من صور صفوف الجداول باستخدام نموذج YOLO.
    يتميز بالخصائص التالية:
    - منطق ديناميكي يعتمد على موقع "الاسم" كنقطة مرجعية (anchor).
    - حساب ذكي لمواقع الخلايا المفقودة (التوقيع والدرجات).
    - ترتيب صحيح للدرجات حسب موقعها الأصلي في الجدول.
    - تنظيف ذكي للخلايا المقتصة لإزالة خطوط الجدول الأفقية.
    """

    def __init__(self, model_path: str):
        """
        تهيئة الكلاس وتحميل نموذج YOLO.
        """
        self.model_path = model_path
        try:
            self.model = YOLO(model_path)
            print("✅ تم تحميل نموذج YOLO بنجاح.")
        except Exception as e:
            raise Exception(f"❌ فشل تحميل نموذج YOLO: {e}")

        # إعدادات الاكتشاف
        self.detection_confidence = 0.9
        self.iou_threshold = 0.4
        # التصنيفات المطلوبة فقط
        self.target_labels = ['dagrea1', 'dagrea2', 'dagrea3', 'dagrea4', 'name']

    def clean_cell_from_lines(self, cell_image: np.ndarray, threshold_ratio: float = 0.7) -> np.ndarray:
        """
        تنظيف صورة الخلية من الخطوط الأفقية في الأعلى والأسفل.
        """
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        
        # البحث عن الحد العلوي الجديد
        y1_new = 0
        for y in range(h):
            row = binary[y, :]
            black_pixel_ratio = np.count_nonzero(row) / w
            if black_pixel_ratio < threshold_ratio:
                y1_new = y
                break
        
        # البحث عن الحد السفلي الجديد
        y2_new = h
        for y in range(h - 1, -1, -1):
            row = binary[y, :]
            black_pixel_ratio = np.count_nonzero(row) / w
            if black_pixel_ratio < threshold_ratio:
                y2_new = y + 1
                break
                
        # قص الصورة بالإحداثيات الجديدة النظيفة
        if y2_new > y1_new:
            return cell_image[y1_new:y2_new, :]
        else:
            return cell_image

    def calculate_cell_positions(self, name_box: Dict, detected_grade_boxes: List[Dict]) -> Dict:
        """
        حساب مواقع جميع الخلايا (المكتشفة والمفقودة) بناءً على موقع الاسم والدرجات المكتشفة.
        
        Args:
            name_box: معلومات مربع الاسم
            detected_grade_boxes: قائمة بالدرجات المكتشفة
            
        Returns:
            قاموس يحتوي على مواقع جميع الخلايا مرتبة
        """
        name_x1, name_y1, name_x2, name_y2 = name_box['coordinates']
        name_width = name_x2 - name_x1
        name_height = name_y2 - name_y1
        
        # ترتيب الدرجات المكتشفة من اليمين إلى اليسار
        detected_grade_boxes.sort(key=lambda x: x['center_x'], reverse=True)
        
        # حساب متوسط عرض وارتفاع مربعات الدرجات
        if detected_grade_boxes:
            grade_widths = []
            grade_heights = []
            for grade_box in detected_grade_boxes:
                gx1, gy1, gx2, gy2 = grade_box['coordinates']
                grade_widths.append(gx2 - gx1)
                grade_heights.append(gy2 - gy1)
            
            avg_grade_width = int(np.mean(grade_widths))
            avg_grade_height = int(np.mean(grade_heights))
        else:
            # في حالة عدم وجود درجات مكتشفة، نقدر الحجم
            avg_grade_width = name_width // 4  # تقدير أن عرض الدرجة ربع عرض الاسم
            avg_grade_height = name_height
        
        # حساب المواقع النظرية لجميع الخلايا (5 خلايا: توقيع + 4 درجات)
        theoretical_positions = {}
        
        # 1. خلية التوقيع - بين الاسم والدرجة الأولى
        signature_x2 = name_x1
        signature_x1 = signature_x2 - avg_grade_width
        theoretical_positions['signature'] = {
            'coordinates': (signature_x1, name_y1, signature_x2, name_y2),
            'confidence': 0.0,  # مستنتجة
            'cell_type': 'signature'
        }
        
        # 2. الدرجات الأربع
        current_x = signature_x1
        for i in range(4):
            grade_x2 = current_x
            grade_x1 = grade_x2 - avg_grade_width
            theoretical_positions[f'grade_{i+1}'] = {
                'coordinates': (grade_x1, name_y1, grade_x2, name_y2),
                'confidence': 0.0,  # سيتم تحديثها إذا كانت مكتشفة
                'cell_type': f'grade_{i+1}'
            }
            current_x = grade_x1
        
        # 3. ربط الدرجات المكتشفة بمواقعها النظرية
        for detected_box in detected_grade_boxes:
            detected_center_x = detected_box['center_x']
            best_match = None
            min_distance = float('inf')
            
            # البحث عن أقرب موقع نظري
            for grade_key in ['grade_1', 'grade_2', 'grade_3', 'grade_4']:
                theoretical_center_x = (theoretical_positions[grade_key]['coordinates'][0] + 
                                      theoretical_positions[grade_key]['coordinates'][2]) // 2
                distance = abs(detected_center_x - theoretical_center_x)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = grade_key
            
            # تحديث الموقع النظري بالموقع المكتشف
            if best_match:
                theoretical_positions[best_match]['coordinates'] = detected_box['coordinates']
                theoretical_positions[best_match]['confidence'] = detected_box['confidence']
        
        return theoretical_positions

    def _crop_and_save_cell(self, original_image: np.ndarray, box_info: Dict, crops_dir: str, 
                           filename_prefix: str, index: int):
        """
        قص خلية واحدة، تنظيفها، وحفظها.
        """
        x1, y1, x2, y2 = box_info['coordinates']
        
        # التحقق من صحة الإحداثيات
        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
            # التحقق من أن الإحداثيات لا تتجاوز حدود الصورة
            h, w = original_image.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            # قص الخلية
            cropped_cell = original_image[y1:y2, x1:x2]
            
            # تنظيف الخلية من الخطوط
            cleaned_cell = self.clean_cell_from_lines(cropped_cell)
            
            # حفظ الخلية النظيفة
            confidence_str = f"{box_info['confidence']:.2f}".replace('.', '_')
            crop_filename = f"{filename_prefix}_{index}_{confidence_str}.png"
            crop_save_path = os.path.join(crops_dir, crop_filename)
            
            cv2.imwrite(crop_save_path, cleaned_cell)
            
            # طباعة معلومات الخلية
            cell_status = "مكتشفة" if box_info['confidence'] > 0 else "مستنتجة"
            print(f"   💾 تم حفظ {filename_prefix} ({cell_status}): {crop_filename}")
        else:
            print(f"⚠️ تم تجاهل '{filename_prefix}' لأن أبعادها غير صالحة")

    def process_image(self, image_path: str, output_dir: str):
        """
        معالجة صورة واحدة: استخراج الدرجات الأربع والاسم مباشرة من نموذج YOLO المدرب.
        """
        if not os.path.exists(image_path):
            print(f"❌ لم يتم العثور على الصورة: {image_path}")
            return

        print(f"\n--- 🚀 بدء المعالجة المتقدمة للصورة: {os.path.basename(image_path)} ---")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        results_dir = os.path.join(output_dir, image_name)
        os.makedirs(results_dir, exist_ok=True)

        # تنفيذ نموذج YOLO
        try:
            results = self.model.predict(
                source=image_path,
                conf=self.detection_confidence,
                iou=self.iou_threshold,
                verbose=False
            )
        except Exception as e:
            print(f"❌ حدث خطأ أثناء التنبؤ: {e}")
            return

        result = results[0]
        if len(result.boxes) == 0:
            print("⚠️ لم يتم اكتشاف أي خلايا في الصورة.")
            return

        # استخراج فقط الخلايا المطلوبة (الدرجات الأربع والاسم)
        all_boxes = []
        for i, box in enumerate(result.boxes):
            label = self.model.names[int(box.cls[0])]
            if label not in self.target_labels:
                continue
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
            all_boxes.append({
                'index': i,
                'label': label,
                'confidence': float(box.conf[0]),
                'coordinates': (x1, y1, x2, y2),
                'center_x': (x1 + x2) // 2
            })

        # ترتيب الخلايا: الاسم ثم الدرجات حسب الترتيب
        crops_dir = os.path.join(results_dir, "cleaned_cells")
        os.makedirs(crops_dir, exist_ok=True)
        original_image = result.orig_img

        # حفظ خلية الاسم
        name_boxes = [b for b in all_boxes if b['label'] == 'name']
        if name_boxes:
            name_box = max(name_boxes, key=lambda b: b['confidence'])
            self._crop_and_save_cell(original_image, name_box, crops_dir, "name", 1)
        else:
            print("⚠️ لم يتم العثور على خانة الاسم.")

        # حفظ الدرجات الأربع حسب الترتيب (تغيير الاسم إلى grade_1, grade_2, ...)
        for idx, grade_label in enumerate(['dagrea1', 'dagrea2', 'dagrea3', 'dagrea4'], start=1):
            grade_boxes = [b for b in all_boxes if b['label'] == grade_label]
            if grade_boxes:
                grade_box = max(grade_boxes, key=lambda b: b['confidence'])
                # حفظ باسم grade_1, grade_2, ...
                self._crop_and_save_cell(original_image, grade_box, crops_dir, f"grade_{idx}", idx)
            else:
                print(f"⚠️ لم يتم العثور على خانة {grade_label}.")

        print(f"✅ انتهت المعالجة. تم حفظ {len(os.listdir(crops_dir))} خلية نظيفة في: {crops_dir}")

        # حفظ الصورة الأصلية مع المربعات للتحقق
        annotated_image = result.plot()
        annotated_image_path = os.path.join(results_dir, f"{image_name}_annotated.png")
        cv2.imwrite(annotated_image_path, annotated_image)

    def process_batch(self, input_folder: str, output_folder: str):
        """
        معالجة جميع الصور في مجلد الإدخال.
        """
        if not os.path.isdir(input_folder):
            print(f"❌ مجلد الإدخال '{input_folder}' غير موجود.")
            return
            
        os.makedirs(output_folder, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"❌ لم يتم العثور على أي صور في المجلد: {input_folder}")
            return
            
        print(f"\n" + "="*60)
        print(f"🔍 تم العثور على {len(image_files)} صورة للمعالجة في '{input_folder}'.")
        print(f"📂 سيتم حفظ النتائج في '{output_folder}'.")
        print("="*60)

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            self.process_image(image_path, output_folder)
            
        print("\n" + "="*60)
        print("🎉 تمت معالجة جميع الصور بنجاح!")
        print("="*60)


def main():
    """
    الدالة الرئيسية لتشغيل الكود.
    """
    # --- إعدادات المستخدم ---
    model_path = r'lasst.pt'  # ❗️ مسار نموذج YOLO
    input_folder = r'output_pipeline\3_extracted_rows'                       # 📂 مجلد الصور المدخلة
    output_folder = 'advanced_extraction_results'     # 🎯 مجلد حفظ النتائج

    try:
        extractor = AdvancedTableExtractor(model_path=model_path)
        extractor.process_batch(input_folder, output_folder)

    except Exception as e:
        print(f"❌ حدث خطأ فادح في البرنامج: {e}")

if __name__ == "__main__":
    main()