import cv2
import numpy as np
import os

class ManualTableExtractor:
    def __init__(self, output_dir: str):
        """
        النسخة المُحسَّنة التي تقبل مجلد الإخراج مباشرة.
        """
        self.output_dir = output_dir
        self.points = []
        self.image_display = None
        self.original_image = None
        self.display_scale = 1.0
        self.display_window_name = "Select 4 Corners (c=confirm, r=reset, q=quit)"
        os.makedirs(self.output_dir, exist_ok=True)

    def _resize_for_display(self, img):
        h, w = img.shape[:2]
        if h == 0 or w == 0: return img, 1.0
        # تحديد أبعاد العرض القصوى
        max_display_width, max_display_height = 1200, 800
        scale = min(max_display_width / w, max_display_height / h, 1.0)
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([int(x / self.display_scale), int(y / self.display_scale)])
            # رسم دائرة لتوضيح النقطة المحددة
            cv2.circle(self.image_display, (x, y), 7, (0, 0, 255), -1)
            cv2.imshow(self.display_window_name, self.image_display)

    def execute(self, image_path: str):
        """
        الدالة الرئيسية لتنفيذ عملية استخراج الجدول.
        """
        self.points = []
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        self.image_display, self.display_scale = self._resize_for_display(self.original_image)
        
        cv2.namedWindow(self.display_window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.display_window_name, self.image_display)
        cv2.setMouseCallback(self.display_window_name, self._mouse_callback)

        print("Please click on the 4 corners of the table.")
        print("Press 'c' to confirm, 'r' to reset, 'q' to quit.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None # إرجاع None إذا ألغى المستخدم العملية
            if key == ord('r'):
                self.points = []
                self.image_display, _ = self._resize_for_display(self.original_image)
                cv2.imshow(self.display_window_name, self.image_display)
            if key == ord('c') and len(self.points) == 4:
                break
        
        cv2.destroyAllWindows()

        # --- ترتيب النقاط وتصحيح المنظور ---
        pts = np.array(self.points, dtype="float32")

        # ترتيب النقاط: أعلى-يسار، أعلى-يمين، أسفل-يمين، أسفل-يسار
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # حساب مصفوفة التحويل وتطبيقها
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.original_image, M, (maxWidth, maxHeight))
        
        # إرجاع الصورة المصححة مباشرة
        return warped