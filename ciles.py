import cv2
import numpy as np
import os

class ManualTableExtractor:

    def __init__(self, image_path, max_display_width=1200, max_display_height=800): # إضافة أبعاد عرض قصوى
        self.image_path = image_path
        self.points = []
        self.original_image = None # لتخزين الصورة الأصلية
        self.image_display = None # لتخزين الصورة المعروضة (قد تكون مصغرة)
        self.display_scale = 1.0 # لتخزين معامل التصغير/التكبير
        self.display_window_name = "Select 4 Corners (Press 'c' to confirm, 'r' to reset, 'q' to quit)"
        # --- أبعاد العرض القصوى ---
        # يمكنك تعديل هذه القيم لتناسب شاشتك
        self.max_display_width = max_display_width
        self.max_display_height = max_display_height
        # ---
        self._ensure_process_dir_exists()

    def _ensure_process_dir_exists(self):
        self.process_dir = "./process_images/manual_table_extractor/"
        if not os.path.exists(self.process_dir):
            os.makedirs(self.process_dir)
            print(f"Created directory: {self.process_dir}")

    def _resize_for_display(self, image):
        """يغير حجم الصورة لتناسب أبعاد العرض القصوى مع الحفاظ على النسبة."""
        h, w = image.shape[:2]
        scale_w = self.max_display_width / w
        scale_h = self.max_display_height / h
        # نستخدم أصغر معامل لضمان أن كلا البعدين يتناسبان
        # نستخدم min(1.0, ...) لمنع تكبير الصور الصغيرة جداً
        scale = min(scale_w, scale_h, 1.0)

        if scale < 1.0: # فقط قم بتغيير الحجم إذا كانت الصورة أكبر من اللازم
            new_w = int(w * scale)
            new_h = int(h * scale)
            # استخدام INTER_AREA هو الأفضل للتصغير
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.display_scale = scale # حفظ معامل التصغير
            print(f"Image resized for display. Scale: {self.display_scale:.2f}")
            return resized_image
        else:
            self.display_scale = 1.0 # لا يوجد تغيير في الحجم
            print("Image displayed at original size.")
            return image.copy() # إرجاع نسخة لتجنب التعديل على الأصل

    def _mouse_callback(self, event, x, y, flags, param):
        """وظيفة معالجة أحداث الفأرة."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                # --- تحويل إحداثيات الشاشة إلى إحداثيات الصورة الأصلية ---
                original_x = int(x / self.display_scale)
                original_y = int(y / self.display_scale)
                # ---------------------------------------------------------

                # رسم دائرة على الصورة المعروضة (بالإحداثيات x, y)
                cv2.circle(self.image_display, (x, y), 5, (0, 0, 255), -1)
                # تحديث النافذة لعرض الدائرة فورًا
                cv2.imshow(self.display_window_name, self.image_display)

                # تخزين الإحداثيات الأصلية
                self.points.append([original_x, original_y])
                print(f"Point {len(self.points)} selected: Original Coords ({original_x}, {original_y}), Display Coords ({x}, {y})")

                if len(self.points) == 4:
                    print("All 4 points selected. Press 'c' to confirm and continue, or 'r' to reset.")
            else:
                print("Already selected 4 points. Press 'c' to confirm or 'r' to reset.")

    def select_points_manually(self):
        """
        يعرض الصورة بحجم مناسب للشاشة ويسمح للمستخدم بتحديد 4 نقاط يدويًا.
        """
        self.read_image() # يقرأ الصورة ويخزنها في self.original_image
        self.store_process_image("0_original.jpg", self.original_image)

        # تغيير حجم الصورة للعرض وتخزينها في self.image_display
        self.image_display = self._resize_for_display(self.original_image)

        # إنشاء نافذة قابلة لتغيير الحجم
        cv2.namedWindow(self.display_window_name, cv2.WINDOW_NORMAL)
        # ضبط حجم النافذة ليتناسب مع الصورة المعروضة (اختياري لكن مفيد)
        cv2.resizeWindow(self.display_window_name, self.image_display.shape[1], self.image_display.shape[0])
        cv2.setMouseCallback(self.display_window_name, self._mouse_callback)

        print("Please click on the 4 corners of the table in any order.")
        print("Press 'r' to reset points.")
        print("Press 'c' to confirm points when 4 are selected.")
        print("Press 'q' to quit.")

        # عرض الصورة الأولية (المصغرة إذا لزم الأمر)
        cv2.imshow(self.display_window_name, self.image_display)

        while True:
            # لم نعد بحاجة لتحديث العرض هنا إلا إذا أردنا إضافة تأثيرات بصرية
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'): # إعادة تعيين النقاط
                self.points = []
                # إعادة رسم الصورة المعروضة من النسخة المصغرة الأصلية
                self.image_display = self._resize_for_display(self.original_image)
                cv2.imshow(self.display_window_name, self.image_display) # تحديث العرض
                print("Points reset. Please select 4 corners again.")

            elif key == ord('c'): # تأكيد النقاط
                if len(self.points) == 4:
                    print("Points confirmed.")
                    break
                else:
                    print(f"Please select {4 - len(self.points)} more points before confirming.")

            elif key == ord('q'): # خروج
                print("Quitting selection.")
                cv2.destroyAllWindows()
                return False

        cv2.destroyAllWindows()

        if len(self.points) != 4:
             print("Error: Did not select exactly 4 points.")
             return False

        # النقاط في self.points هي بالفعل بالإحداثيات الأصلية
        pts_np = np.array(self.points, dtype="float32")
        self.contour_with_max_area_ordered = self.order_points(pts_np)

        # رسم النقاط المرتبة على الصورة الأصلية للتأكيد
        self.image_with_points_plotted = self.original_image.copy() # ارسم على الأصلية
        for i, point in enumerate(self.contour_with_max_area_ordered):
            point_coordinates = (int(point[0]), int(point[1]))
            cv2.circle(self.image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)
            cv2.putText(self.image_with_points_plotted, str(i), point_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.store_process_image("1_manual_points_ordered.jpg", self.image_with_points_plotted)

        return True

    def execute(self):
        """
        ينفذ عملية استخراج الجدول بعد تحديد النقاط يدويًا.
        """
        if not self.select_points_manually():
             print("Manual point selection failed or was cancelled.")
             return None

        # العمليات اللاحقة تتم على الصورة الأصلية والنقاط الأصلية
        self.calculate_new_width_and_height_of_image() # تستخدم النقاط الأصلية
        self.apply_perspective_transform() # تستخدم الصورة الأصلية والنقاط الأصلية
        self.store_process_image("2_perspective_corrected.jpg", self.perspective_corrected_image)

        self.add_10_percent_padding()
        self.store_process_image("3_perspective_corrected_with_padding.jpg", self.perspective_corrected_image_with_padding)

        print("Manual table extraction complete.")
        return self.perspective_corrected_image

    # --- الدوال المساعدة (معظمها كما هي، لكن تأكد من أنها تستخدم الصورة الأصلية عند الحاجة) ---

    def read_image(self):
        # تم تغيير اسم المتغير إلى original_image
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to load image from path: {self.image_path}")
        print(f"Image loaded successfully from {self.image_path}")

    def order_points(self, pts):
        # لا تغيير هنا
        if pts.shape != (4, 2):
             if pts.shape == (4, 1, 2):
                 pts = pts.reshape(4, 2)
             else:
                 raise ValueError(f"Input points shape unexpected: {pts.shape}. Expected (4, 2) or (4, 1, 2).")

        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def calculateDistanceBetween2Points(self, p1, p2):
        # لا تغيير هنا
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

    def calculate_new_width_and_height_of_image(self):
        # لا تغيير هنا، تستخدم النقاط المرتبة (بالإحداثيات الأصلية)
        (tl, tr, br, bl) = self.contour_with_max_area_ordered
        widthA = self.calculateDistanceBetween2Points(br, bl)
        widthB = self.calculateDistanceBetween2Points(tr, tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = self.calculateDistanceBetween2Points(tr, br)
        heightB = self.calculateDistanceBetween2Points(tl, bl)
        maxHeight = max(int(heightA), int(heightB))
        self.new_image_width = maxWidth
        self.new_image_height = maxHeight
        print(f"Calculated new dimensions: Width={self.new_image_width}, Height={self.new_image_height}")

    def apply_perspective_transform(self):
        # تستخدم الصورة الأصلية والنقاط المرتبة (بالإحداثيات الأصلية)
        pts1 = np.float32(self.contour_with_max_area_ordered)
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height], [0, self.new_image_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # تأكد من استخدام الصورة الأصلية هنا
        self.perspective_corrected_image = cv2.warpPerspective(self.original_image, matrix, (self.new_image_width, self.new_image_height))

    def add_10_percent_padding(self):
        # لا تغيير هنا
        padding_percent = 0.10
        image_height, image_width = self.perspective_corrected_image.shape[:2]
        padding_h = int(image_height * padding_percent)
        padding_w = int(image_width * padding_percent)
        padding_top = max(0, padding_h)
        padding_bottom = max(0, padding_h)
        padding_left = max(0, padding_w)
        padding_right = max(0, padding_w)
        self.perspective_corrected_image_with_padding = cv2.copyMakeBorder(
            self.perspective_corrected_image,
            padding_top, padding_bottom, padding_left, padding_right,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        print(f"Added padding: top/bottom={padding_top}, left/right={padding_left}")

    def store_process_image(self, file_name, image):
        # لا تغيير هنا
        if image is None:
             print(f"Warning: Attempted to save a None image as {file_name}")
             return
        self._ensure_process_dir_exists()
        path = os.path.join(self.process_dir, file_name)
        try:
            cv2.imwrite(path, image)
            print(f"Saved process image: {path}")
        except Exception as e:
            print(f"Error saving image {path}: {e}")


# --- طريقة الاستخدام (كما هي) ---
if __name__ == "__main__":
    # image_path = r"C:\Users\Owner\Documents\test\mm\yaloo\17.jpg" #
    image_path =r"D:\smarcard\sorc\asa\asa\neww\17.jpg" 

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
    else:
        # يمكنك تمرير أبعاد عرض مختلفة إذا أردت
        # manual_extractor = ManualTableExtractor(image_path, max_display_width=1000, max_display_height=700)
        manual_extractor = ManualTableExtractor(image_path)
        extracted_table = manual_extractor.execute()

        if extracted_table is not None:
            cv2.imshow("Extracted Table (Manual)", extracted_table)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Process finished.")
        else:
            print("Process could not be completed.")