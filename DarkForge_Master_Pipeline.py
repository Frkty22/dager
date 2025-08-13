# DarkForge-X Master Control Program
# SHADOW-CORE MODE: ACTIVE - Final Architecture
# Mission: Execute the pipeline with direct, unaltered calls to the now GUI-compatible modules.
# All complex logic is correctly delegated to the specialized sub-modules.

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import cv2
import threading
import easyocr
import pandas as pd
from datetime import datetime
import sys # --- ADDED ---

# --- ADDED: Function to find files in PyInstaller's temp folder ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- Import Core Functional Modules ---
# This now assumes readdagre.py has been modified to be GUI-safe.
try:
    from ciles import ManualTableExtractor
    from color import preprocess_image
    from tapel import SmartRowExtractor
    from raws3 import AdvancedTableExtractor
    from readname import enhance_text_for_ocr
    # from readdagre import SmartDigitExtractor
    from model1 import YOLODigitExtractor      # للدقة الجيدة
    from model2 import MultiStageDigitExtractor  # للدقة العالية
except ImportError as e:
    messagebox.showerror("Module Import Error", f"A required module is missing: {e}. Please ensure all .py files are in the same directory.")
    exit()

# --- MODIFIED: Use resource_path for all model files ---
YOLO_CELL_MODEL_PATH = resource_path('bahh.pt')
YOLO_ROW_MODEL_PATH = resource_path('best2.pt')
TENSORFLOW_MODEL_PATH = resource_path('digit_recognition_model.h5')

# --- Output Directory Management ---
OUTPUT_DIR = "output_pipeline"
CORRECTED_TABLE_DIR = os.path.join(OUTPUT_DIR, "1_corrected_table")
PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, "2_preprocessed_table")
ROWS_DIR = os.path.join(OUTPUT_DIR, "3_extracted_rows")
CELLS_DIR = os.path.join(OUTPUT_DIR, "4_extracted_cells")

class EditableTreeview(ttk.Treeview):
    """فئة مخصصة للجدول القابل للتعديل مع دعم اللغة العربية"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Double-1>", self.on_double_click)
        self.bind("<Key>", self.on_key_press)
        self.bind("<Button-1>", self.on_click)
        self.editing_item = None
        self.editing_column = None
        self.entry_widget = None
        self.current_column_index = 0
        self.selected_cell = None
        
    def on_click(self, event):
        """تحديد الخلية عند النقر البسيط"""
        region = self.identify("region", event.x, event.y)
        if region == "cell":
            item = self.identify_row(event.y)
            column = self.identify_column(event.x)
            if item and column:
                columns = self['columns']
                if column in columns:
                    self.current_column_index = list(columns).index(column)
                    self.selected_cell = (item, column)
                    
    def on_double_click(self, event):
        """بدء تحرير الخلية عند النقر المزدوج"""
        region = self.identify("region", event.x, event.y)
        if region == "cell":
            item = self.identify_row(event.y)
            column = self.identify_column(event.x)
            if item and column:
                self.start_edit(item, column)
                
    def on_key_press(self, event):
        """التنقل بالأسهم"""
        if not self.selection():
            return
            
        current_item = self.selection()[0]
        
        if event.keysym == 'Up':
            prev_item = self.prev(current_item)
            if prev_item:
                self.selection_set(prev_item)
                self.focus(prev_item)
        elif event.keysym == 'Down':
            next_item = self.next(current_item)
            if next_item:
                self.selection_set(next_item)
                self.focus(next_item)
        elif event.keysym in ['Left', 'Right']:
            # التنقل بين الأعمدة
            self.move_between_columns(event.keysym, current_item)
        elif event.keysym == 'Return' or event.keysym == 'F2':
            # بدء التحرير بالضغط على Enter أو F2
            column = "#1"  # العمود الأول افتراضياً
            self.start_edit(current_item, column)
            
    def move_between_columns(self, direction, item):
        """التنقل بين الأعمدة باستخدام الأسهم اليمين واليسار"""
        columns = self['columns']
        if not hasattr(self, 'current_column_index'):
            self.current_column_index = 0
            
        if direction == 'Right' and self.current_column_index < len(columns) - 1:
            self.current_column_index += 1
        elif direction == 'Left' and self.current_column_index > 0:
            self.current_column_index -= 1
            
        # تحديد الخلية الحالية بصرياً
        self.focus(item)
        
    def start_edit(self, item, column):
        """بدء تحرير الخلية"""
        if self.entry_widget:
            self.finish_edit()
            
        self.editing_item = item
        self.editing_column = column
        
        # الحصول على موقع الخلية
        x, y, width, height = self.bbox(item, column)
        
        # إنشاء صندوق النص للتحرير
        self.entry_widget = tk.Entry(self, justify='right')  # محاذاة النص لليمين
        self.entry_widget.place(x=x, y=y, width=width, height=height)
        
        # تعبئة النص الحالي
        current_value = self.set(item, column)
        self.entry_widget.insert(0, current_value)
        self.entry_widget.select_range(0, tk.END)
        self.entry_widget.focus()
        
        # ربط الأحداث
        self.entry_widget.bind("<Return>", lambda e: self.finish_edit())
        self.entry_widget.bind("<Escape>", lambda e: self.cancel_edit())
        self.entry_widget.bind("<FocusOut>", lambda e: self.finish_edit())
        
    def finish_edit(self):
        """إنهاء التحرير وحفظ التغييرات"""
        if self.entry_widget and self.editing_item and self.editing_column:
            new_value = self.entry_widget.get()
            self.set(self.editing_item, self.editing_column, new_value)
            self.entry_widget.destroy()
            self.entry_widget = None
            self.editing_item = None
            self.editing_column = None
            
            # إعادة حساب الألوان إذا تم تغيير الدرجات
            if hasattr(self.master.master, 'update_row_colors'):
                self.master.master.update_row_colors()
                
    def cancel_edit(self):
        """إلغاء التحرير بدون حفظ"""
        if self.entry_widget:
            self.entry_widget.destroy()
            self.entry_widget = None
            self.editing_item = None
            self.editing_column = None

class MasterApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("DarkForge-X: نظام ذكي لتحليل الوثائق")
        self.root.geometry("1200x800")

        # --- تصميم الواجهة مع دعم اللغة العربية ---
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=6, relief="flat", background="#333", foreground="white")
        style.map("TButton", background=[('active', '#555')])
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
        style.configure("Treeview", rowheight=30, font=('Arial', 9))
        
        # إعداد ألوان خاصة للصفوف الصحيحة
        style.configure("Correct.Treeview", background="#90EE90")  # أخضر فاتح

        # --- الإطار الرئيسي ---
        main_frame = tk.Frame(root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- إطار أزرار التحكم ---
        control_frame = tk.Frame(main_frame, bg="#2b2b2b")
        control_frame.pack(fill=tk.X, pady=5)
        
        # الصف الأول من الأزرار
        top_buttons_frame = tk.Frame(control_frame, bg="#2b2b2b")
        top_buttons_frame.pack(fill=tk.X, pady=2)
        
        # زر تشغيل البرنامج
        self.select_button = ttk.Button(top_buttons_frame, text="تحديد الصورة وتشغيل النظام", command=self.start_pipeline_thread)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        # زر تصدير إلى Excel
        self.export_button = ttk.Button(top_buttons_frame, text="تصدير إلى Excel", command=self.export_to_excel)
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # زر إعادة حساب الألوان
        self.recalc_button = ttk.Button(top_buttons_frame, text="إعادة حساب الألوان", command=self.update_row_colors)
        self.recalc_button.pack(side=tk.LEFT, padx=5)
        
        # الصف الثاني - أزرار اختيار الدقة
        accuracy_frame = tk.Frame(control_frame, bg="#2b2b2b")
        accuracy_frame.pack(fill=tk.X, pady=2)
        
        # تسمية لأزرار الدقة
        accuracy_label = tk.Label(accuracy_frame, text="دقة التعرف:", bg="#2b2b2b", fg="white", font=('Arial', 10, 'bold'))
        accuracy_label.pack(side=tk.LEFT, padx=5)
        
        # متغير لتتبع نوع الدقة المختار
        self.accuracy_mode = tk.StringVar()
        self.accuracy_mode.set("good")  # الافتراضي: دقة جيدة
        
        # زر الدقة الجيدة
        self.good_accuracy_btn = tk.Radiobutton(accuracy_frame, text="جيدة (سريع)", 
                                                variable=self.accuracy_mode, value="good",
                                                command=self.change_accuracy_mode,
                                                bg="#2b2b2b", fg="white", selectcolor="#333",
                                                activebackground="#2b2b2b", activeforeground="white",
                                                font=('Arial', 9))
        self.good_accuracy_btn.pack(side=tk.LEFT, padx=5)
        
        # زر الدقة العالية
        self.high_accuracy_btn = tk.Radiobutton(accuracy_frame, text="عالية (بطيء)", 
                                                variable=self.accuracy_mode, value="high",
                                                command=self.change_accuracy_mode,
                                                bg="#2b2b2b", fg="white", selectcolor="#333",
                                                activebackground="#2b2b2b", activeforeground="white",
                                                font=('Arial', 9))
        self.high_accuracy_btn.pack(side=tk.LEFT, padx=5)
        
        # --- إطار الجدول ---
        table_frame = tk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # تعريف الأعمدة باللغة العربية ومن اليمين إلى اليسار
        cols = ("التوقيع", "الدرجة الرابعة", "الدرجة الثالثة", "الدرجة الثانية", "الدرجة الأولى", "الاسم", "الرقم")
        
        # إنشاء الجدول القابل للتحرير
        self.tree = EditableTreeview(table_frame, columns=cols, show='headings')
        
        # إعداد العناوين والأعمدة
        for col in cols:
            self.tree.heading(col, text=col, anchor='center')
            if col == "الاسم":
                self.tree.column(col, width=200, anchor='center')
            else:
                self.tree.column(col, width=100, anchor='center')
        
        # أشرطة التمرير
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # ترتيب العناصر
        vsb.pack(side='left', fill='y')  # تم تغيير الجانب إلى اليسار
        hsb.pack(side='bottom', fill='x')
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # --- شريط الحالة ---
        self.status_var = tk.StringVar()
        self.status_var.set("جاهز. في انتظار الأوامر.")
        status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, 
                              anchor='e', bg="#333", fg="white", font=('Arial', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- متغيرات البيانات ---
        self.table_data = []  # لحفظ بيانات الجدول
        
        # --- تهيئة النماذج ---
        self.ocr_reader_ar = None
        self.digit_extractor = None
        self.initialize_dependencies()
        
        # ربط حدث تغيير الخلايا لإعادة حساب الألوان
        self.tree.bind("<<TreeviewSelect>>", self.on_selection_change)
        
    def on_selection_change(self, event):
        """معالج تغيير التحديد في الجدول"""
        pass  # يمكن إضافة وظائف إضافية هنا
        
    def update_row_colors(self):
        """تحديث ألوان الصفوف بناءً على صحة مجموع الدرجات"""
        for item in self.tree.get_children():
            values = self.tree.item(item, 'values')
            if len(values) >= 7:
                try:
                    # الحصول على الدرجات (من اليمين إلى اليسار)
                    grade4 = float(values[1]) if values[1] != "-" and values[1] != "" else 0  # الدرجة الرابعة
                    grade3 = float(values[2]) if values[2] != "-" and values[2] != "" else 0  # الدرجة الثالثة  
                    grade2 = float(values[3]) if values[3] != "-" and values[3] != "" else 0  # الدرجة الثانية
                    grade1 = float(values[4]) if values[4] != "-" and values[4] != "" else 0  # الدرجة الأولى
                    
                    # حساب المجموع
                    calculated_sum = grade1 + grade2 + grade3
                    
                    # التحقق من صحة المجموع مع تساهل في الفاصلة العشرية
                    if abs(calculated_sum - grade4) < 0.01:  # تساهل 0.01
                        # تطبيق اللون الأخضر
                        self.tree.item(item, tags=("correct",))
                    else:
                        # إزالة التلوين
                        self.tree.item(item, tags=("incorrect",))
                        
                except ValueError:
                    # في حالة وجود قيم غير رقمية
                    self.tree.item(item, tags=("incorrect",))
                    
        # تطبيق الألوان
        self.tree.tag_configure("correct", background="#90EE90")  # أخضر فاتح
        self.tree.tag_configure("incorrect", background="white")  # أبيض للصفوف غير الصحيحة
        
    def change_accuracy_mode(self):
        """تغيير نمط الدقة وإعادة تهيئة النموذج"""
        def reinit_models():
            self.update_status("جاري تغيير نمط الدقة...")
            try:
                if self.accuracy_mode.get() == "good":
                    # استخدام model1 للدقة الجيدة - # --- MODIFIED ---
                    self.digit_extractor = YOLODigitExtractor(yolo_model_path=resource_path('cels.pt'), digit_model_path=TENSORFLOW_MODEL_PATH)
                    self.update_status("تم تفعيل الدقة الجيدة (سريع). جاهز.")
                else:
                    # استخدام model2 للدقة العالية مع API key - # --- MODIFIED ---
                    API_KEY = "K83990210688957"  # نفس API key من model2.py
                    self.digit_extractor = MultiStageDigitExtractor(
                        digit_model_path=TENSORFLOW_MODEL_PATH,
                        ocr_space_api_key=API_KEY
                    )
                    self.update_status("تم تفعيل الدقة العالية (بطيء). جاهز.")
            except Exception as e:
                self.update_status("خطأ: فشل في تغيير نمط الدقة.")
                messagebox.showerror("خطأ", f"فشل في تغيير نمط الدقة: {e}")
        
        # تشغيل التغيير في خيط منفصل
        threading.Thread(target=reinit_models, daemon=True).start()
    
    def export_to_excel(self):
        """تصدير الجدول إلى ملف Excel منسق - طريقة main_controller"""
        if not self.tree.get_children():
            messagebox.showwarning("تصدير فارغ", "لا توجد بيانات لتصديرها.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
            title="حفظ كملف Excel"
        )
        if not file_path: return

        try:
            # جمع البيانات من الجدول
            data = [list(self.tree.item(item_id, 'values')) for item_id in self.tree.get_children()]
            cols_rtl = list(self.tree['columns'])
            df = pd.DataFrame(data, columns=cols_rtl)
            df.to_excel(file_path, index=False, engine='openpyxl')
            messagebox.showinfo("نجاح", f"تم تصدير البيانات بنجاح إلى:\n{file_path}")
        except Exception as e:
            messagebox.showerror("خطأ في التصدير", f"حدث خطأ أثناء تصدير الملف:\n{e}")

    def initialize_dependencies(self):
        """Initializes all required models in a background thread."""
        def init():
            self.update_status("Initializing AI Models...")
            try:
                # السطر الجديد والصحيح
                self.ocr_reader_ar = easyocr.Reader(['ar'], gpu=False, model_storage_directory=resource_path('easyocr'))
                # Initialize the grade reader with its model path - # --- MODIFIED ---
                self.digit_extractor = YOLODigitExtractor(yolo_model_path=resource_path('cels.pt'), digit_model_path=TENSORFLOW_MODEL_PATH)
                self.update_status("Models Ready. Awaiting command.")
            except Exception as e:
                self.update_status("Error: Failed to initialize AI models.")
                messagebox.showerror("Model Error", f"Could not initialize models: {e}")
        
        threading.Thread(target=init, daemon=True).start()

    def start_pipeline_thread(self):
        self.select_button.config(state=tk.DISABLED)
        for i in self.tree.get_children():
            self.tree.delete(i)
        pipeline_thread = threading.Thread(target=self.run_full_pipeline, daemon=True)
        pipeline_thread.start()

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def run_full_pipeline(self):
        """
        This pipeline now contains only orchestration logic. All complex data processing
        is correctly delegated to the imported modules.
        """
        try:
            # Stages 0-4: Setup, Point Selection, Preprocessing, Row/Cell Extraction (unchanged)
            self.update_status("Awaiting image selection...")
            image_path = filedialog.askopenfilename(title="Select Table Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
            if not image_path:
                self.update_status("Operation cancelled by user."); self.select_button.config(state=tk.NORMAL); return
            for d in [OUTPUT_DIR, CORRECTED_TABLE_DIR, PREPROCESSED_DIR, ROWS_DIR, CELLS_DIR]: os.makedirs(d, exist_ok=True)
            self.update_status("Awaiting manual 4-point table selection...")
            manual_extractor = ManualTableExtractor(image_path)
            corrected_table_img = manual_extractor.execute()
            if corrected_table_img is None: raise Exception("Manual point selection failed or was cancelled.")
            corrected_table_path = os.path.join(CORRECTED_TABLE_DIR, "corrected_table.png")
            cv2.imwrite(corrected_table_path, corrected_table_img)
            self.update_status("Stage 1/6: Table perspective corrected.")
            preprocessed_image_path = os.path.join(PREPROCESSED_DIR, "111.jpg")
            preprocess_image(corrected_table_path, preprocessed_image_path, padding_percent=0.0, scale_factor=2)
            self.update_status("Stage 2/6: Image preprocessed for detection.")
            row_extractor = SmartRowExtractor(model_path=YOLO_ROW_MODEL_PATH)
            row_extractor.crop_and_save(image_path=preprocessed_image_path, output_folder=ROWS_DIR, conf=0.7,iou=0.4, padding=15)
            self.update_status("Stage 3/6: Table rows extracted.")
            cell_extractor = AdvancedTableExtractor(model_path=YOLO_CELL_MODEL_PATH)
            row_files = sorted([f for f in os.listdir(ROWS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
            if not row_files: raise Exception("No rows were extracted from the table.")
            for i, row_filename in enumerate(row_files):
                self.update_status(f"Stage 4/6: Processing Row {i+1} of {len(row_files)}...")
                cell_extractor.process_image(os.path.join(ROWS_DIR, row_filename), CELLS_DIR)
            self.update_status("Stage 5/6: All cells extracted. Initiating OCR...")
            
            # Stage 6: Simplified and Correct Data Aggregation
            final_results = []
            result_folders = sorted([d for d in os.listdir(CELLS_DIR) if os.path.isdir(os.path.join(CELLS_DIR, d))])
            for i, folder_name in enumerate(result_folders):
                self.update_status(f"Stage 6/6: Reading data from Row {i+1}...")
                row_data = {"Row": i + 1, "Name": "-", "Grade 1": "-", "Grade 2": "-", "Grade 3": "-", "Grade 4": "-", "Signature": "-"}
                current_cells_path = os.path.join(CELLS_DIR, folder_name, "cleaned_cells")
                if not os.path.exists(current_cells_path): continue
                cell_files = os.listdir(current_cells_path)
                
                # Name Reading (unchanged)
                name_file = next((f for f in cell_files if f.startswith('name')), None)
                if name_file and self.ocr_reader_ar:
                    enhanced_name_img = enhance_text_for_ocr(os.path.join(current_cells_path, name_file), show=False)
                    ocr_results = self.ocr_reader_ar.readtext(enhanced_name_img, paragraph=False)
                    if ocr_results: row_data["Name"] = ' '.join([res[1] for res in ocr_results])
                
                # Grade Reading - تحديث للتعامل مع النموذجين
                for grade_num in range(1, 5):
                    grade_key = f"Grade {grade_num}"
                    grade_prefix = f"grade_{grade_num}"
                    grade_file = next((f for f in cell_files if f.startswith(grade_prefix)), None)
                    if grade_file and self.digit_extractor:
                        grade_path = os.path.join(current_cells_path, grade_file)
                        
                        try:
                            if self.accuracy_mode.get() == "good":
                                # استخدام model1 (الدقة الجيدة)
                                result, _ = self.digit_extractor.extract_with_multiple_methods(image_path=grade_path, debug=False)
                            else:
                                # استخدام model2 (الدقة العالية)
                                result, _ = self.digit_extractor.run_extraction_pipeline(image_path=grade_path, debug=False)
                            
                            row_data[grade_key] = result if result else "-"
                        except Exception as e:
                            print(f"خطأ في قراءة الدرجة {grade_num}: {e}")
                            row_data[grade_key] = "-"

                # Signature detection (unchanged)
                if next((f for f in cell_files if f.startswith('signature')), None):
                    row_data["Signature"] = "Detected"
                final_results.append(row_data)

            # Final Display - تحديث لعرض البيانات بالترتيب الصحيح RTL
            self.update_status("تعبئة جدول النتائج...")
            for result in final_results:
                # ترتيب البيانات من اليمين إلى اليسار حسب الأعمدة الجديدة
                # ("التوقيع", "الدرجة الرابعة", "الدرجة الثالثة", "الدرجة الثانية", "الدرجة الأولى", "الاسم", "الرقم")
                rtl_values = [
                    result["Signature"],      # التوقيع
                    result["Grade 4"],        # الدرجة الرابعة  
                    result["Grade 3"],        # الدرجة الثالثة
                    result["Grade 2"],        # الدرجة الثانية
                    result["Grade 1"],        # الدرجة الأولى
                    result["Name"],           # الاسم
                    result["Row"]             # الرقم
                ]
                self.tree.insert("", tk.END, values=rtl_values)
                
            # تطبيق حساب الألوان بعد إدراج البيانات
            self.update_row_colors()
            
            messagebox.showinfo("نجح", f"تم تنفيذ النظام بنجاح. تم استخراج البيانات من {len(final_results)} صف.")

        except Exception as e:
            messagebox.showerror("Pipeline Error", f"An error occurred during execution:\n\n{str(e)}")
        finally:
            self.update_status("Ready. Awaiting command.")
            self.select_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    # --- MODIFIED: Check for model files using the resource_path function ---
    # This check now works correctly both in development and as a packaged .exe
    # You might need to add other files like 'lasst.pt' if they are used in your other modules.
    models_to_check = [
        'bahh.pt', 
        'best2.pt', 
        'digit_recognition_model.h5',
        'cels.pt'  # Added because it's used directly in the code
    ]
    
    # We check the existence of each model file using our special function
    missing_models = [model for model in models_to_check if not os.path.exists(resource_path(model))]
    
    if missing_models:
        messagebox.showerror("Model Missing", f"Could not find required model file(s): {', '.join(missing_models)}. Ensure they are included when building the executable.")
    else:
        app_root = tk.Tk()
        app = MasterApplication(app_root)
        app_root.mainloop()