# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import get_module_path

# زيادة عمق الاستدعاءات لضمان عدم حدوث خطأ
sys.setrecursionlimit(sys.getrecursionlimit() * 10)

# --- كود جديد وقوي لجمع المكتبات ---
# قائمة ملفات النماذج الخاصة بك
datas_list = [
    ('bahh.pt', '.'),
    ('best2.pt', '.'),
    ('cels.pt', '.'),
    ('digit_recognition_model.h5', '.'),
    ('lasst.pt', '.'),
]

# قائمة بالمكتبات الضخمة التي نريد إجبار PyInstaller على نسخها بالكامل
# هذا يضمن عدم ضياع أي ملف .dll أو ملفات ضرورية أخرى
force_include_libs = ['tensorflow', 'torch', 'ultralytics', 'easyocr', 'cv2', 'pandas', 'skimage']

for lib in force_include_libs:
    try:
        # البحث عن مسار المكتبة داخل بيئة الخادم السحابي
        lib_path = get_module_path(lib)
        if lib_path:
            # إضافة المجلد بالكامل إلى قائمة النسخ
            datas_list.append((lib_path, lib))
            print(f'Successfully included {lib} from {lib_path}')
    except Exception as e:
        print(f'Warning: Could not include library {lib}. Reason: {e}')
# --- نهاية الكود الجديد ---


a = Analysis(
    ['DarkForge_Master_Pipeline.py'],
    pathex=[],
    binaries=[],
    # استخدام القائمة التي أنشأناها والتي تحتوي على النماذج والمكتبات الكاملة
    datas=datas_list,
    hiddenimports=['easyocr', 'sklearn.utils._typedefs'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='SUNSHE',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False, # مهم جدًا أن يكون False
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    binaries=a.binaries,
    datas=a.datas,
)
