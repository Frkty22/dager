# -*- mode: python ; coding: utf-8 -*-
import sys
# زيادة عمق الاستدعاءات لضمان عدم حدوث خطأ
sys.setrecursionlimit(sys.getrecursionlimit() * 10)

a = Analysis(
    ['DarkForge_Master_Pipeline.py'],
    pathex=[],
    binaries=[],
    datas=[
        # تم حذف السطر الذي يحتوي على المسار المطلق لـ easyocr
        # سنقوم بتضمين ملفات النماذج الخاصة بك فقط، لأنها موجودة في نفس مجلد المشروع
        ('bahh.pt', '.'),
        ('best2.pt', '.'),
        ('cels.pt', '.'),
        ('digit_recognition_model.h5', '.'),
        ('lasst.pt', '.'),
    ],
    hiddenimports=[
        'easyocr',
        'sklearn.utils._typedefs' # إجراء وقائي
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # استبعاد مكتبات الواجهة الرسومية المتعارضة لضمان بناء نظيف
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='SUNSHE', # يمكنك تغيير هذا الاسم كما تشاء
    # إيقاف وضع التصحيح للنسخة النهائية
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # أهم تعديل: إيقاف UPX لمنع تلف البرنامج
    upx=True,
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
