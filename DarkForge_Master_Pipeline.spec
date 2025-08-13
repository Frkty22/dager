# -*- mode: python ; coding: utf-8 -*-
# -*- mode: python ; coding: utf-8 -*-
import sys
# زيادة عمق الاستدعاءات التكرارية لحل مشكلة RecursionError
sys.setrecursionlimit(sys.getrecursionlimit() * 10)

a = Analysis(
    ['DarkForge_Master_Pipeline.py'],
    pathex=[],
    binaries=[],
    datas=[('bahh.pt', '.'), ('best2.pt', '.'), ('cels.pt', '.'), ('digit_recognition_model.h5', '.'), ('lasst.pt', '.')],
    hiddenimports=['easyocr','sklearn.utils._typedefs'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AI_Project_App',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
