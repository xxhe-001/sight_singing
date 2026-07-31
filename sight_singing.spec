# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('midi_generator.py', '.')
    ],
    hiddenimports=[
        'sklearn',
        'librosa',
        'numpy',
        'scipy',
        'fastdtw',
        'pretty_midi',
        'flask',
        'flask_cors',
        'soundfile',
        'audioread',
        'resampy',
        'numba',
        'pooch',
        'joblib',
        'llvmlite',
        '_soundfile_data'   # 有时需要这个
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='视唱评分',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,               # 开启 UPX 压缩，缩小体积
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,           # 如果想隐藏黑窗口就改成 False，但不推荐，方便调试
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None               # 如果有 .ico 图标可以填上路径
)