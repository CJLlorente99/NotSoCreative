from setuptools import setup

APP=['Layout_2_4.py']
OPTIONS = {
    'argv_emulation':True,
}

setup(
    app =APP,
    options={'py2app':OPTIONS},
    setup_requires=['py2app']
)