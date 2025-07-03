from setuptools import setup, find_packages

setup(
    name='call_center_1999',
    version='0.1.0',
    description='Análisis y modelado de datos del Call Center 1999',
    author='Anonymous',
    packages=find_packages(where='02_src'),
    package_dir={'': '02_src'},
    install_requires=[
        # Las dependencias principales están en requirements.txt
    ],
)
