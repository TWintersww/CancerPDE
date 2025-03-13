from setuptools import setup, find_packages

setup(
    name="CancerPDE",         # Package name
    version="0.1.0",           # Version number
    packages=find_packages(),  # Automatically find submodules
    install_requires=[         # Dependencies
        "numpy",
        "matplotlib"
    ],
    author="Evan Wu"
)
