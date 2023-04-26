from setuptools import setup, find_packages


setup(
    name="ffm_torch",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13.0",
    ],
)
