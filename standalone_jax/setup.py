from setuptools import setup, find_packages


setup(
    name="ffm_jax",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["jax", "jaxlib", "equinox"],
)
