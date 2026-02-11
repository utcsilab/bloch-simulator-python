from setuptools import setup, Extension
import numpy as np

# Define the compiled extension here (keeps pyproject.toml simple)
ext_modules = [
    Extension(
        "bloch.bloch_simulator",
        sources=["bloch/bloch_simulator.c"],
        include_dirs=[np.get_include()],
    )
]

# Minimal setup call so setuptools sees the ext_modules at build time.
# Metadata is read from pyproject.toml (PEP 621).
setup(ext_modules=ext_modules)

