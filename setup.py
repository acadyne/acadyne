from setuptools import setup, find_packages

setup(
    name="acadyne",
    version="0.2.4",
    description="Una biblioteca biónica para la manipulación de tensores numéricos y simbólicos, facilitando cálculos avanzados en computación científica.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jose Fabian Soltero Escobar",
    author_email="acadyne@gmail.com",
    url="https://github.com/acadyne/acadyne",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21",
        "sympy>=1.8",
        "scipy>=1.5",
    ],
    extras_require={
        "testing": ["pytest", "pytest-cov"],
    },
    include_package_data=True,
    keywords=["tensor", "computation", "symbolic", "mathematics", "simulation",'bionic',''],
    project_urls={
        "Documentation": "https://github.com/acadyne/acadyne/wiki",
        "Source": "https://github.com/acadyne/acadyne",
        "Tracker": "https://github.com/acadyne/acadyne/issues",
    },
)
