from setuptools import setup, find_packages

# read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="lettuceSee",
    url="https://github.com/Chizz98/LettuceSee",
    packages=find_packages(),
    author="Chris Dijkstra",
    author_email="chris_dijkstra98@hotmail.com",
    description="A package of image analysis algorithms suited for plants",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.2.2",
    install_requires=[
        "scikit-image",
        "scipy",
        "numpy",
        "networkx"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Intended Audience :: Science/Research"
    ]
)
