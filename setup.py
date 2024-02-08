from setuptools import setup, find_packages
from lettuceSee import __version__

setup(
    name="lettuceSee",
    url="https://github.com/Chizz98/LettuceSee",
    packages=find_packages(),
    author="Chris Dijkstra",
    author_email="chris_dijkstra98@hotmail.com",
    version=__version__,
    install_requires=[
        "scikit-image>=0.10.0"
    ]
)
