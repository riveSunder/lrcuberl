from os.path import join, dirname, realpath
from setuptools import setup
import sys

setup(
    name="LRCubeRl",
    py_modules=["cube"],
    version='0.1',
    install_requires=["numpy"],
    description="Solving Rubik's Cube with reinforcement learning",
    author="Rive Sunder",
)
