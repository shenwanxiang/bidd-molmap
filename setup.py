import sys
import os
from setuptools import setup, find_packages
from setuptools import Command
from setuptools.command.test import test as TestCommand
from datetime import datetime
import molmap

def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]


with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

NAME = "molmap"
VERSION = molmap.__version__
AUTHOR = "WanXiang Shen"
DESCRIPTION = "MolMap: An Efficient Convolutional Neural Network Based Molecular Deep Learning Tool"
URL = "https://github.com/shenwanxiang/bidd-molmap"

REQUIRED_PYTHON_VERSION = (3, 7)
PACKAGES = find_packages(exclude = ['test', 'gallery', 'example', '.ipynb_checkpoints',])
INSTALL_DEPENDENCIES = parse_requirements('./requirements.txt')
SETUP_DEPENDENCIES = []
TEST_DEPENDENCIES = ["pytest"]
EXTRA_DEPENDENCIES = {"dev": ["pytest"]}

#PACKAGE_DATA = {'molmap': ['*.xlsx',  '*.fdef', '*.csv', '*.txt', '*.doc', '*.pkl']}
    
# 'dataset/*.csv', 
# 'feature/fingerprint/*.xlsx',
# 'feature/fingerprint/*.fdef', 
#print(PACKAGE_DATA)
    
if sys.version_info < REQUIRED_PYTHON_VERSION:
    sys.exit("Python >= 3.7 is required. Your version:\n" + sys.version)


class PyTest(TestCommand):
    """
    Use pytest to run tests
    """

    user_options = [("pytest-args=", "a", "Arguments to pass into py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name=NAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    version=VERSION,
    author=AUTHOR,
    packages=PACKAGES,
    include_package_data=True,
    #package_data = PACKAGE_DATA,
    install_requires=INSTALL_DEPENDENCIES,
    setup_requires=SETUP_DEPENDENCIES,
    tests_require=TEST_DEPENDENCIES,
    extras_require=EXTRA_DEPENDENCIES,
    cmdclass={"test": PyTest},
)