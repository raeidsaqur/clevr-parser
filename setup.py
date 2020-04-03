import sys
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = '0.0.5'
url = "https://github.com/raeidsaqur/clevr-parser"

install_requires = [
    'numpy',
    'scipy',
    'networkx',
    'scikit-learn',
    'scikit-image',
    'requests',
    'tabulate',
    'stanfordnlp',
    'spacy-transformers'
]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []
setup_requires = [] + pytest_runner
tests_require = ['pytest', 'pytest-cov', 'mock']

setup(
    name='clevr-parser',
    version=__version__,
    description='PGM factoring, object candidate proposal generator for CLEVR dataset',
    author='Raeid Saqur',
    author_email='raeidsaqur@cs.toronto.edu',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=url,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        'pytorch',
        'CLEVR',
        'language compositionality',
        'geometric learning',
        'graph neural networks'
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
)
