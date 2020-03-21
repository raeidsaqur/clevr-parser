from setuptools import setup, find_packages

__version__ = '0.1.0'
url = "https://github.com/raeidsaqur/clevr-parser"

install_requires = []
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='clevr_parser',
    version=__version__,
    description='PGM factoring, object candidate proposal generator for CLEVR dataset',
    author='Raeid Saqur',
    author_email='raeidsaqur@cs.toronto.edu',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'CLEVR',
        'language compositionality',
        'graph-matching'
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
)
