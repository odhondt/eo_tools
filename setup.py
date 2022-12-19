from setuptools import setup, find_namespace_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='s1pro',
    setup_requires=['setuptools_scm'],
    use_scm_version=True,
    description="A Sentinel-1 processor for backscatter intensity, InSAR coherence and Dual-pol H/a decomposition from SLC data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/SAR-ARD/S1_NRB",
    author="Johannes Löw, Steven Hill",
    author_email="Johannes Löw <johannes.loew@geo.uni-halle.de>",
    packages=find_namespace_packages(where='.'),
    include_package_data=True,
    install_requires=['gdal',
                      'click',
                      'lxml',
                      'pystac',
                      'pyroSAR',
                      'spatialist',
                      'pip',
                      'tqdm',
                      'pystac',
                      'geopandas',
                      'pandas',
                      'asf_search'],
    extras_require={
          'docs': ['sphinx', 'sphinxcontrib-bibtex', 'nbsphinx', 'sphinx_rtd_theme', 'sphinx-toolbox'],
    },
    python_requires='>=3.8',
    license='MIT',
    zip_safe=False,
    entry_points={
        'console_scripts': ['s1pro=s1pro.cli:cli']
    }
)
