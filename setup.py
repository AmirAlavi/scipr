#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['scikit-learn', 'numpy', 'scipy', 'torch>=1.2']

test_requirements = ['anndata']

setup_requirements = [ ]

extra_requirements = {
        'tensorboard':  ['tensorboard>=1.14'],
        'test': test_requirements
}

setup(
    author="Amir Alavi",
    author_email='amiralavi@cmu.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Single Cell Iterative Point set Registration (SCIPR)",
    install_requires=requirements,
    extras_require=extra_requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='scipr',
    name='scipr',
    packages=find_packages(include=['scipr', 'scipr.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/amiralavi/scipr',
    version='0.2.1',
    zip_safe=False,
)
