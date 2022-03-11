#!/usr/bin/python3

from setuptools import setup, find_packages


setup(name='jkonet',
      version='1.0',
      description='Proximal Optimal Transport Modeling of Population Dynamics',
      url='https://github.com/bunnech/jkonet',
      author='Charlotte Bunne',
      author_email='bunnec@ethz.ch',
      license='MIT',
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[],
      zip_safe=False)
