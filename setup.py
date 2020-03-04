from setuptools import setup, find_packages

setup(name="shfl", packages=find_packages(), install_requires=['keras', 'numpy', 'emnist', 'scikit-learn', 'pytest', 'tensorflow',
                                                               'scipy'])
