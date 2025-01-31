from setuptools import setup #, find_packages

# For compiling external C++ code, see:
# https://github.com/MLAlex1/DecisionTree_C_plus_plus_Python/blob/master/setup.py
# and https://towardsdatascience.com/calling-c-code-from-python-with-ctypes-module-58404b9b3929

CLASSIFIERS = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3 :: Only
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Operating System :: MacOS
"""

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='pyrea',
    version='1.0.12',
    author='Marcus D. Bloice, Bastian Pfeifer',
    license='MIT',
    author_email='marcus.bloice@medunigraz.at',
    description='Multi-view clustering with deep ensemble structures.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mdbloice/Pyrea',
    keywords='multi-view, clustering, ensemble clustering',
    python_requires='>=3.6',
    # package_dir = {"": "src"}
    # packages=find_packages("Parea/*", exclude=["test"]),
    # py_modules=["xyz"],
    packages=['pyrea'],
    install_requires=[
        'pandas>0.20.0',
        'numpy>1.0.0',
        'scikit-learn>1.1.0',
        'scipy>1.5.4',
        'deap>1.3.0',
    ],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f]
)
