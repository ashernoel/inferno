from setuptools import setup, find_packages

setup(
    name='inferno-ml',
    version='0.1.0',
    author='Asher Noel',
    author_email='asher13a@gmail.com',
    description='A package for accelerating deep learning inference',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ashernoel/inferno', 
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', 
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8', 
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',  
)