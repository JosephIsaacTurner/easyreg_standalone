from setuptools import setup, find_packages
import os

setup(
    name='easyreg',
    version='0.1.0',
    author='Your Name',
    author_email='jiturner@bwh.harvard.edu',
    description="A standlone distribution of FreeSurfer's easyreg (registration and warping tool for MRI data)",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/josephisaacturner/easyreg_standalone',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        line.strip() for line in open('requirements.txt', 'r') if line.strip() and not line.strip().startswith('#')
    ],
    entry_points={
        'console_scripts': [
            'easyreg-mri=easyreg.mri_easyreg:main',
            'easywarp-mri=easyreg.mri_easywarp:main', 
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)
