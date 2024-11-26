from setuptools import setup, find_packages

setup(
    name='ran_face_recognizer',
    version='1.0.0',
    description='A Python library for face detection, encoding, and recognition using dlib.',
    author='Tejash Katuwal',
    author_email='tejashkatuwal99@gmail.com.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',  # Replace with the minimum version you require
        'opencv-python>=4.5.0',  # Replace with the minimum version you require
        'dlib>=19.24.0'  # Replace with the minimum version you require
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Adjust based on your library's Python compatibility
)
