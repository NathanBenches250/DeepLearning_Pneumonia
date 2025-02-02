from setuptools import setup, find_packages

setup(
    name="chest-xray-pneumonia",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "transformers>=4.18.0",
        "flask>=2.0.1",
        "numpy>=1.19.5",
        "opencv-python>=4.5.3",
        "scikit-learn>=0.24.2",
        "pillow>=8.3.1",
        "werkzeug>=2.0.1",
    ],
    author="Blank",
    author_email="blank@blank.com",
    description="An AI-powered web application for detecting pneumonia from chest X-ray images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/blank/chest-xray-pneumonia",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)