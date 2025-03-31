from setuptools import setup, find_packages

setup(
    name="nof1",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    description="A simulation tool for trading strategies.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/nof1-trading-sim",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)