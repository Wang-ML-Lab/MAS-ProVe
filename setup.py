from setuptools import setup, find_packages

setup(
    name="mas_proceval",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
    ],
    author="Haizhou Shi & Vishal V @ Rutgers ML Lab",
    author_email="haizhou.shi.057@gmail.com",
    description="A simple package that implements the library of the process evaluation for multi-agent systems.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/haizhou-shi/mas-process-eval",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)