from setuptools import setup, find_packages

setup(
    name="ALIGN_FL",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here
    ],
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="ICML submission repository for ALIGN_FL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ALIGN_FL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
