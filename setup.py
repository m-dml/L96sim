import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="L96sim",
    version="0.0.1",
    author="David Greenberg",
    author_email="david.greenberg@hzg.de",
    description="Efficient simulation from L96 model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m-dml/L96sim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    python_requires='>=3.6',
)