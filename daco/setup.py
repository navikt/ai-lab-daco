import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="daco",
    version="0.0.1",
    author="Jon Vegard Sparre",
    author_email="jon.vegard.sparre@nav.no",
    description="A package for comparing two datasets",
    url="https://navikt.github.io/ai-lab-daco/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)