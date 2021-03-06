import setuptools
import versioneer

setuptools.setup(
    name="daco",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Jon Vegard Sparre",
    author_email="jon.vegard.sparre@nav.no",
    description="A package for comparing two datasets",
    url="https://navikt.github.io/ai-lab-daco/",
    packages=['daco'],
    package_dir={'daco': 'daco'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # install_requires=[
    #     'cycler==0.10.0',
    #     'kiwisolver==1.0.1',
    #     'matplotlib==3.0.2',
    #     'numpy==1.16.1',
    #     'pandas==0.23.4',
    #     'pyparsing==2.3.1',
    #     'python-dateutil==2.7.5',
    #     'pytz==2018.9',
    #     'scipy==1.2.0',
    #     'seaborn==0.9.0',
    #     'six==1.12.0',
    # ]
)