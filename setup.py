import setuptools

setuptools.setup(
    name="daco",
    version="0.0.3",
    author="Jon Vegard Sparre",
    author_email="jon.vegard.sparre@nav.no",
    description="A package for comparing two datasets",
    url="https://navikt.github.io/ai-lab-daco/",
    packages=['daco', 'daco_plot'],
    package_dir={'daco': 'src', 'daco_plot': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'cycler==0.10.0',
        'daco==0.0.1',
        'kiwisolver==1.0.1',
        'matplotlib==3.0.2',
        'numpy==1.16.0',
        'pandas==0.23.4',
        'pyparsing==2.3.1',
        'python-dateutil==2.7.5',
        'pytz==2018.9',
        'scipy==1.2.0',
        'seaborn==0.9.0',
        'six==1.12.0',
    ]
)