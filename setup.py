import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='nonstationarities',
    version='0.0.1',
    author='Guy Wilson',
    author_email='ghwilson@stanford.edu',
    description='Code repository associated to the publication \'Long-term unsupervised recalibration of cursor BCIs\'',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/guyhwilson/nonstationarities',
    project_urls = {
        "Bug Tracker": "https://github.com/guyhwilson/nonstationarities/issues"
    },
    license='MIT',
    packages=['utils'],
    install_requires=['numba', 'numpy', 'scipy', 'scikit-learn'],
)