import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyba",
    version="0.1",
    author="Semih Günel",
    packages=["pyba"],
    description="Python Bundle Adjustment Routines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/semihgunel/PyBundleAdjustment"
)