import re
from setuptools import setup


def get_requirements(filename: str):
    with open(filename) as f:
        requirement_file = f.read().strip().split("\n")
    requirements = []
    for line in requirement_file:
        if line.startswith("-r "):
            requirements += get_requirements(line.split()[1])
        else:
            requirements.append(line)
    return requirements


def find_version(filepath: str):
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name="afo",
    version=find_version("afo/__init__.py"),
    author="Juan Villamizar",
    author_email="Juan.Villamizar@amd.com",
    description="Test items that considered as part of the AFO criteria",
    long_description="",
    entry_points={
        "console_scripts": ["afo=afo.command_line:cli"],
    },
    include_package_data=True,
    packages=["afo"],
    install_requires=get_requirements("requirements.txt"),
    # package_dir={"rocBlaster": "rocBlaster/"},
    # package_data={"rocBlaster": ["*.so"]},
    # zip_safe=False,
)
