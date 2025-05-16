from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="activeflow",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    description="activeflow",
    author="Active Flow",
    author_email="activeflow@mails.xxx.edu.cn",
)