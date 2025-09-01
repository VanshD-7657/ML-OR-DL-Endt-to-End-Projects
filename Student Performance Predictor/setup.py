from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DASH = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of requirements.
    :param file_path: Path to the requirements file.
    :return: List of requirements.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPHEN_E_DASH in requirements:
            requirements.remove(HYPHEN_E_DASH)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Vansh',
    author_email = 'vanshdhall918@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)