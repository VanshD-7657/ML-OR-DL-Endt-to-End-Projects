from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """ This function will return the list of requirements"""

    requirements_list: List[str] = []
    try:
        with open('requirements.txt','r') as file:
            # Read the requirements from the file
            lines = file.readlines()
            # Process each line
            for line in lines:
                requirement = line.strip()
                # ignore empty lines and -e.
                if requirement and requirement != '-e .':
                    requirements_list.append(requirement)

    except FileNotFoundError:
        print("The requirements.txt file was not found.")

    return requirements_list

setup(
    name='NetworkSecurity',
    version='0.0.1',
    author='vansh',
    author_email='vdhall340@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)
