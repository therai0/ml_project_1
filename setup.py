from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirement(file_path:str)->List[str]:
    '''
    function will return the list of the requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n"," ") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='ml project',
    version='0.0.1',
    author='raibhaskar',
    author_email='iamraibhaskar@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)