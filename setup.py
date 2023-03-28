from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this will return the list of requirements.txt
    '''

    hypen_e_dot = '-e .'


    requirements = []
    with open(file_path) as file_:
        requirements = file_.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements

setup(
    name = 'ml_project',
    version = '0.1',
    author = 'har0101',
    author_email = 'helloharsh0101@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')

)