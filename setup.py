from setuptools import setup

PROJECT_NAME = 'dungeon'

setup(
    name=PROJECT_NAME,
    version='0.1.0',
    packages=[
        'dungeon'
    ],
    url='',
    author='Mathieu G.',
    author_email='mattguillouet@gmail.com',
    description='',
    entry_points={
        "console_scripts": ["{0}={0}.__main__:main".format(
            PROJECT_NAME)]}
)
