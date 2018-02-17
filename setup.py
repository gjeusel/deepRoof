from setuptools import setup, find_packages

setup(
    name='deeproof',
    version='0.0.1',
    url='https://github.com/gjeusel/deepRoof.git',
    author='Guillaume Jeusel',
    description='Engie datascience challenge for rooftop orientation classification',

    # https://stackoverflow.com/questions/14417236/setup-py-renaming-src-package-to-project-name
    packages=['deeproof'],
    package_dir={'': 'src'},
    # install_requires = ['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)
