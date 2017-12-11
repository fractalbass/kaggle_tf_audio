from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras>=2.1.2', 'scikit-learn>=0.19.1', 'urllib3>=1.22', 'h5py>=2.7.1', 'google-cloud-storage>=1.6.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)
