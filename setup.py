import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'antz',
    version = '0.1',
    author = 'Florian Mueller, Bernhard Fuchs',
    author_email = 'jajadinimueter@gmail.com',
    description = ('An ant optimization implementation'),
    license = 'BSD',
    keywords = 'ant colony optimization',
    url = 'git@github.com:jajadinimueter/Antz.git',
    packages=['antz'],
    long_description=read('README.md'),
    install_requires=['pygame'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Utilities',
        'License :: OSI Approved :: BSD License',
    ],
)
