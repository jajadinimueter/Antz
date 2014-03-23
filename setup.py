import os

installdatafiles = []

try:
    import py2exe
    share_target = os.path.join('share','pgu','themes')
    share = os.path.join(os.path.dirname(py2exe.__file__), '..', '..', '..', 'share', 'pgu', 'themes')
    installdatafiles = []
    for name in ('default', 'gray', 'tools'):
        installdatafiles.append((os.path.join(share_target, name),
                                 glob.glob(os.path.join(share, name, '*'))))
    origIsSystemDLL = py2exe.build_exe.isSystemDLL
    def isSystemDLL(pathname):
           if os.path.basename(pathname).lower() in ["sdl_ttf.dll"]:
                   return 0
           return origIsSystemDLL(pathname)
    py2exe.build_exe.isSystemDLL = isSystemDLL
except ImportError:
    pass
import glob
from distutils.core import setup

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
    install_requires=['pygame', 'pgu', 'py2exe'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Utilities',
        'License :: OSI Approved :: BSD License',
    ],
    data_files=installdatafiles,
	console=['antz/gui.py']
)
