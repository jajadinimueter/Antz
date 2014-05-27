import os
import glob

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup_args = dict(
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
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Utilities',
        'License :: OSI Approved :: BSD License',
    ]
)

try:
    import py2exe
    has_py2exe = True
except ImportError:
    has_py2exe = False

if has_py2exe:
    from distutils.core import setup
else:
    # dont use distutils if we not absolutely have to!
    from setuptools import setup
    setup_args.update(dict(
        install_requires=['pygame', 'pgu'],
    ))

installdatafiles = []

share_target = os.path.join('share', 'pgu', 'themes')

if has_py2exe:
    import matplotlib

    # we have to include datafiles only if we want
    # to build the exe. Otherwise the pgu dependency 
    # should ensure the presence of them
    share = os.path.join('c:\\', 'Python27', 'share', 'pgu', 'themes')
    for name in ('default', 'gray', 'tools'):
        search_path = os.path.join(share, name, '*')
        print(search_path)
        installdatafiles.append((os.path.join(share_target, name),
                                 glob.glob(search_path)))

    installdatafiles.extend(matplotlib.get_py2exe_datafiles())

    print(installdatafiles)

    # some hack I found. when not present, fonts are
    # not found :) somehow such thing were to be expected
    # when trying to create an exe out of python code
    origIsSystemDLL = py2exe.build_exe.isSystemDLL
    def isSystemDLL(pathname):
           if os.path.basename(pathname).lower() in ["sdl_ttf.dll"]:
                   return 0
           return origIsSystemDLL(pathname)
    py2exe.build_exe.isSystemDLL = isSystemDLL

    setup_args.update(dict(
        data_files=installdatafiles,
        console=['antz/main.py'],
        options={
            'py2exe': {
                # 'excludes': ['_gtkagg', '_tkagg'],
                'includes': ["matplotlib.backends.backend_tkagg"],
            }
        }
    ))


## ACTUALLY RUN SETUP
setup(**setup_args)
