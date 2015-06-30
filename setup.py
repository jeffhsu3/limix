#change CC to include -std=c++0x flags.
#this is a hack as distutils does not permit specifying seperate build flags for .c and .cpp files
import os
#os.environ['CC'] = 'gcc -std=c++0x'

import distutils.cmd
from setuptools import find_packages
import sys,os,re
from distutils.core import Extension
from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.errors import CompileError

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import pdb
import glob

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def file_list_recursive(dir_name,exclude_list=[],ext=[]):
    """create a recursive file list"""
    FL = []
    for root, dirs, files in os.walk(dir_name):
        FL_ = [os.path.join(root,fn) for fn in files]
        #filter and append
        for fn in FL_:
            if not any([ex in fn for ex in exclude_list]):
                if (os.path.splitext(fn)[1] in ext):
                    FL.append(fn)
    return FL

def strip_rc(version):
    return re.sub(r"rc\d+$", "", version)

def check_versions(min_versions):
    """
    Check versions of dependency packages
    """
    from distutils.version import StrictVersion

    try:
        import scipy
        spversion = scipy.__version__
    except ImportError:
        raise ImportError("LIMIX requires scipy")

    try:
        import numpy
        npversion = numpy.__version__
    except ImportError:
        raise ImportError("LIMIX requires numpy")

    try:
        import pandas
        pandasversion = pandas.__version__
    except ImportError:
        raise ImportError("LIMIX requires pandas")
    #match version numbers
    try:
        assert StrictVersion(strip_rc(npversion)) >= min_versions['numpy']
    except AssertionError:
        raise ImportError("Numpy version is %s. Requires >= %s" %
                (npversion, min_versions['numpy']))
    try:
        assert StrictVersion(strip_rc(spversion)) >= min_versions['scipy']
    except AssertionError:
        raise ImportError("Scipy version is %s. Requires >= %s" %
                (spversion, min_versions['scipy']))
    try:
        assert StrictVersion(strip_rc(pandasversion)) >= min_versions['pandas']
    except AssertionError:
        raise ImportError("pandas version is %s. Requires >= %s" %
                (pandasversion, min_versions['pandas']))

def get_source_files(reswig=True):
    """build list of source files. swig=True means the interfaces is 'reswigged'. Otherwise the distribution
    version of the numpy python wrappers are retained"""
    FL = []
    #nlopt sources files
    nlopt=['direct/DIRect.cpp',
        'direct/direct_wrap.cpp',
        'direct/DIRserial.cpp',
        'direct/DIRsubrout.cpp',
        'cdirect/cdirect.cpp','cdirect/hybrid.cpp',
        'praxis/praxis.cpp','luksan/plis.cpp','luksan/plip.cpp','luksan/pnet.cpp', 'luksan/mssubs.cpp','luksan/pssubs.cpp',
        'crs/crs.cpp',
        'mlsl/mlsl.cpp',
        'mma/mma.cpp','mma/ccsa_quadratic.cpp',
        'cobyla/cobyla.cpp',
        'newuoa/newuoa.cpp',
        'neldermead/nldrmd.cpp','neldermead/sbplx.cpp',
        'auglag/auglag.cpp',
        'esch/esch.cpp',
        'bobyqa/bobyqa.cpp',
        'isres/isres.cpp',
        'slsqp/slsqp.cpp',
        'api/general.cpp','api/options.cpp','api/optimize.cpp','api/deprecated.cpp','api/f77api.cpp',
        'util/mt19937ar.cpp','util/sobolseq.cpp','util/timer.cpp','util/stop.cpp','util/redblack.cpp','util/qsort_r.cpp','util/rescale.cpp',
        'stogo/global.cc','stogo/linalg.cc','stogo/local.cc','stogo/stogo.cc','stogo/tools.cc'
        ]
    #limix sourcs files
    #python wrapper
    FL.extend(file_list_recursive('./src',exclude_list=['src/archive','src/testing','src/interfaces'],ext=['.cpp','.c']))
    #nlopt
    nlopt = ['./External/nlopt/%s' % fn for fn in nlopt]
    #add header files
    if reswig:
        FL.extend(['src/interfaces/python/limix.i'])
    else:
        pass
        FL.extend(['src/interfaces/python/limix_wrap.cpp'])
    FL.extend(nlopt)
    return FL

def get_include_dirs():
    include_dirs = ['src']
    include_dirs.extend(['External','External/nlopt'])
    nlopt_include_dir = ['stogo','util','direct','cdirect','praxis','luksan','crs','mlsl','mma','cobyla','newuoa','neldermead','auglag','bobyqa','isres','slsqp','api','esch']
    nlopt_include_dir = ['./External/nlopt/%s' % fn for fn in nlopt_include_dir]
    include_dirs.extend(nlopt_include_dir)
    #add numpy include dir
    numpy_inc_path = [numpy.get_include()]
    include_dirs.extend(numpy_inc_path)
    return include_dirs

def get_swig_opts():
    swig_opts=['-c++', '-Isrc','-outdir','limix/deprecated']
    return swig_opts

def get_extra_compile_args():
    # return ['-std=c++0x', '-stdlib=libc++']
    return []

def try_to_add_compile_args():
    return ['-std=c++0x', '-stdlib=libc++']

import numpy

class CustomBuild(build):
    sub_commands = [
        ('build_ext', build.has_ext_modules),
        ('build_py', build.has_pure_modules),
        ('build_clib', build.has_c_libraries),
        ('build_scripts', build.has_scripts),
    ]

class CustomBuildExt(build_ext):
    def build_extensions(self):
        import tempfile

        flags = try_to_add_compile_args()

        f = tempfile.NamedTemporaryFile(suffix=".cpp", delete=True)
        f.name
        c = self.compiler

        ok_flags = []

        for flag in flags:
            try:
                c.compile([f.name], extra_postargs=ok_flags+[flag])
            except CompileError:
                pass
            else:
                ok_flags.append(flag)

        for ext in self.extensions:
            ext.extra_compile_args += ok_flags

        f.close()
        build_ext.build_extensions(self)

reswig = False
if '--reswig' in sys.argv:
    index = sys.argv.index('--reswig')
    sys.argv.pop(index)  # Removes the '--foo'
    reswig = True

#1. find packages (parses the local 'limix' tree')
# exclude limix.deprecated. This is a placeholder and will be replaced with the
# actual deprecated limix source tree
#packages = find_packages(exclude=['limix.deprecated'])
packages = find_packages(exclude=['tests', 'test', 'test_limix*',
                                  'limix.modules2*'])
#3. add depcreated limix packages in src/interfaces/python (see below)
#packages.extend(['limix.deprecated', 'limix.deprecated.io',
#                 'limix.deprecated.modules', 'limix.deprecated.stats',
#                 'limix.deprecated.utils'])
reqs = ['numpy', 'scipy', 'matplotlib >=1.2']

FL = get_source_files(reswig=reswig)

#fore deployment version to be the current MAC release and not what is stored in distutils
#this is key for some distributions like anaconda, which otherwise build for an outdated target.
from sys import platform as _platform
if _platform == 'darwin':
    from distutils import sysconfig
    import platform
    sysconfig._config_vars['MACOSX_DEPLOYMENT_TARGET'] = platform.mac_ver()[0]


def get_test_suite():
    from unittest import TestLoader
    from unittest import TestSuite
    test_suite1 = TestLoader().discover('limix')
    test_suite2 = TestLoader().discover('test_limix')
    return TestSuite([test_suite1, test_suite2])


#create setup:
setup(
    name = 'limix',
    version = '0.7.6',
    cmdclass={'build': CustomBuild, 'build_ext': CustomBuildExt},
    author = 'Christoph Lippert, Paolo Casale, Oliver Stegle',
    author_email = "stegle@ebi.ac.uk",
    description = ('A flexible and fast mixed model toolbox written in C++/python'),
    url = "http://",
    long_description = read('README'),
    license = 'BSD',
    keywords = 'linear mixed models, GWAS, QTL, Variance component modelling',
    ext_package = 'limix.deprecated',
    ext_modules = [Extension('_core',get_source_files(reswig=reswig),include_dirs=get_include_dirs(),swig_opts=get_swig_opts(),extra_compile_args = get_extra_compile_args())],
    py_modules = ['limix.deprecated.core'],
    scripts = ['scripts/limix_runner','scripts/mtSet_postprocess','scripts/mtSet_preprocess','scripts/mtSet_simPheno','scripts/mtSet_analyze'],
    packages = packages,
    package_dir = {'limix': 'limix'},
    #dependencies
    #requires = ['scipy','numpy','matplotlib','pandas','scons'],
    requires=map(lambda x: x.split(" ")[0], reqs),
    install_requires = reqs,
    test_suite='setup.get_test_suite'
    )
