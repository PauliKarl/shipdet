import os
import platform
import subprocess
import time
import numpy as np
from setuptools import find_packages, setup, Extension


from Cython.Build import cythonize
#from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_git_hash():

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha
setup(name='shipdet',
      version='0.1',
      description='shipdet gaofen',
      classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
      url='https://www.python.org/',
      author='paulikarl',
      author_email='paulikarlcn@gmail.com',
      license='paulikarl',
      packages=find_packages(exclude=["build"]),
      zip_safe=True
     )