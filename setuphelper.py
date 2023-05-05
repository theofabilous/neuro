import setuptools
from setuptools.command.build_ext import build_ext
# from pybind11.setup_helpers import Pybind11Extension
import confighelper as cfg

'''
https://pybind11.readthedocs.io/en/stable/compiling.html#setup-helpers-pep518
https://docs.python.org/3/library/sysconfig.html
https://people.duke.edu/~ccc14/sta-663-2016/18G_C++_Python_pybind11.html

python setup.py build_ext -i
'''


class c_build_ext(build_ext):
    def build_extensions(self):
        self.compiler.set_executable("compiler_so", cfg.c_compiler)
        self.compiler.set_executable("compiler", cfg.c_compiler)
        self.compiler.set_executable("compiler_cxx", cfg.cpp_compiler)
        self.compiler.set_executable("linker_so", cfg.c_linker)
        build_ext.build_extensions(self)

    @classmethod
    def cmd_class(cls):
        return {'build_ext': cls}

class cpp_build_ext(build_ext):
    def build_extensions(self):
        self.compiler.set_executable("compiler_so", cfg.c_compiler)
        self.compiler.set_executable("compiler", cfg.c_compiler)
        self.compiler.set_executable("compiler_cxx", cfg.cpp_compiler)
        self.compiler.set_executable("linker_so", cfg.cpp_linker)
        build_ext.build_extensions(self)

    @classmethod
    def cmd_class(cls):
        return {'build_ext': cls}

'''
>> Example:

my_module = setuptools.Extension('<name>', sources=['main.cpp'])
setuptools.setup(
        name = '<name>',
        version = '1.0',
        ext_modules = [my_module],
        cmdclass = cpp_build_ext.cmd_class(),
)
'''




