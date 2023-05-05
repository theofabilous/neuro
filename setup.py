
import setuptools
import setuphelper
import confighelper
import os
import subprocess

SANITIZE = False

sanitizer_compile_args = []
sanitizer_link_args = []

if SANITIZE:
    sanitizer_compile_args = [
        '-fsanitize=address',
        '-fno-omit-frame-pointer',
    ]

    sanitizer_link_args = [
        '-fsanitize=address',
        '-shared-libasan'
    ]

neuro = setuptools.Extension(
        'neuro', 
        sources=['neuro.cpp'],
        extra_compile_args=[
            '-std=c++20',
            '-O3',
            '-Wno-deprecated-anon-enum-enum-conversion',
            '-fstrict-aliasing',
            *sanitizer_compile_args,
        ],
        extra_link_args=[
            *sanitizer_link_args,
        ],
        include_dirs=[confighelper.py_include, confighelper.numpy_include],
        )

setuptools.setup(
        name = 'neuro',
        version = '1.0',
        ext_modules = [neuro],
        cmdclass = setuphelper.cpp_build_ext.cmd_class(),
)
