import sysconfig
import numpy

get_all = sysconfig.get_config_vars
get     = sysconfig.get_config_var

def print_all():
    _ = tuple(print(f'{k} : {v}') for k, v in get_all().items())

def print_var(var):
    print(get(var))

def get_filtered(*patterns):
    args = get_all()
    if not patterns:
        return args
    print((x for x in patterns))
    return {
        k:v for k, v in args.items() 
            if any((x in k for x in patterns))
    }


def print_filtered(*patterns):
    args = get_filtered(*patterns)
    _ = tuple(print(f'{k} : {v}') for k, v in args.items())


c_compiler    = get("CC")
c_linker      = get("LDSHARED")
c_flags       = get("CFLAGS")

cpp_compiler  = get("CXX")
cpp_linker    = get("LDCXXSHARED")
cpp_flags     = get("CPPFLAGS")

ld_flags      = get("LDFLAGS")

py_include    = get("INCLUDEPY")
numpy_include = numpy.get_include()


if __name__ == '__main__':
    # print_filtered("SHARE", "LD")
    # # print_filtered("C")
    # print_var("CC")
    # print_var("CXX")
    # print_var("INCLUDE")
    print_filtered("FLAGS", "INCLUDE")
    # print_all()
