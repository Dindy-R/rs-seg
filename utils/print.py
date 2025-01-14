# -*- encoding: utf-8 -*-
'''
@Description:
'''

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    DEBUG = '\033[31;40m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_info(*args):
    print(bcolors.OKBLUE + " ".join(map(str, args)) + bcolors.ENDC)


def print_success(*args):
    print(bcolors.OKGREEN + " ".join(map(str, args)) + bcolors.ENDC)


def print_failure(*args):
    print(bcolors.FAIL + " ".join(map(str, args)) + bcolors.ENDC)


def print_error(*args):
    print_failure(*args)


def print_warning(*args):
    print(bcolors.WARNING + " ".join(map(str, args)) + bcolors.ENDC)


def print_debug(*args):
    print(bcolors.DEBUG + " ".join(map(str, args)) + bcolors.ENDC)


def print_format_table():
    """
    prints table of formatted text format options
    """
    for style in range(8):
        for fg in range(30, 38):
            s1 = ''
            for bg in range(40, 48):
                format = ';'.join([str(style), str(fg), str(bg)])
                s1 += '\x1b[%sm %s \x1b[0m' % (format, format)
            print(s1)
        print('\n')



if __name__ == '__main__':
    print_format_table()
    print_info("Info")
    print_success("Success")
    print_failure("Failure")
    print_error("ERROR")
    print_warning("WARNING")
    print_debug("Debug")
