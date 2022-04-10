import sys, itertools, math
import numpy as np
import scipy, sys, itertools, math, os, random
from os.path import join, basename
import scipy.sparse
import inspect

import colorama

colorama.init()


class Logger:
    _instance = None

    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        White = "\033[97m"
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        On_Black = "\033[40m"  # Black
        On_Red = "\033[41m"  # Red
        On_Green = "\033[42m"  # Green
        On_Yellow = "\033[43m"  # Yellow
        On_Blue = "\033[44m"  # Blue
        On_Purple = "\033[45m"  # Purple
        On_Cyan = "\033[46m"  # Cyan
        On_White = "\033[47m"  # White
        On_Black = "\033[40m"

    def __init__(self):
        self.new_line = True

    @staticmethod
    def get_instance():
        if Logger._instance is None:
            Logger._instance = Logger()
        return Logger._instance

    @staticmethod
    def log(*args, **kwargs):
        s = ''
        newline = True
        func_stack_back = 0
        color = Logger.bcolors.On_Black + Logger.bcolors.White
        for arg in args:
            s += str(arg)
        for key, value in kwargs.items():
            if key == 'newline':
                newline = value
            elif key == 'func_stack_back':
                func_stack_back = value
            elif key == 'color':
                color = value
            else:
                s += str(value)

        newline_str = Logger.bcolors.ENDC + (
                    str(inspect.stack()[1 + func_stack_back][1]).split('CodeProject')[-1][1:]  # .split('.')[0]
                    + ':' + str(inspect.stack()[1 + func_stack_back][2])).ljust(18) + " | " + str(
            inspect.stack()[1 + func_stack_back][3]).ljust(25) + " | "
        newline_str += color

        s = s.replace('\n', '\n' + newline_str)
        if Logger.get_instance().new_line:
            s = newline_str + s
        if newline:
            s = s + '\n'
        Logger.get_instance().new_line = newline
        # end = '\n' if newline else ''
        print(s, end='')

    @staticmethod
    def log_pass(*args, **kwargs):
        Logger.log(*args, **kwargs, func_stack_back=1, color=Logger.bcolors.On_Green + Logger.bcolors.White)

    @staticmethod
    def log_fail(*args, **kwargs):
        Logger.log(*args, **kwargs, func_stack_back=1, color=Logger.bcolors.On_Red + Logger.bcolors.White)

    @staticmethod
    def get_char():
        Logger.log("", func_stack_back=1, newline=False)
        c = sys.stdin.read(1)
        return c


if __name__ == "__main__":
    Logger.log("Test")
    Logger.log_pass("Pass")
    Logger.log("Test")
    Logger.log_fail("Fail")
