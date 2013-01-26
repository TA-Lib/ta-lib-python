import sys
import unittest

from talib.tests import func_test
from talib.tests import abstract_test

def get_test_cases():
    ret = []
    ret += func_test.get_test_cases()
    ret += abstract_test.get_test_cases()
    return ret

def main():
    suite = unittest.TestSuite()
    suite.addTests(get_test_cases())
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

def profile():
    import cProfile
    import pstats
    profFile = "prof"
    cProfile.run("main()", profFile)
    p = pstats.Stats(profFile)
    # p.dump_stats("runtests.profile")
    p.strip_dirs().sort_stats("time").print_stats()

if __name__ == '__main__':
    main()
    #profile()
