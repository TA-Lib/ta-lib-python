import sys
import unittest

from talib.tests import func_test
from talib.tests import abstract_test

def get_test_cases():
    ret = []
    ret += func_test.get_test_cases()
    ret += abstract_test.get_test_cases()
    return ret

def run():
    suite = unittest.TestSuite()
    suite.addTests(get_test_cases())
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run()
