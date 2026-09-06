import ast
import os

import talib

STUBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'talib')


def _stub(name):
    with open(os.path.join(STUBS, name)) as f:
        return f.read()


# The stubs ship as package data beside py.typed, and nothing else in the build
# reads them, so a broken one reaches every downstream type checker in silence.
def test_stubs_parse():
    for name in ('_ta_lib.pyi', 'abstract.pyi', 'stream.pyi'):
        ast.parse(_stub(name), filename=name)


def test_stubs_cover_every_function():
    declared = {node.name for node in ast.parse(_stub('_ta_lib.pyi')).body
                if isinstance(node, ast.FunctionDef)}
    handles = {node.name for node in ast.parse(_stub('stream.pyi')).body
               if isinstance(node, ast.ClassDef)}
    for name in talib.__TA_FUNCTION_NAMES__:
        assert name in declared, name
        assert name in handles, name
