import inspect
import talib
import talib.abstract as abstract

HEADER = """
from typing import overload, Tuple, Union
import numpy as np
import pandas as pd
"""

NUMPY = "np.ndarray"
SERIES = "pd.Series"
DATAFRAME = "pd.DataFrame"


def parse_signature(name: str) -> str:
    """
    Return a comma-separated function signature for the given TA-Lib abstract function
    using the latest talib (OrderedDict parameters)
    """
    func = abstract.Function(name)
    func_info = func.info
    params = ["real"]  # always first

    for param_name, default_val in func_info["parameters"].items():
        if default_val is not None:
            params.append(f"{param_name}={repr(default_val)}")
        else:
            params.append(param_name)

    return ", ".join(params)


def output_type(func):
    outputs = func.output_names

    if len(outputs) == 1:
        return NUMPY, SERIES

    tuple_np_type = f"Tuple[{', '.join([NUMPY]*len(outputs))}]"

    return tuple_np_type, DATAFRAME


def clean_doc(doc):
    # Remove redundant first line
    return "\n".join(doc.splitlines()[1:]).strip()


def generate_function(name: str):
    func = abstract.Function(name)

    params = parse_signature(name)
    np_type, pd_type = output_type(func)

    # get talib module docstring (Python-level)
    doc = clean_doc(inspect.getdoc(getattr(talib, name)))

    lines = []

    if doc:
        lines.append(f'"""{doc}"""')

    # Union overload for Series + ndarray (catches both)
    union_params = params.replace("real", "real: Union[pd.Series, np.ndarray]")
    lines.append("@overload")
    lines.append(f"def {name}({union_params}) -> {np_type}: ...")

    # DataFrame overload
    df_params = params.replace("real", "real: pd.DataFrame")
    lines.append("@overload")
    lines.append(f"def {name}({df_params}) -> {pd_type}: ...")

    return "\n".join(lines)


def main():
    print(HEADER)

    for name in talib.get_functions():
        print(generate_function(name))
        print()


if __name__ == "__main__":
    main()
