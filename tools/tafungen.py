from talib import abstract, get_functions


def generate_code(filename: str = None):
    file = None
    if filename is not None:
        file = open(filename, "w")
        file.write("from talib import abstract\n\n\n")

    for fun_name in sorted(get_functions()):
        fun = abstract.Function(fun_name)
        print("\n")
        function_code = generate_function_code(fun.__dict__)
        print(function_code)
        if file:
            file.write(function_code)
            file.write("\n\n\n")

    if file:
        file.close()


def get_type_name(i: int):
    return {0: "float",
            1: "str",
            2: "int",
            3: "MA_TYPE"
            }[i]


t = "    "


def generate_function_code(d: dict):
    s_def = f"def {d['_Function__info']['name']}(inputs"

    # list_params = []
    for k, v in d['_Function__info']["parameters"].items():
        # if type(v) == list:
        #     list_params.append(k)
        #     s_def += f", {k}=None"
        # else:
        s_def += f", {k}={v}"

    # list_params2 = []
    for k, v in d['_Function__info']["input_names"].items():
        if type(v) == str:
            s_def += f", {k}='{v}'"
        # elif type(v) == list:
        #     s_def += f", {k}=None"
        #     list_params2.append(k)
        else:
            s_def += f", {k}={v}"

    s_def += "):"

    s_def += f'''
    """
    **{d['_Function__info']['display_name']}**

    Group: {d['_Function__info']['group']}
    '''

    if d['_Function__info']['function_flags'] is not None:
        s_def += f"\n{t}Function flags: {'. '.join(d['_Function__info']['function_flags'])}\n"

    s_def += f"\n{t}:param inputs: input values"

    for k, v in d['_Function__input_names'].items():
        name = d['_Function__input_names'][k]['name']
        default_value = d['_Function__input_names'][k]['price_series']
        if type(default_value) == str:
            default_value = f"'{default_value}'"
        s_def += f"\n{t}:param {k}: {name} (**default**: {default_value})"

    for k, v in d['_Function__opt_inputs'].items():
        definition = d['_Function__opt_inputs'][k]['display_name']
        help_text = d['_Function__opt_inputs'][k]['help']
        default_value = d['_Function__opt_inputs'][k]['default_value']
        type_name = get_type_name(d['_Function__opt_inputs'][k]['type'])
        s_def += f"\n{t}:param {k}: {definition} ({help_text} - **default**: {default_value})"
        s_def += f"\n{t}:type {k}: {type_name}"

    if len(d["_Function__info"]["output_names"]) > 1:
        s_def += f'\n{t}:rtype: Tuple'
        s_def += f'\n{t}:returns: {tuple(d["_Function__info"]["output_names"])}'
    else:
        s_def += f'\n{t}:returns: {d["_Function__info"]["output_names"][0]}'

    s_def += f'\n{t}"""'

    # for lp in list_params:
    #     s_def += f"\n{t}if {lp} is None:"
    #     s_def += f"\n{t}{t}{lp} = {d['_Function__info']['parameters'][lp]}"
    #
    # for lp in list_params2:
    #     s_def += f"\n{t}if {lp} is None:"
    #     s_def += f"\n{t}{t}{lp} = {d['_Function__info']['input_names'][lp]}"

    s_def += f"\n{t}return abstract.Function('{d['_Function__info']['name']}'"

    for k, v in d['_Function__info']["parameters"].items():
        s_def += f", {k}={k}"

    for k, v in d['_Function__info']["input_names"].items():
        s_def += f", {k}={k}"

    s_def += ")(inputs)"

    return s_def


if __name__ == '__main__':
    generate_code("tafun.py")
