import talib
from talib.abstract import Function

def run():
    for group, funcs in talib.get_function_groups().items():
	print
	print '## %s Functions' % group
	print "```"
	for func in funcs:
	    f = Function(func)
	#    print f.info
	    inputs = f.info['input_names']
	    if 'prices' in inputs:
		in_ = ', '.join(inputs['prices'])
	    else:
		in_ = ', '.join([x for x in inputs.values() if x])
		if not in_:
		    print inputs
	    out = ', '.join(f.info['output_names'])
	    def_ = ', '.join(['%s=%i' % (param, default) for param, default in f.info['parameters'].items()])
	    
	    if def_:
		print '%s = %s(%s, %s)' % (out, func.upper(), in_, def_)
	    else:
		print '%s = %s(%s)' % (out, func.upper(), in_)
	print "```"

if __name__ == '__main__':
    run()
