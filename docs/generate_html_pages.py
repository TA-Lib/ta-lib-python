"""
USAGE:

To convert markdown docs into html docs:
$ python generate_html_pages.py /path/to/gh-pages/dir

To generate pygments code highlighting stylesheet:
$ pygmentize -f html -S [STYLE_NAME] -a .highlight > /path/to/gh-pages/stylesheets/dir/pygments_style.css

To list available style names (at python prompt)
>>> from pygments import styles
>>> sorted(styles.get_all_styles())
# default, lovelace and xcode are "normal" styles
"""

from __future__ import print_function

import os
import sys
import talib

import mistune
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters.html import HtmlFormatter

from bs4 import BeautifulSoup

from talib.abstract import Function


INPUT_DIR = os.path.dirname(os.path.realpath(__file__))
FUNCTION_GROUPS_DIR = os.path.join(INPUT_DIR, 'func_groups')
OUTPUT_DIR = os.path.join(INPUT_DIR, 'html')


HEADER = '''\
<!DOCTYPE html>
<html>

  <head>
    <meta charset='utf-8' />
    <meta http-equiv="X-UA-Compatible" content="chrome=1" />
    <meta name="description" content="TA-Lib : Python wrapper for TA-Lib (http://ta-lib.org/)." />
    <link rel="stylesheet" type="text/css" media="screen" href="stylesheets/stylesheet.css">
    <title>TA-Lib</title>
  </head>

  <body>
    <div id="header_wrap" class="outer">
        <header class="inner">
            <a id="forkme_banner" href="https://github.com/mrjbq7/ta-lib">View on GitHub</a>
            <div class="clearfix">
                <ul id="menu" class="drop">
                    <li><a href="index.html">Home</a></li>
                    <li><a href="doc_index.html">Documentation</a></li>
                </ul>
            </div>
            <br>
            <h1 id="project_title"><a href="http://mrjbq7.github.io/ta-lib/">TA-Lib</a></h1>
            <h2 id="project_tagline">Python wrapper for TA-Lib (http://ta-lib.org/).</h2>
            <section id="downloads">
                <a class="zip_download_link" href="https://github.com/mrjbq7/ta-lib/zipball/master">Download this project as a .zip file</a>
                <a class="tar_download_link" href="https://github.com/mrjbq7/ta-lib/tarball/master">Download this project as a tar.gz file</a>
            </section>
        </header>
    </div>

    <!-- MAIN CONTENT -->
    <div id="main_content_wrap" class="outer">
        <section id="main_content" class="inner">
'''

FOOTER = '''\
        </section>
    </div>

    <!-- FOOTER  -->
    <div id="footer_wrap" class="outer">
      <footer class="inner">
        <p class="copyright">TA-Lib written by <a href="https://github.com/mrjbq7">mrjbq7</a>
        and <a href="https://github.com/mrjbq7/ta-lib/network/members">contributors</a></p>
        
        <p>Published with <a href="http://pages.github.com">GitHub Pages</a></p>
      </footer>
    </div>

  </body>
</html>
'''


def slugify(string):
    return string.lower().replace(' ', '_')


def get_doc_links():
    """Returns a dictionary of function names -> upstream documentation link"""
    tadoc_homepage = 'http://www.tadoc.org/'
    html_file_path = os.path.join(INPUT_DIR, '.tadoc.org.html')
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r') as f:
            html = f.read()
    else:
        if sys.version_info < (2, 8):
            from urllib2 import urlopen
        else:
            from urllib.request import urlopen

        html = urlopen(tadoc_homepage).read()
        with open(html_file_path, 'w') as f:
            f.write(html)

    # find every link that's for an indicator and convert to absolute urls
    soup = BeautifulSoup(html, 'html.parser')
    links = [a for a in soup.findAll('a') if 'indicator' in a['href']]
    ret = {}
    for a in links:
        url = ''.join([tadoc_homepage, a['href']])
        func = url[url.rfind('/')+1:url.rfind('.')]
        ret[func] = url
    return ret


def generate_groups_markdown():
    """Generate and save markdown files for function group documentation"""
    for group, group_docs in get_groups_markdown().items():
        file_path = os.path.join(FUNCTION_GROUPS_DIR, '%s.md' % group)
        with open(file_path, 'w') as f:
            f.write(group_docs)


def get_groups_markdown():
    """Generate markdown for function groups using the Abstract API

    Returns a dictionary of group_name -> documentation for group functions
    """
    def unpluralize(noun):
        if noun.endswith('s'):
            if len(noun) > 2 and noun[-2] not in ["'", 'e']:
                return noun[:-1]
        return noun

    doc_links = get_doc_links()
    ret = {}
    for group, funcs in talib.get_function_groups().items():
        h1 = '# %s' % unpluralize(group)
        h1 = h1 + ' Functions' if 'Function' not in h1 else h1 + 's'
        group_docs = [h1]
        for func in funcs:
            # figure out this function's options
            f = Function(func)
            inputs = f.info['input_names']
            if 'price' in inputs and 'prices' in inputs:
                names = [inputs['price']]
                names.extend(inputs['prices'])
                input_names = ', '.join(names)
            elif 'prices' in inputs:
                input_names = ', '.join(inputs['prices'])
            else:
                input_names = ', '.join([x for x in inputs.values() if x])

            params = ', '.join(
                ['%s=%i' % (param, default)
                 for param, default in f.info['parameters'].items()])
            outputs = ', '.join(f.info['output_names'])

            # print the header
            group_docs.append('### %s - %s' % (func, f.info['display_name']))

            if f.function_flags and 'Function has an unstable period' in f.function_flags:
                group_docs.append('NOTE: The ``%s`` function has an unstable period.  ' % func)

            # print the code definition block
            group_docs.append("```python")
            if params:
                group_docs.append('%s = %s(%s, %s)' % (
                    outputs, func.upper(), input_names, params))
            else:
                group_docs.append('%s = %s(%s)' % (
                    outputs, func.upper(), input_names))
            group_docs.append("```\n")


            # print extra info if we can
            if func in doc_links:
                group_docs.append('Learn more about the %s at [tadoc.org](%s).  ' % (
                    f.info['display_name'], doc_links[func]))

        group_docs.append('\n[Documentation Index](../doc_index.html)')
        group_docs.append('[FLOAT_RIGHTAll Function Groups](../funcs.html)')

        ret[slugify(group)] = '\n'.join(group_docs) + '\n'
    return ret


def get_markdown_file_paths():
    file_names = [
        'index.md',
        'doc_index.md',
        'install.md',
        'func.md',
        'funcs.md',
        'abstract.md',
    ]
    file_names.extend(
        ['func_groups/%s' % x for x in os.listdir(FUNCTION_GROUPS_DIR) if x.endswith('.md')]
    )
    return [os.path.join(INPUT_DIR, fn) for fn in file_names]


def _get_markdown_renderer():
    """Returns a function to convert a Markdown string into pygments-highlighted HTML"""
    class PygmentsHighlighter(mistune.Renderer):
        def block_code(self, code, lang=None):
            if not lang:
                return '\n<pre><code>%s</code></pre>\n' % mistune.escape(code)
            lexer = get_lexer_by_name(lang, stripall=True)
            formatter = HtmlFormatter(classprefix='highlight ')
            return highlight(code, lexer, formatter)
    return mistune.Markdown(renderer=PygmentsHighlighter())


def run_convert_to_html(output_dir):
    """Converts markdown files into their respective html files"""
    markdown_to_html = _get_markdown_renderer()
    for md_file_path in get_markdown_file_paths():
        with open(md_file_path, 'r') as f:
            html = markdown_to_html(f.read())

        head = HEADER
        if 'func_groups' in md_file_path:
            head = head.replace('"index.html"', '"../index.html"')
            head = head.replace('"doc_index.html"', '"../doc_index.html"')
            head = head.replace('"stylesheets/', '"../stylesheets/')

        lines = html.split('\n')
        for i, line in enumerate(lines):
            if 'FLOAT_RIGHT' in line:
                line = line.replace('FLOAT_RIGHT', '')
                lines[i] = line.replace('<a ', '<a class="float-right" ')
        html = ''.join([head, '\n'.join(lines), FOOTER])

        save_file_path = os.path.abspath(
            md_file_path.replace(INPUT_DIR, output_dir).replace('.md', '.html')
        )
        if not os.path.exists(os.path.dirname(save_file_path)):
            os.mkdir(os.path.dirname(save_file_path))
        with open(save_file_path, 'w') as f:
            f.write(html)
            print('Wrote %s' % save_file_path)


if __name__ == '__main__':
    generate_groups_markdown()
    run_convert_to_html(
        OUTPUT_DIR if len(sys.argv) == 1 else sys.argv[1]
    )
