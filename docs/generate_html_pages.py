import os
import sys
import talib

from grip import render_page
from bs4 import BeautifulSoup

from talib.abstract import Function


DIR = os.path.dirname(os.path.realpath(__file__))

header = '''\
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
            <a id="forkme_banner" href="https://github.com/mrjbq7/ta-lib">Browse TA-Lib on GitHub</a>
            <div class="clearfix">
                <ul id="menu" class="drop">
                    <li><a href="index.html">Home</a></li>
                    <li><a href="doc_index.html">Documentation</a></li>
                </ul>
            </div>
            <br>
            <h1 id="project_title">TA-Lib</h1>
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

footer = '''\
        </section>
    </div>

    <!-- FOOTER  -->
    <div id="footer_wrap" class="outer">
      <footer class="inner">
        <p class="copyright">TA-Lib written by <a href="https://github.com/mrjbq7">mrjbq7</a>
        with contributions by <a href="https://github.com/briancappello">briancappello</a>
        and <a href="https://github.com/mrjbq7/ta-lib/network/members">others</a></p>
        
        <p>Published with <a href="http://pages.github.com">GitHub Pages</a></p>
      </footer>
    </div>

  </body>
</html>
'''


def get_doc_links(update=False):
    tadoc_homepage = 'http://www.tadoc.org/'

    # if not update load a cached copy if we can, otherwise download new html
    html_file_path = os.path.join(DIR, 'tadoc.org.html')
    if not update and os.path.exists(html_file_path):
        with open(html_file_path, 'r') as f:
            html = f.read()
    else:
        import urllib2
        html = urllib2.urlopen(tadoc_homepage).read()
        with open(html_file_path, 'w') as f:
            f.write(html)

    # find every link that's for an indicator and convert to absolute urls
    soup = BeautifulSoup(html)
    links = [a for a in soup.findAll('a') if 'indicator' in a['href']]
    ret = {}
    for a in links:
        url = ''.join([tadoc_homepage, a['href']])
        func = url[url.rfind('/')+1:url.rfind('.')]
        ret[func] = url
    return ret

def get_groups_markdown(update=False):

    def unpluralize(noun):
        if noun.endswith('s'):
            if len(noun) > 2 and noun[-2] not in ["'", 'e']:
                return noun[:-1]
        return noun

    doc_links = get_doc_links(update)
    ret = {}
    for group, funcs in talib.get_function_groups().items():
        h1 = '# %s' % unpluralize(group)
        h1 = h1 + ' Functions' if 'Function' not in h1 else h1 + 's'
        ret[group] = [h1]
        for func in funcs:
            # figure out this function's options
            f = Function(func)
            inputs = f.info['input_names']
            if 'prices' in inputs:
                input_names = ', '.join(inputs['prices'])
            else:
                input_names = ', '.join([x for x in inputs.values() if x])

            params = ', '.join(
                ['%s=%i' % (param, default)
                 for param, default in f.info['parameters'].items()])
            outputs = ', '.join(f.info['output_names'])

            # print the header
            ret[group].append('### %s - %s' % (func, f.info['display_name']))

            # print the code definition block
            ret[group].append("```")
            if params:
                ret[group].append('%s = %s(%s, %s)' % (
                    outputs, func.upper(), input_names, params))
            else:
                ret[group].append('%s = %s(%s)' % (
                    outputs, func.upper(), input_names))
            ret[group].append("```\n")

            # print extra info if we can
            if func in doc_links:
                ret[group].append(
                    'Learn more about the %s at [tadoc.org](%s).  ' % (
                        f.info['display_name'], doc_links[func]))
        ret[group].append('\n[Documentation Index](../doc_index.html)')
        ret[group].append('[FLOAT_RIGHTAll Function Groups](../funcs.html)')
        ret[group] = '\n'.join(ret[group]) + '\n'
    return ret

def save_group_files(markdown_groups):
    group_order = [
        'Overlap Studies',
        'Momentum Indicators',
        'Volume Indicators',
        'Volatility Indicators',
        'Pattern Recognition',
        'Cycle Indicators',
        'Statistic Functions',
        'Price Transform',
        'Math Transform',
        'Math Operators',
        ]

    def slug(name):
        return name.lower().replace(' ', '_')

    for group in group_order:
        file_path = os.path.join(DIR, 'func_groups', '%s.md' % slug(group))
        with open(file_path, 'w') as f:
            f.write(markdown_groups[group])

def generate_function_groups_md(update=False):
    save_group_files(get_groups_markdown(update))

def get_open_save_file_paths():
    ret = []
    file_names = [
        'index.md',
        'doc_index.md',
        'install.md',
        'func.md',
        'funcs.md',
        'abstract.md',
        ]
    groups = ['func_groups/%s' % x
              for x in os.listdir(os.path.join(DIR, 'func_groups'))]
    file_names.extend(groups)
    for file_name in file_names:
        open_file_path = os.path.join(DIR, file_name)
        save_file_path = os.path.join(DIR, 'html', file_name.replace('.md', '.html'))
        ret.append((open_file_path, save_file_path))
    return ret

def run_convert_to_html():
    for md_file_path, save_file_path in get_open_save_file_paths():
        with open(md_file_path, 'r') as f:
            html = render_page(f.read(), os.path.split(md_file_path)[1])

        html = html.replace('<pre><code>', '<pre>')
        html = html.replace('</code></pre>', '</pre>')
        html = html[html.find('<body>')+len('<body>'):] # chop off the generated header
        html = html[:html.rfind('</body>')] # chop off </body></html>

        head = header
        if 'func_groups' in save_file_path:
            head = head.replace('"index.html"', '"../index.html"')
            head = head.replace('"doc_index.html"', '"../doc_index.html"')
            head = head.replace('"stylesheets/', '"../stylesheets/')

        lines = html.split('\n')
        for i, line in enumerate(lines):
            if 'FLOAT_RIGHT' in line:
                line = line.replace('FLOAT_RIGHT', '')
                lines[i] = line.replace('<a ', '<a class="float-right" ')
        html = ''.join([head, '\n'.join(lines), footer])

        with open(save_file_path, 'w') as f:
            f.write(html)

if __name__ == '__main__':
    generate_function_groups_md(update=False)
    run_convert_to_html()
