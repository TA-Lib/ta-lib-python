import os
import sys

from grip import render_page


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
        <p class="copyright">TA-Lib written by <a href="https://github.com/mrjbq7">mrjbq7</a></p>
        <p>Published with <a href="http://pages.github.com">GitHub Pages</a></p>
      </footer>
    </div>

  </body>
</html>
'''

def run():
    path = os.path.dirname(os.path.realpath(__file__))
    file_names = [
        'index.md',
        'doc_index.md',
        'install.md',
        'func.md',
        'funcs.md',
        'abstract.md',
        ]
    for file_name in file_names:
        with open(os.path.join(path, file_name), 'r') as f:
            html = render_page(f.read(), file_name)
        html = html.replace('<pre><code>', '<pre>')
        html = html.replace('</code></pre>', '</pre>')
        html = html[html.find('<body>')+len('<body>'):] # chop off the generated header
        html = html[:html.rfind('</body>')] # chop off </body></html>

        html = ''.join([header, html, footer])
        save_file_path = os.path.join(path, 'html', file_name.replace('.md', '.html'))
        with open(save_file_path, 'w') as f:
            f.write(html)

if __name__ == '__main__':
    run()
