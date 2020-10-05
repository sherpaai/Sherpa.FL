#!/usr/bin/env python
"""
Runs and validates all the notebooks.
LICENSE: Public Domain
"""


import os
import glob
import time
import sys
import traceback

import nbconvert
import nbformat

ep = nbconvert.preprocessors.ExecutePreprocessor(
    extra_arguments=["--log-level=40"],
    timeout=-1,
    kernel_name="FederatedLearning"
)


def run_notebook(path):
    path = os.path.abspath(path)
    assert path.endswith('.ipynb')
    nb = nbformat.read(path, as_version=4)
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
    except Exception as e:
        print("\nException raised while running '{}'\n".format(path))
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


if __name__ == '__main__':
    print('Running notebooks might take a long time...')
    print('===========================================\n')
    for path_tmp in glob.iglob('../**/*.ipynb', recursive=True):
        root, ext = os.path.splitext(os.path.basename(path_tmp))
        if root.endswith('_'):
            continue
        s = time.time()
        sys.stdout.write('Now running ' + path_tmp)
        sys.stdout.flush()
        run_notebook(path_tmp)
        sys.stdout.write(' -- Finish in {}s\n'.format(int(time.time()-s)))


print('\n\033[92m'
      '==========================='
      ' Notebook testing done '
      '==========================='
      '\033[0m')
