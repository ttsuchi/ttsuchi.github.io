#!/bin/bash
BLOG_DIR=`pwd` ipython nbconvert --config jekyll.py notebooks/$1.ipynb
rsync --remove-source-files assets/*.md _posts/
