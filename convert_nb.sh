#!/bin/bash
a=`echo $1 | tr '[:upper:]' '[:lower:]'`
rm -rf assets/*${a}_files
cd notebooks
ipython nbconvert --config jekyll.py $1.ipynb
cd ..
mv _posts/*${a}_files assets/