#!/bin/sh

cat *csv | split -l $1
for f in x*; do sed -i '1ilabel,title,text' $f; done
