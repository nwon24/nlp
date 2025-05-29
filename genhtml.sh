#!/bin/sh

pandoc -f markdown -s --mathjax -t html nlp.md -o nlp.html
