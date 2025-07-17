#!/bin/sh

pandoc --toc=true -f markdown -s -t latex nlp.md -o nlp.tex && latexmk -pdflatex nlp.tex
