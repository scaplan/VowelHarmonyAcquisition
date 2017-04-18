#!/bin/bash

inputfile=$1
outputlang=$2

sed 's/,/\t/' $inputfile > tabdelim.txt
sed 's/"//' tabdelim.txt > nofirstquote.txt
sed 's/"//' nofirstquote.txt > nosecondquote.txt
awk ' { t = $1; $1 = $2; $2 = t; print; } ' nosecondquote.txt > wordlist_$2.txt
rm tabdelim.txt nofirstquote.txt nosecondquote.txt