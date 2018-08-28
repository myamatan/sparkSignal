#!/bin/sh

# Usage
#./print.sh ./data/20180710/all.c300a900.2h/

list=$1
for csv in `ls $list`
do
	python3 read.py $list $csv
done
