#!/bin/sh

list=$1

for csv in `ls $list`
do
	python3 read.py $list $csv
done
