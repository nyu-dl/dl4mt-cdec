#!/bin/bash
# {1} is the directory name


for f in ${1}/*.xml
do
    cat $f | grep "</seg>" | sed "s/’/'/g" | sed "s/“/\"/g" | sed "s/”/\"/g" > ${f}.fixed
done

