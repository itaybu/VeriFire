#!/bin/bash
curdir=$pwd
mydir="${0%/*}"
version=$1

cd $mydir

# TODO: add progress bar, -q is quite, if removing it the progress bar is in
# multiple lines
#echo "downloading pybind"
#wget https://github.com/pybind/pybind11/archive/v$version.tar.gz -O pybind11-$version.tar.gz  -q --show-progress --progress=bar:force:noscroll
echo "unzipping pybind"
tar xzvf pybind11-$version.tar.gz >> /dev/null

cd $curdir
