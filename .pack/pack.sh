#!/bin/bash
. .pack/packrc
basedir=$(pwd)

pack=$1
if [ "x$pack" == "x" ];then pack="pack";fi
confile="pack.conf"

if [ $pack = "pack" ];then
    echo "Packing..."
    find $STOREDIR -name "*--*" -type d | xargs rm -rf 
    for file in $(cat $STOREDIR/$confile |grep -v "#")
    do
	echo -e "\tfile $file..."
	fname=$(basename $file)
	dname=$(dirname $file)
	uname=$(echo $dname |sed -e s/\\//_/)
	sdir="$STOREDIR/$uname--$fname"
	mkdir -p "$sdir"
	cd $sdir
	split -b 2000k $basedir/$file $fname-
	cd - &> /dev/null
	git add -f "$STOREDIR/$uname--$fname/"
    done
    find $STOREDIR -name "*--*" -type d | xargs git add -f 
else
    echo "Unpacking..."
    for file in $(cat $STOREDIR/$confile |grep -v "#")
    do
	echo -e "\tUnpacking $file..."
	fname=$(basename $file)
	dname=$(dirname $file)
	uname=$(echo $dname |sed -e s/\\//_/)
	sdir="$STOREDIR/$uname--$fname"
	cat "$sdir"/$fname-* > $dname/$fname
    done
fi
