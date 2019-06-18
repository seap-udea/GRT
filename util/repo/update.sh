#!/bin/bash
. .pack/packrc
. $REPODIR/reporc


branch=$(cat .branch)
if [ "x$branch" = "x" ];then
    echo "You must choose which branch to update in .branch"
    exit 1
fi

rawuri=https://raw.githubusercontent.com/seap-udea/RepoTemplate/$branch/

qsel=0
if [ "x$1" != "x" ]
then
    qsel=1
    array=("$@")
fi

for file in $(cat $REPODIR/repofiles.list |grep -v "#")
do
    if [ $file = "$REPODIR/reporc" -o $file = "$REPODIR/sonarc" -o $file = ".pack/packrc" ]
    then 
	echo "Skipping $file"
    fi

    if [ $qsel -gt 0 ]
    then
	if [[ " ${array[@]} " =~ " ${file} " ]];then
	    :
	else
	    continue
	fi
    fi
    echo -n "Downloading $file: "
    fname=/tmp/$(basename $file)
    curl -s -o $fname $rawuri/$file
    if [ ! -e $file ];then
	echo -n "Local file does not exist, retrieving. "
	cp -rf $fname $file
    elif [ "x$(diff -q -w $fname $file)" != "x" ];then 
	echo -n "local file is different. are you sure?(y/n)[y]:"
	read ans
	if [ "x$ans" = "x" -o "$ans" = "y" ];then 
	    echo -n "retrieving. "
	    cp -rf $fname $file
	else
	    echo -n "skipping. "
	fi
    else
	echo -n "local file is the same, skipping. "
    fi
    echo "done."
done
