#################################################################################
#     ______                 ____             ___                               # 
#    / ____/________ __   __/ __ \____ ___  _|__ \                              #
#   / / __/ ___/ __ `/ | / / /_/ / __ `/ / / /_/ /                              #
#  / /_/ / /  / /_/ /| |/ / _, _/ /_/ / /_/ / __/                               #
#  \____/_/   \__,_/ |___/_/ |_|\__,_/\__, /____/                               #
#                                    /____/                                     #
#  .-. .-. . . .-. .   .-. .-. .  . .-. . . .-.                                 #
#  |  )|-  | | |-  |   | | |-' |\/| |-  |\|  |                                  #
#  `-' `-' `.' `-' `-' `-' '   '  ` `-' ' `  '                                  #
#################################################################################
# Jorge I. Zuluaga (C) 2019                                                     #
#################################################################################
#!/bin/bash
. .pack/packrc

if [ "x$1" = "x" ];then 
    echo "You must provide at least one .ipynb file"
fi

function newer()
{
    file1=$1;shift
    file2=$1;shift
    if [ ! -e $file1 ];then
	echo -1
	return
    fi

    if [ ! -e $file2 ];then
	echo 1
	return 
    fi

    dif=$(($(date -r $file1 +%s)-$(date -r $file2 +%s)))
    echo $dif
    return
}

function convert()
{
    notebook=$1;shift
    target=$1;shift
    
    echo -e "\tConverting from ipynb $notebook to python $target..."
    jupyter nbconvert --to python $notebook --stdout 2> /dev/null | grep -v "# In" | cat -s > /tmp/convert.py 

    echo -e "\tTriming..."
    nlines=$(cat -n /tmp/convert.py | grep -e "--End--" | cut -f 1 )
    if [ "x$nlines" = "x" ];then nlines=$(cat /tmp/convert.py|wc -l)
    else ((nlines--))
    fi

    echo -e "\tProcessing magic commands..."
    sed -ie "s/get_ipython().magic('timeit\(.*\))$/get_ipython().magic('timeit\1,scope=globals())/" /tmp/convert.py

    echo -e "\tAdding header..."
    (cat header.py;head -n $nlines /tmp/convert.py) > $target 
}

for notebook in $@
do
    if [ ! -e $notebook ];then 
	echo "Notebook $notebook does not exist. Skipping."
	continue
    fi

    devfile=$(basename $notebook)
    devdir=$(dirname $notebook)

    # Parse script name
    IFS="-"
    targetdir="."
    for dir in $devfile
    do
	if [ -d $targetdir/$dir ];then
	    targetdir="$targetdir/$dir"
	else
	    filename=$dir
	fi
    done
    IFS=" "
    filename=$(echo $filename |awk -F'.' '{print $1}')

    if ! [[ $notebook == *"$PACKNAME-"* ]]
    then 
	target=$devdir/$filename.py
    else
	target=$targetdir/$filename.py
    fi

    # Check if notebook is more recent than target file
    if [ $1 != "force" ];then 
	if [ $(newer $notebook $target) -lt 0 ];then continue;fi
    fi

    echo "Analysing file $devfile:"
    git add -f $notebook

    echo -e "\tDirectory: $targetdir"
    echo -e "\tFilename: $filename"
    echo -e "\tTarget object: $target"

    convert $notebook $target

    if [[ $notebook == *"$PACKNAME-"* ]]
    then
	git add -f $target
    fi
done
echo "Completed."
