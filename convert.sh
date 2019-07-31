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
	echo "$file1 does not exist."
	echo -1
	return
    fi

    if [ ! -e $file1 ];then
	echo "Target $file2 does not exist, creating."
	echo 1
	return 
    fi

    dif=$(($(date -r $file1 +%s)-$(date -r $file2 +%s)))
    echo $dif
}

for notebook in $@
do
    if ! [[ $notebook == *"$PACKNAME-"* ]];then continue;fi

    devfile=$(basename $notebook)

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
    target=$targetdir/$filename.py

    # Check if notebook is more recent than target file
    if [ $(newer $notebook $target) -lt 0 ];then continue;fi
    echo "Analysing file $devfile:"
    git add $notebook

    echo -e "\tDirectory: $targetdir"
    echo -e "\tFilename: $filename"
    echo -e "\tTarget object: $target"
    
    echo -e "\tConverting from ipynb $notebook to python $target..."
    jupyter nbconvert --to python $notebook --stdout 2> /dev/null | grep -v "# In" | cat -s > /tmp/convert.py 

    echo -e "\tTriming and adding header..."
    nlines=$(cat -n /tmp/convert.py | grep -e "--End--" | cut -f 1 )
    if [ "x$nlines" = "x" ];then nlines=$(cat /tmp/convert.py|wc -l)
    else ((nlines--))
    fi
    (cat header.py;head -n $nlines /tmp/convert.py) > $target 
    git add $target
done
echo "Completed."
