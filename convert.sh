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

for notebook in $@
do
    if ! [[ $notebook == *"$PACKNAME-"* ]];then continue;fi

    devfile=$(basename $notebook)

    echo "Analysing file $devfile:"
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
done
echo "Completed."
