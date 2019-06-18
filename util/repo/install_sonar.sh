#!/bin/bash
. .pack/packrc
. $REPODIR/reporc

#Install sonar scanner
if ${SONARSCANNER} --help &> /dev/null
then 
    echo "Sonar scanner is already installed."
else
    #Download sonar-scanner
    if [ ! -f /tmp/sonar-scanner-${SYSTEM}.zip ]
    then
	wget -O /tmp/sonar-scanner-${SYSTEM}.zip $SONARBIN-${SYSTEM}.zip
    else
	echo "Sonar scanner binaries already download."
    fi
    mkdir -p $HOME/src/
    echo "Unzipping sonar-scanner binaries"
    unzip -q -d $HOME/src/ /tmp/sonar-scanner-${SYSTEM}.zip
    mv $HOME/src/sonar-scanner-* $HOME/src/sonar-scanner/
fi

#Install build-wrapper
if ${BUILDWRAPPER}-${SYSTEM} --out-dir /tmp make clean &> /dev/null
then 
    echo "Build-wrapper is already installed."
else
    #Download sonar-scanner
    if [ ! -f $REPODIR/build/build-wrapper-${SYSTEM}.zip ]
    then
	make unpack
    else
	echo "Build-wrapper binaries already unpacked."
    fi
    mkdir -p $HOME/src/
    echo "Unzipping build-wrapper binaries"
    unzip -q -d $HOME/src/ $REPODIR/build/build-wrapper-${SYSTEM}.zip
fi

echo "Done."
