#!/bin/bash
. .pack/packrc
. $REPODIR/reporc

make cleancrap
find . -type f \
    |grep -v "./.git/" \
    |grep -E "util/repo/|util/sonar/|./.pack" \
    |sed -e "s/.\///" \
    |tee $REPODIR/files.list |tee $REPODIR/repofiles.list

