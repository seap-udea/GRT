echo "Running build wrapper..."
. .pack/packrc
. $REPODIR/reporc

$BUILDWRAPPER-${SYSTEM} --out-dir $REPODIR/build $BUILD

bjson=$REPODIR/build/build-wrapper-dump.json
if [ "x$(grep '\"captures\":' $bjson)" = "x" ]
then
    echo '{"version":0,"captures":[' $(cat $bjson) > $bjson
fi
