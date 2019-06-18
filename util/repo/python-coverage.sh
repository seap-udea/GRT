echo "Running tests and coverage analysis..."
. .pack/packrc
. $REPODIR/reporc

rm -rf .coverage &> /dev/null

#Options: https://nose.readthedocs.io/en/latest/man.html
$NOSETESTS \
--with-coverage \
--cover-package=./ \
--cover-xml --cover-xml-file=$REPODIR/meta/python-coverage.xml \
--with-xunit --xunit-file=$REPODIR/meta/python-tests.xml \
--cover-html --cover-html-dir=$REPODIR/meta/python-coverage-html \

sed -i.bak 's/filename="/filename=".\//g' $REPODIR/meta/python-coverage.xml
