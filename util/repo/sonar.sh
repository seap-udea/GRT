echo "Running sonar scanner..."
. .pack/packrc
. $REPODIR/reporc

$SONARSCANNER \
 $@ \
 -Dsonar.sources=.\
 -Dsonar.exclusions=$REPODIR/**,$REPODIR/**,util/**\
 -Dsonar.host.url=https://sonarcloud.io\
 -Dsonar.python.coverage.reportPaths=$REPODIR/meta/python-coverage.xml\
 -Dsonar.cfamily.gcov.reportsPath=./\
 -Dsonar.sourceEncodings=UTF-8\
 -Dsonar.cfamily.build-wrapper-output=$REPODIR/build\
 -Dproject.settings=$REPODIR/sonarc\
 |tee $REPODIR/sonar.log

