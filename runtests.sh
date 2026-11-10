#!/bin/bash
# Runs the unit and functional tests. Exits non-zero if anything fails (the
# ssshtest functional tests only print their failures, so their summary line
# is checked).
status=0

echo "running unit tests:"
python test/unit/samplot_test.py || status=1

for tests in test/func/samplot_test.sh test/func/samplot_vcf_test.sh; do
    echo "running functional tests in $tests:"
    output=$(bash $tests 2>&1)
    echo "$output"
    echo "$output" | sed 's/\x1b\[[0-9;]*m//g' | grep -Eq '^0 +Failures' || status=1
done

exit $status
