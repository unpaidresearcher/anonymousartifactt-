#!/bin/bash
# cover_repo.sh <repo_dir> <python_bin> <source_pkg> <test_target> <timeout>
REPO=$1; PYBIN=$2; SRC=$3; TESTS=$4; TO=${5:-3600}; NAME=$(basename $REPO)
M=/data/home/hrish/kg-adapter/KG-Adapter/moa
cd $REPO
echo "[$NAME] coverage start $(date)"
PYTHONPATH=$REPO timeout $TO $PYBIN -m coverage run --source=$REPO/$SRC -m pytest $TESTS -q -p no:cacheprovider --no-header >$M/covrun_$NAME.log 2>&1
echo "[$NAME] coverage run exit=$? $(date)"
$PYBIN -m coverage json -o $M/cov_$NAME.json 2>/dev/null && echo "[$NAME] json ok"
cd $M
python3 -c "
import json
try: cov=json.load(open('cov_$NAME.json'))['files']
except Exception as e: print('[$NAME] no cov json:',e); raise SystemExit
items=[x for x in json.load(open('verif_benchmark_final.json')) if x['repo']=='$NAME']
covered=[]
for it in items:
    ks=[k for k in cov if k.endswith(it['file'])]
    if ks and (set(cov[ks[0]]['executed_lines']) & set(range(it['start'],it['end']+1))): covered.append(it)
json.dump(covered, open('covered_$NAME.json','w'), indent=1)
print('[$NAME] items',len(items),'covered',len(covered))
"
echo "COVER_DONE_$NAME"
