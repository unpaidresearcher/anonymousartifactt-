PY38=~/miniconda3/envs/bipy_py38/bin/python
R=/data/home/hrish/kg-adapter/swebench_repos/sympy__sympy
M=/data/home/hrish/kg-adapter/KG-Adapter/moa
cd $R
$PY38 -m coverage erase
for d in $(cat $M/sympy_testdirs.txt); do
  PYTHONPATH=$R timeout 150 $PY38 -m coverage run --append --source=$R/sympy -m pytest $d -q -p no:cacheprovider >/dev/null 2>&1
  echo "[sympy] covered $d (exit $?)"
done
$PY38 -m coverage json -o $M/cov_sympy__sympy.json 2>/dev/null && echo "json ok"
cd $M
python3 -c "
import json
cov=json.load(open('cov_sympy__sympy.json'))['files']
its=[x for x in json.load(open('verif_benchmark_final.json')) if x['repo']=='sympy__sympy']
covered=[]
for it in its:
    ks=[k for k in cov if k.endswith(it['file'])]
    if ks and (set(cov[ks[0]]['executed_lines']) & set(range(it['start'],it['end']+1))): covered.append(it)
json.dump(covered,open('covered_sympy__sympy.json','w'),indent=1)
print('[sympy] items',len(its),'covered',len(covered))
"
echo "COVER_DONE_sympy"
