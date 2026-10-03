source /home/hrish/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate kg-blackwell 2>/dev/null
DJ=/data/home/hrish/kg-adapter/swebench_repos/django__django
cd $DJ
echo "coverage run start $(date)"
PYTHONPATH=$DJ timeout 5400 coverage run --source=$DJ/django tests/runtests.py --parallel=1 >/dev/null 2>&1
echo "coverage run done $(date), exit=$?"
PYTHONPATH=$DJ coverage json -o /data/home/hrish/kg-adapter/KG-Adapter/moa/django_cov.json 2>/dev/null
cd /data/home/hrish/kg-adapter/KG-Adapter/moa
python3 -c "
import json
cov=json.load(open('django_cov.json'))['files']
items=json.load(open('verif_django__django.json'))
covered=[]
for it in items:
    # coverage json keys are absolute paths of django/*.py
    key=[k for k in cov if k.endswith(it['file'])]
    if not key: continue
    ex=set(cov[key[0]]['executed_lines'])
    if ex & set(range(it['start'], it['end']+1)):
        covered.append(it)
json.dump(covered, open('verif_django_covered.json','w'), indent=1)
print('django items total:', len(items), '| test-COVERED:', len(covered))
import statistics
if covered: print('covered n_lines median:', statistics.median([c['n_lines'] for c in covered]))
"
echo DJANGO_COV_DONE
