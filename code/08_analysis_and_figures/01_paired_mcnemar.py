"""Firm up the iterate-vs-apply_all +9 claim with a PAIRED (McNemar) test on the SAME items, split by
comprehension-containing vs not (pre-specified from the collateral analysis). Also report the guards-vs-tests
AGREEMENT as a measured property (not the overclaim 'guards capture correctness'). GPU-free, from saved outputs."""
import json, collections, math
outs=json.load(open("base_as_fixer_outputs.json"))
def key(o): return (o["repo"],o["file"],o["func"],o["start"])
it={key(o):o for o in outs["iterate"]}; aa={key(o):o for o in outs["apply_all"]}

for label,pred in [("comprehension items", lambda cs:"comprehension" in cs),
                   ("NON-comprehension",   lambda cs:"comprehension" not in cs)]:
    b=c=both=neither=n=0
    for k in set(it)&set(aa):
        if not pred(it[k]["concerns"]): continue
        ic=it[k]["guards"]["correct"]; ac=aa[k]["guards"]["correct"]; n+=1
        if ic and not ac: b+=1
        elif ac and not ic: c+=1
        elif ic and ac: both+=1
        else: neither+=1
    disc=b+c
    # McNemar (no continuity corr): chi2 = (b-c)^2/(b+c); p via normal approx z=(b-c)/sqrt(b+c)
    z=(b-c)/math.sqrt(disc) if disc else 0.0
    print(f"{label} (n={n}): iterate-only-right(b)={b}  apply-only-right(c)={c}  both={both}  neither={neither}")
    print(f"    net = b-c = {b-c} ({100*(b-c)//max(n,1):+d} pts) | McNemar z={z:.2f} (|z|>=1.96 sig) -> "
          f"{'SIGNIFICANT' if abs(z)>=1.96 else 'NOT significant (consistent with predicted mechanism, not a proven gain)'}")

# guards-vs-tests agreement (measured property, from the behavior run: full==cg meant 0 disagreements among cg-correct)
print("\nGuards-vs-tests AGREEMENT (measured, not assumed):")
print("  From base_as_fixer_tests: of the 217 concerns+guards-CORRECT outputs (iterate 98 + apply_all 91 + single 28),")
print("  0 failed the repo tests (+behavior == concerns+guards for all 3 methods). => ruff+guards is a faithful")
print("  behavior proxy ON THIS DATA (validates Tier-B ruff-only grading); NOT a claim that guards subsume behavior in general.")
print("DONE")
