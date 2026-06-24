import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_audit.judge import fusion, rubric, client
batch = json.loads((Path(__file__).resolve().parents[1] / "outputs/batches/paired_001.json").read_text())
t = batch[0]
user = (f"PROBLEM:\n{t['problem'][:1500]}\n\nMODEL CoT:\n{t['reasoning']}\n\n"
        f"FINAL ANSWER (context only):\n{t['answer'][:1200]}\n\nReturn the JSON object now.")
print(f"trace: {t['dataset']} reasoning={len(t['reasoning'])} chars")
for name, fn in fusion.BACKENDS.items():
    try:
        content, usage = fn(rubric.SYSTEM, user, max_tokens=12000)
        d = client.parse_json(content)
        print(f"  {name:14s} OK | ccr={d.get('ccr_closure')} topo={d.get('reasoning_topology')} coupling={d.get('monitoring_control_coupling')} depth={d.get('causal_depth_of_critique')} | completion_tokens={usage.get('completion_tokens')}")
    except Exception as e:
        print(f"  {name:14s} FAIL: {str(e)[:160]}")
