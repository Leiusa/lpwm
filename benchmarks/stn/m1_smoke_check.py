"""Verdict of the milestone-1 smoke test. Harness validation only: the tolerance below checks that the new evaluation script
reproduces earlier recorded numbers; it is NOT a quality threshold and plays no role in the study's analysis."""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--smoke-dir", required=True)
ap.add_argument("--unit-rc", type=int, required=True)
ap.add_argument("--tiny-rc", type=int, required=True)
ap.add_argument("--eval-rc", type=int, required=True)
ap.add_argument("--gate-recorded", type=int, default=1, help="1: the comparison with the earlier A6000 numbers gates the verdict (only meaningful on an A6000)")
ap.add_argument("--recorded", required=True)          # {"E1": mse, "E2": mse, "E3": mse} from the earlier seed-2 evaluations (1,2,3 of the audit)
args = ap.parse_args()
SANITY_REL_TOL = 1e-3
d = args.smoke_dir
res, problems = {}, []


def need(cond, msg):
    res[msg] = bool(cond)
    if not cond:
        problems.append(msg)


need(args.unit_rc == 0, "hardware decision unit checks")
need(args.tiny_rc == 0, "tiny pair driver exit code 0")
need(args.eval_rc == 0, "seed-2 four-way evaluation exit code 0")
try:
    tr = json.load(open(os.path.join(d, "tiny", "train_report.json")))
    need(tr["hardware"]["verified"], "tiny: hardware verified in python")
    need(tr["hardware"]["nvidia_smi"][0]["name"] == tr["hardware"]["expected_gpu"], "tiny: GPU model equals the expected model")
    need(all(r["runtime_evidence"]["evidence_ok"] for r in tr["runs"]), "tiny: both trainings' runtime evidence ok")
    need(all(r["epochs_completed"] == 1 and r["final_checkpoint_exists"] for r in tr["runs"]), "tiny: both trainings complete with checkpoint")
    need(tr.get("pairing", {}).get("init_weights_identical") and tr.get("pairing", {}).get("first_batch_identical"), "tiny: identical init weights and first batch")
    need(all(r["settings_unchanged"] for r in tr["runs"]), "tiny: agreed settings unchanged by training")
    te = json.load(open(os.path.join(d, "tiny", "eval_m1.json")))
    need(te["all_evidence_ok"], "tiny: all four evaluations' evidence ok")
    need(len(te["evaluations"]) == 4 and all(v["finite"] for v in te["evaluations"].values()), "tiny: four finite evaluations")
except Exception as exc:  # noqa: BLE001
    need(False, f"tiny outputs readable ({type(exc).__name__}: {exc})")
try:
    se = json.load(open(os.path.join(d, "eval_seed2_existing.json")))
    rec = json.load(open(args.recorded))
    need(se["all_evidence_ok"], "seed-2: all four evaluations' evidence ok")
    need(se["validation_images"] == 1200, "seed-2: 1,200 validation images")
    for k in ("E1", "E2", "E3"):
        new = se["evaluations"][k]["full_validation"]["mse"]
        rel = abs(new - rec[k]) / rec[k]
        if args.gate_recorded:
            res[f"seed-2 {k} pooled MSE vs recorded: rel diff {rel:.2e}"] = rel < SANITY_REL_TOL
            if rel >= SANITY_REL_TOL:
                problems.append(f"seed-2 {k} differs from the recorded value by {rel:.2e}")
        else:
            print(f"INFO (not gated: different GPU model than the recorded A6000 numbers) seed-2 {k} pooled MSE rel diff vs recorded: {rel:.2e}")
    print("seed-2 new values:", {k: (v["full_validation"]["mse"], v["full_validation"]["psnr_pooled_db"], v["full_validation"]["psnr_per_image_mean_db"]) for k, v in se["evaluations"].items()})
except Exception as exc:  # noqa: BLE001
    need(False, f"seed-2 outputs readable ({type(exc).__name__}: {exc})")
for k, v in res.items():
    print(("PASS " if v else "FAIL ") + k)
ok = not problems
open(os.path.join(d, "SMOKE_PASS" if ok else "SMOKE_FAIL"), "w").write(json.dumps({"problems": problems}, indent=1))
print("SMOKE_PASS" if ok else "SMOKE_FAIL: " + "; ".join(problems))
sys.exit(0 if ok else 1)
