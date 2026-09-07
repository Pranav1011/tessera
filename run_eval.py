"""Run the multi-seed slice eval and (if a baseline exists) the regression gate.
Writes results/eval.json. On ABC data this produces the real bullet-4 numbers.

    python run_eval.py            # evaluate + write results
    python run_eval.py --gate     # also fail (exit 1) if a slice regressed vs baseline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tessera.eval_harness import regression_gate, run_multiseed

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASELINE = RESULTS / "baseline.json"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    res = run_multiseed(seeds=(0, 1, 2), n_per=16)
    (RESULTS / "eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_seed"}, indent=2))

    if "--gate" in sys.argv and BASELINE.exists():
        base = json.loads(BASELINE.read_text())
        fails = regression_gate(res["per_seed"][0], base["per_seed"][0])
        if fails:
            print("\nREGRESSION GATE FAILED:")
            print("\n".join(f"  - {f}" for f in fails))
            sys.exit(1)
        print("\nregression gate: OK")
    elif not BASELINE.exists():
        BASELINE.write_text(json.dumps(res, indent=2))
        print(f"\nwrote baseline -> {BASELINE}")


if __name__ == "__main__":
    main()
