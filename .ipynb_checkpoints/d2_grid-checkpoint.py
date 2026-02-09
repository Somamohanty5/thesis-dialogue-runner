#!/usr/bin/env python3
import itertools, json, subprocess, shlex, time
from pathlib import Path
import requests

RUN_ONE = "run_one_test.py"
TOPIC = "internships"
K = 12

CONDITION = "C"
TURNS_LIST = [6, 8, 12]
STRENGTHS = ["weak", "average", "strong"]

PROFILES_FILE = Path("survey_profiles_clusterclip_balanced.json")

def ping_ollama(url: str = "http://localhost:11434/api/tags", timeout: float = 5.0) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def pick_one_sid_per_strength():
    """Match original run_one.py behavior: pick first sid per strength from profiles."""
    profiles = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    sids = {}
    for strength in STRENGTHS:
        pool = [p for p in sorted(profiles, key=lambda x: x.get("student_id", 0))
                if p.get("profile_strength") == strength]
        if not pool:
            raise SystemExit(f"No profiles with strength={strength}")
        sids[strength] = pool[0].get("student_id")
    return sids

def run_one(cond, strength, topic, k, sid, turns, retries=2, sleep_between=1.0, ping_timeout=5.0):
    cmd = (
        f"python3 {RUN_ONE} "
        f"--condition {cond} "
        f"--strength {strength} "
        f"--topic {topic} "
        f"--k {k} "
        f"--sid {sid} "
        f"--turns {turns}"
    )

    # ping (only matters for local Ollama mode)
    if not ping_ollama(timeout=ping_timeout):
        print("[wait] Ollama not responding; retry ping in 3s...")
        time.sleep(3)

    attempts = retries + 1
    logdir = Path("runs_grid/logs")
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"{cond}_{strength}_{topic}_k{k}_sid{sid}_t{turns}.log"

    for i in range(attempts):
        try:
            print("→", cmd)
            with logfile.open("a", encoding="utf-8") as lf:
                lf.write(f"\n=== Attempt {i+1}/{attempts} ===\n")
                subprocess.run(shlex.split(cmd), check=True, stdout=lf, stderr=lf, text=True)
            if sleep_between > 0:
                time.sleep(sleep_between)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[error] exit {e.returncode} attempt {i+1}/{attempts}")
            if i < attempts - 1:
                backoff = max(1.5, sleep_between * (i + 2))
                print(f"[retry] sleeping {backoff:.1f}s…")
                time.sleep(backoff)
    print("[fatal] giving up on this run:", cmd)
    return False

def main():
    sids = pick_one_sid_per_strength()

    runs = []
    for strength, turns in itertools.product(STRENGTHS, TURNS_LIST):
        runs.append({
            "condition": CONDITION,
            "strength": strength,
            "topic": TOPIC,
            "k": K,
            "sid": sids[strength],
            "turns": turns
        })

    Path("runs_grid").mkdir(exist_ok=True)
    Path("runs_grid/manifest_9.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")

    ok = 0
    for r in runs:
        if run_one(r["condition"], r["strength"], r["topic"], r["k"], r["sid"], r["turns"]):
            ok += 1

    print(f"\n Completed {ok}/{len(runs)} runs.")
    print("Outputs are in runs_one/... (from run_one.py) and logs in runs_grid/logs/")

if __name__ == "__main__":
    main()
