#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, glob, json, os, re, time, textwrap
from pathlib import Path

import requests

API_BASE = "http://localhost:11434"

def ping_ollama(url=f"{API_BASE}/api/tags", timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def call_chat(model, messages, temperature=0.0, timeout=120, max_tokens=256, retries=3, backoff=2.0):
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
            "repeat_penalty": 1.1,
            "num_ctx": 1536  # modest to reduce OOM / stalls
        },
        "stream": False
    }
    url = f"{API_BASE}/api/chat"
    err = None
    for attempt in range(retries):
        try:
            if not ping_ollama():
                time.sleep(min(3*(attempt+1), 8))
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and "message" in data:
                    return (data["message"]["content"] or "").strip()
                # some older servers return list of deltas
                if isinstance(data, list) and data and "message" in data[-1]:
                    return (data[-1]["message"]["content"] or "").strip()
                return ""
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            err = str(e)
        time.sleep(backoff*(attempt+1))
    raise RuntimeError(f"/api/chat failed after retries: {err}")

RUBRIC = textwrap.dedent("""\
You are a strict but fair evaluator of short student↔advisor chats for academic advising.

Rate EACH metric from 1 (poor) to 5 (excellent) using the definitions:

- Empathy: acknowledges feelings briefly and respectfully.
- Clarity: direct, specific, easy to follow; avoids meta or rambling.
- Usefulness: gives concrete, realistic guidance for the student’s question.
- Personalization: adapts to the student's situation (not generic boilerplate).
- Continuity: builds on prior turn (answers the exact question asked).
- NextStep: ends advisor turn with exactly one clear, actionable step (imperative, short).

Return STRICTLY a compact JSON object with integers 1–5 and a one-sentence comment:
{"Empathy": int, "Clarity": int, "Usefulness": int, "Personalization": int, "Continuity": int, "NextStep": int, "Comment": "…"}
Do not add extra text.
""")

def load_convo(path: Path):
    """Return header, convo_text (trimmed if needed)."""
    raw = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    header = lines[0] if lines else ""
    convo = "\n".join(lines[1:])
    # keep context manageable (older small models can stall on very long inputs)
    if len(convo) > 5000:
        convo = convo[-5000:]
    return header, convo

def build_eval_prompt(convo_text: str):
    return textwrap.dedent(f"""\
    Conversation transcript (STUDENT/ADVISOR):

    \"\"\"{convo_text}\"\"\"

    Evaluate ONLY the advisor messages according to the rubric.
    Output JSON per the format exactly (no code fences, no commentary).
    """)

def judge_file(model: str, path: Path):
    header, convo = load_convo(path)
    user = build_eval_prompt(convo)
    msgs = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": user},
    ]
    out = call_chat(model, msgs, temperature=0.0, timeout=120, max_tokens=256)
    # try parse JSON; if model added junk, find first {...}
    m = re.search(r'\{.*\}', out, flags=re.S)
    if not m:
        raise ValueError(f"Judge returned non-JSON for {path.name}: {out[:200]}")
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        # light repair: remove trailing commas and re-try once
        cleaned = re.sub(r',\s*}', '}', m.group(0))
        obj = json.loads(cleaned)
    # coerce to ints range 1–5
    row = {
        "file": str(path),
        "header": header,
        "Empathy": int(max(1, min(5, int(obj.get("Empathy", 1))))),
        "Clarity": int(max(1, min(5, int(obj.get("Clarity", 1))))),
        "Usefulness": int(max(1, min(5, int(obj.get("Usefulness", 1))))),
        "Personalization": int(max(1, min(5, int(obj.get("Personalization", 1))))),
        "Continuity": int(max(1, min(5, int(obj.get("Continuity", 1))))),
        "NextStep": int(max(1, min(5, int(obj.get("NextStep", 1))))),
        "Comment": str(obj.get("Comment", "")).strip()[:500]
    }
    # composite average
    row["Avg"] = round(sum([row[k] for k in ("Empathy","Clarity","Usefulness","Personalization","Continuity","NextStep")]) / 6.0, 2)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "llama3.2:3b-instruct-q4_K_M"))
    ap.add_argument("--glob", default="runs_one/*/*.txt")
    ap.add_argument("--out", default="eval_judge.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"No files matched {args.glob}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    print(f"[judge] model={args.model} files={len(files)} → {outp}")

    rows = []
    for i, f in enumerate(files, 1):
        p = Path(f)
        try:
            row = judge_file(args.model, p)
            rows.append(row)
            print(f"[{i}/{len(files)}] {p.name} Avg={row['Avg']}  ({row['Empathy']},{row['Clarity']},{row['Usefulness']},{row['Personalization']},{row['Continuity']},{row['NextStep']})")
        except Exception as e:
            print(f"[warn] {p.name}: {e}")

        # tiny pacing to avoid hammering the server
        time.sleep(0.25)

    with outp.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["file","header","Empathy","Clarity","Usefulness","Personalization","Continuity","NextStep","Avg","Comment"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] wrote {outp} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
