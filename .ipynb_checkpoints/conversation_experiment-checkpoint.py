
# conversation_experiment.py
# Minimal, serverless prototype for Soma's thesis experiment.
# - Student turns: raw LLM with student-style few-shots (from Ethics text)
# - Professor turns: alternating Hybrid mode (Template -> Rewrite) for odd turns; Raw LLM for even turns
# - Two conditions: LLM_ONLY vs HYBRID
# - Logs results to CSV for human raters and runs LLM-as-judge scoring

import os, json, csv, random, time
from datetime import datetime
from typing import List, Dict

# ==== 0) CONFIG ====
# Set your API key in environment: export OPENAI_API_KEY=sk-...
OPENAI_MODEL_MAIN = "gpt-4o-mini"    # main generator and judge
OPENAI_MODEL_REWRITE = "gpt-4o-mini" # advisor rewrite (you can swap with HF endpoint for T5)
MAX_TURNS = 6  # (Professor 1, Student 1, Professor 2, Student 2, ...)
SEED = 42
random.seed(SEED)

# Files
ETHICS_STUDENT_EXAMPLES = "ethics_student_examples.txt"  # provide 10-30 short snippets of student writing style here
ADVISOR_TEMPLATES_JSON = "advisor_templates.json"        # schema below
OUTPUT_DIALOGUES_JSONL = "generated_dialogues.jsonl"
OUTPUT_JUDGE_CSV = "llm_judge_scores.csv"

# ==== 1) Helpers ====
def load_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(path: str, records: List[Dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def openai_chat(model: str, messages: List[Dict], temperature=0.7) -> str:
    # Minimal inline client using requests to avoid extra deps.
    import requests
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ==== 2) Personas ====
def student_turn_raw(context: str, student_fewshots: List[str]) -> str:
    fewshot_block = "\n".join([f"- {ex}" for ex in student_fewshots[:6]]) if student_fewshots else "- (no examples provided)"
    sys = "You are a student speaking informally about academic/career concerns. Keep it concise, natural, and authentic."
    usr = f"""Continue the dialogue as the student.
Style examples (do not copy content; mimic tone, phrasing, sentence length):
{fewshot_block}

Dialogue so far:
{context}

Student reply:"""
    return openai_chat(OPENAI_MODEL_MAIN, [{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.8)

def advisor_template_fill(template_obj: Dict, facts: Dict) -> str:
    # simple {slot} formatting
    text = template_obj["template"]
    return text.format(**facts)

def advisor_rewrite(structured_text: str) -> str:
    sys = "You are a supportive academic advisor. Keep facts unchanged. Goals: (1) accuracy, (2) empathy, (3) clear next step. 70–130 words."
    usr = f"Rewrite the following structured content into a natural, supportive advisor response:\n---\n{structured_text}\n---\nKeep all factual values unchanged."
    return openai_chat(OPENAI_MODEL_REWRITE, [{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.6)

def advisor_turn_raw(context: str) -> str:
    sys = "You are a supportive academic advisor. Be accurate, non-committal about unknowns, and propose small next steps."
    usr = f"Continue this dialogue as the advisor with a concise, supportive response.\n\nDialogue so far:\n{context}\n\nAdvisor reply:"
    return openai_chat(OPENAI_MODEL_MAIN, [{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.7)

# ==== 3) Generate one dialogue under a condition ====
def generate_dialogue(topic_prompt: str, facts: Dict, condition: str, student_fewshots: List[str], templates: List[Dict]) -> Dict:
    """
    condition in {"LLM_ONLY", "HYBRID"}
    HYBRID: professor odd turns use Template->Rewrite; professor even turns raw-LLM
    LLM_ONLY: both roles raw-LLM
    """
    ctx = f"Topic: {topic_prompt}\n"
    transcript = []
    tnum = 0
    while tnum < MAX_TURNS:
        # Professor turn
        tnum += 1
        if condition == "HYBRID" and (tnum % 2 == 1):  # odd professor turn -> Template->Rewrite
            template_obj = random.choice(templates)
            filled = advisor_template_fill(template_obj, facts)
            prof_text = advisor_rewrite(filled)
            transcript.append({"role":"advisor", "method":"template_rewrite", "text": prof_text, "structured":filled})
            ctx += f"\nAdvisor (T{tnum}): {prof_text}"
        else:
            prof_text = advisor_turn_raw(ctx)
            transcript.append({"role":"advisor", "method":"raw_llm", "text": prof_text})
            ctx += f"\nAdvisor (T{tnum}): {prof_text}"

        if tnum >= MAX_TURNS:
            break

        # Student turn
        tnum += 1
        stu_text = student_turn_raw(ctx, student_fewshots)
        transcript.append({"role":"student", "method":"raw_llm", "text": stu_text})
        ctx += f"\nStudent (T{tnum}): {stu_text}"

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "topic_prompt": topic_prompt,
        "condition": condition,
        "facts": facts,
        "transcript": transcript
    }

# ==== 4) LLM-as-judge ====
def judge_dialogue(transcript: List[Dict]) -> Dict:
    # Compact dialogue for judging
    compact = "\n".join([f"{t['role'].capitalize()}({t['method']}): {t['text']}" for t in transcript])
    sys = "You are a careful evaluator of advising dialogues."
    usr = f"""Rate the dialogue on the following 1–5 (integers only). Provide JSON.
Criteria:
- factuality: factual correctness and absence of hallucination
- empathy: supportive, respectful tone
- relevance: stays on-topic and offers actionable next steps
- naturalness: sounds like real student–advisor conversation

Dialogue:
{compact}

Return JSON with keys: factuality, empathy, relevance, naturalness."""
    out = openai_chat(OPENAI_MODEL_MAIN, [{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.2)
    # Best-effort parse
    try:
        data = json.loads(out)
    except Exception:
        data = {"factuality": None, "empathy": None, "relevance": None, "naturalness": None, "raw": out}
    return data

# ==== 5) Runner ====
def main():
    # Load resources
    student_fewshots = load_lines(ETHICS_STUDENT_EXAMPLES)
    templates = load_json(ADVISOR_TEMPLATES_JSON) if os.path.exists(ADVISOR_TEMPLATES_JSON) else [
        {"template":"Given CGPA {cgpa} and missing prereq {missing_course}, acknowledge stress and recommend two lighter-load courses aligned with {interest}."},
        {"template":"With CGPA {cgpa} and {credits_left} credits remaining, discuss eligibility for {opportunity} and propose next steps."}
    ]

    # Example facts per dialogue (replace with your data source)
    facts_pool = [
        {"cgpa": 3.2, "missing_course":"CS351", "interest":"systems", "credits_left":24, "opportunity":"research assistantship"},
        {"cgpa": 3.8, "missing_course":"", "interest":"AI", "credits_left":12, "opportunity":"graduate fellowships"}
    ]
    topics = [
        "I feel overwhelmed and not sure how many courses to take next term.",
        "I want internships but my GPA dipped. What should I focus on this semester?"
    ]

    records = []
    judge_rows = []
    for i, topic in enumerate(topics):
        facts = random.choice(facts_pool)

        for cond in ["LLM_ONLY", "HYBRID"]:
            dlg = generate_dialogue(topic, facts, cond, student_fewshots, templates)
            # Save dialogue
            records.append(dlg)
            # Judge
            score = judge_dialogue(dlg["transcript"])
            judge_rows.append({
                "dialogue_id": f"d{i}_{cond}",
                "condition": cond,
                **{k: score.get(k) for k in ["factuality","empathy","relevance","naturalness"]}
            })
            time.sleep(1)

    save_jsonl(OUTPUT_DIALOGUES_JSONL, records)

    # write judge csv
    fieldnames = ["dialogue_id","condition","factuality","empathy","relevance","naturalness"]
    with open(OUTPUT_JUDGE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in judge_rows:
            w.writerow(row)

if __name__ == "__main__":
    main()
