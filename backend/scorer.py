import os
import json
import logging
from typing import List, Dict, Any, Tuple

import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# =========================================================
# SETUP
# =========================================================

load_dotenv()
logging.basicConfig(level=logging.INFO)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_gemini = genai.GenerativeModel("gemini-1.5-flash")


# =========================================================
# PYDANTIC SCHEMA
# =========================================================

class ScoreSchema(BaseModel):
    matched_skills:  List[str]
    missing_skills:  List[str]
    skill_score:     int
    experience_score: int
    seniority_score: int
    remote_score:    int
    bonus_score:     int
    total_score:     int
    reasoning:       str


# =========================================================
# WEIGHTS — total must equal 100
# =========================================================

WEIGHTS = {
    "skill_match":  55,   # core — how many required skills match
    "experience":   20,   # years of experience fit
    "seniority":    15,   # level fit (junior/mid/senior)
    "remote":        5,   # remote preference
    "bonus":         5,   # nice-to-have skills bonus
}


# =========================================================
# SENIORITY LEVEL MAP
# =========================================================

SENIORITY_LEVELS = {
    "intern":   0,
    "junior":   1,
    "entry":    1,
    "mid":      2,
    "middle":   2,
    "senior":   3,
    "lead":     4,
    "principal":5,
    "staff":    5,
    "manager":  5,
    "director": 6,
}


# =========================================================
# SKILL ALIASES — for fuzzy matching
# =========================================================

SKILL_ALIASES = {
    "javascript":      ["js", "es6", "es2015", "ecmascript"],
    "typescript":      ["ts"],
    "nodejs":          ["node", "node.js"],
    "python":          ["py"],
    "postgresql":      ["postgres", "psql"],
    "mongodb":         ["mongo"],
    "kubernetes":      ["k8s"],
    "tensorflow":      ["tf"],
    "machine learning":["ml", "machine-learning"],
    "deep learning":   ["dl", "deep-learning"],
    "generative ai":   ["gen ai", "genai"],
    "llm":             ["llms", "large language model"],
    "react":           ["reactjs", "react.js"],
    "vuejs":           ["vue", "vue.js"],
    "nextjs":          ["next", "next.js"],
    "aws":             ["amazon web services"],
    "gcp":             ["google cloud"],
    "ci/cd":           ["cicd", "continuous integration"],
    "nlp":             ["natural language processing"],
    "computer vision": ["cv", "image processing"],
}

# Build reverse map: alias → canonical
_ALIAS_REVERSE: Dict[str, str] = {}
for canonical, aliases in SKILL_ALIASES.items():
    for alias in aliases:
        _ALIAS_REVERSE[alias] = canonical


# =========================================================
# SKILL NORMALIZATION
# =========================================================

def normalize_skill(skill: str) -> str:
    """Normalize a skill to its canonical form."""
    skill = skill.lower().strip()
    return _ALIAS_REVERSE.get(skill, skill)


def normalize_skill_list(skills: List[str]) -> List[str]:
    """Normalize entire skill list."""
    return list({normalize_skill(s) for s in skills})


# =========================================================
# FUZZY SKILL MATCHING
# =========================================================

def skills_match(candidate_skill: str,
                 required_skill: str) -> bool:
    """
    Check if candidate skill satisfies required skill.
    Handles exact match + alias match + substring match.
    """
    c = normalize_skill(candidate_skill)
    r = normalize_skill(required_skill)

    # Exact match
    if c == r:
        return True

    # Substring match (python matches python developer)
    if c in r or r in c:
        return True

    # Alias cross-check
    c_canonical = _ALIAS_REVERSE.get(c, c)
    r_canonical = _ALIAS_REVERSE.get(r, r)
    if c_canonical == r_canonical:
        return True

    return False


def match_skills(
    candidate_skills: List[str],
    required_skills:  List[str]
) -> Tuple[List[str], List[str]]:
    """
    Returns (matched_skills, missing_skills).
    """
    matched = []
    missing = []

    for req in required_skills:
        found = any(
            skills_match(cand, req)
            for cand in candidate_skills
        )
        if found:
            matched.append(req)
        else:
            missing.append(req)

    return matched, missing


# =========================================================
# EXPERIENCE SCORE
# =========================================================

def score_experience(
    candidate_years: int,
    required_years:  int
) -> int:
    """
    Dynamic experience scoring.
    Penalizes under-qualification more than over-qualification.
    """
    if required_years <= 0:
        return WEIGHTS["experience"]

    ratio = candidate_years / required_years

    if ratio >= 1.0:
        # Meets or exceeds requirement — full or near full score
        return min(WEIGHTS["experience"],
                   int(WEIGHTS["experience"] * min(ratio, 1.3) / 1.3))
    elif ratio >= 0.7:
        # Slightly under — partial score
        return int(WEIGHTS["experience"] * ratio * 0.85)
    elif ratio >= 0.4:
        # Significantly under — heavy penalty
        return int(WEIGHTS["experience"] * ratio * 0.5)
    else:
        # Way under qualified
        return int(WEIGHTS["experience"] * ratio * 0.2)


# =========================================================
# SENIORITY SCORE
# =========================================================

def score_seniority(
    candidate_level: str,
    required_level:  str
) -> int:
    """
    Dynamic seniority scoring based on level gap.
    """
    c_level = SENIORITY_LEVELS.get(
        candidate_level.lower().strip(), 2
    )
    r_level = SENIORITY_LEVELS.get(
        required_level.lower().strip(), 2
    )

    gap = abs(c_level - r_level)

    if gap == 0:
        return WEIGHTS["seniority"]           # perfect fit
    elif gap == 1:
        return int(WEIGHTS["seniority"] * 0.75)  # close fit
    elif gap == 2:
        return int(WEIGHTS["seniority"] * 0.45)  # stretch
    else:
        return int(WEIGHTS["seniority"] * 0.15)  # mismatch


# =========================================================
# SCORE LABEL + COLOR
# =========================================================

def get_score_label(score: int) -> Dict[str, str]:
    """
    Dynamic label and color based on score.
    Green/Amber/Red system for UI.
    """
    if score >= 80:
        return {
            "label": "Excellent Match",
            "color": "green",
            "emoji": "🟢",
            "advice": "Apply immediately — strong fit"
        }
    elif score >= 65:
        return {
            "label": "Great Match",
            "color": "green",
            "emoji": "🟢",
            "advice": "Apply with confidence"
        }
    elif score >= 50:
        return {
            "label": "Fair Match",
            "color": "amber",
            "emoji": "🟡",
            "advice": "Apply with a strong cover letter"
        }
    elif score >= 35:
        return {
            "label": "Weak Match",
            "color": "red",
            "emoji": "🔴",
            "advice": "Skill gap exists — apply carefully"
        }
    else:
        return {
            "label": "Poor Match",
            "color": "red",
            "emoji": "🔴",
            "advice": "Significant mismatch — consider skipping"
        }


# =========================================================
# GEMINI REASONING (optional enrichment)
# =========================================================

REASONING_PROMPT = """
You are a recruiter scoring a job match.

Candidate skills: {candidate_skills}
Required skills:  {required_skills}
Matched skills:   {matched}
Missing skills:   {missing}
Score:            {score}/100

Write ONE short sentence (max 20 words) explaining this match score.
Be direct and honest. No fluff.
"""


def get_gemini_reasoning(
    candidate_skills: List[str],
    required_skills:  List[str],
    matched:          List[str],
    missing:          List[str],
    score:            int
) -> str:
    """
    Use Gemini to generate a human-readable match reasoning.
    Falls back to rule-based if Gemini fails.
    """
    try:
        prompt = REASONING_PROMPT.format(
            candidate_skills=", ".join(candidate_skills[:10]),
            required_skills=", ".join(required_skills[:10]),
            matched=", ".join(matched[:5]),
            missing=", ".join(missing[:5]),
            score=score
        )
        response = _gemini.generate_content(
            prompt,
            generation_config={"temperature": 0.3}
        )
        return response.text.strip()

    except Exception as e:
        logging.warning(f"Gemini reasoning failed: {e}")
        return _rule_based_reasoning(matched, missing, score)


def _rule_based_reasoning(
    matched: List[str],
    missing: List[str],
    score:   int
) -> str:
    """Fallback reasoning without Gemini."""
    if score >= 65:
        return (f"Strong match with {len(matched)} skills aligned"
                f" and only {len(missing)} gaps.")
    elif score >= 45:
        return (f"Partial match — {len(matched)} skills fit"
                f" but {len(missing)} key skills missing.")
    else:
        return (f"Weak match — {len(missing)} required skills"
                f" missing from candidate profile.")


# =========================================================
# MAIN SCORER FUNCTION
# =========================================================

def score_job(
    candidate_skills:  List[str],
    required_skills:   List[str],
    nice_skills:       List[str]   = None,
    candidate_years:   int         = 0,
    required_years:    int         = 0,
    candidate_level:   str         = "mid",
    required_level:    str         = "mid",
    remote_friendly:   bool        = True,
    job_type:          str         = "full-time",
    use_ai_reasoning:  bool        = True
) -> Dict[str, Any]:
    """
    Main scoring function.
    Returns complete score breakdown + label + reasoning.
    """

    nice_skills = nice_skills or []

    # Normalize all skill lists
    c_skills  = normalize_skill_list(candidate_skills)
    r_skills  = normalize_skill_list(required_skills)
    n_skills  = normalize_skill_list(nice_skills)

    # ── Skill match score (55 pts) ─────────────────────────
    matched, missing = match_skills(c_skills, r_skills)

    total_required = max(len(r_skills), 1)
    skill_score    = round(
        (len(matched) / total_required) * WEIGHTS["skill_match"]
    )

    # ── Nice-to-have bonus (5 pts) ─────────────────────────
    if n_skills:
        nice_matched, _ = match_skills(c_skills, n_skills)
        bonus_score     = round(
            (len(nice_matched) / max(len(n_skills), 1))
            * WEIGHTS["bonus"]
        )
    else:
        bonus_score = WEIGHTS["bonus"] // 2    # neutral if no data

    # ── Experience score (20 pts) ──────────────────────────
    exp_score = score_experience(candidate_years, required_years)

    # ── Seniority score (15 pts) ───────────────────────────
    sen_score = score_seniority(candidate_level, required_level)

    # ── Remote fit score (5 pts) ───────────────────────────
    remote_score = WEIGHTS["remote"] if remote_friendly else 2

    # ── Total ──────────────────────────────────────────────
    total = min(
        skill_score + bonus_score + exp_score +
        sen_score   + remote_score,
        100
    )

    # ── Label + color ──────────────────────────────────────
    label_data = get_score_label(total)

    # ── AI reasoning ───────────────────────────────────────
    if use_ai_reasoning:
        reasoning = get_gemini_reasoning(
            c_skills, r_skills, matched, missing, total
        )
    else:
        reasoning = _rule_based_reasoning(matched, missing, total)

    return {
        "score":           total,
        "label":           label_data["label"],
        "color":           label_data["color"],
        "emoji":           label_data["emoji"],
        "advice":          label_data["advice"],
        "matched_skills":  matched,
        "missing_skills":  missing,
        "reasoning":       reasoning,
        "breakdown": {
            "skill_match":  skill_score,
            "experience":   exp_score,
            "seniority":    sen_score,
            "remote_fit":   remote_score,
            "bonus":        bonus_score,
        }
    }


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    result = score_job(
        candidate_skills = ["python", "flask", "ml",
                            "mongodb", "docker", "js"],
        required_skills  = ["python", "machine learning",
                            "nodejs", "aws", "kubernetes"],
        nice_skills      = ["docker", "mongodb"],
        candidate_years  = 3,
        required_years   = 4,
        candidate_level  = "mid",
        required_level   = "senior",
        remote_friendly  = True,
        use_ai_reasoning = True
    )

    print(json.dumps(result, indent=2))