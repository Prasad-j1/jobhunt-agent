import os
import json
import logging
import re
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

from backend.scorer import score_job

# =========================================================
# SETUP
# =========================================================

load_dotenv()
logging.basicConfig(level=logging.INFO)


# =========================================================
# PYDANTIC OUTPUT SCHEMA
# =========================================================

class JobResult(BaseModel):
    title:          str
    company:        str       = "Unknown"
    url:            str
    location:       str       = "Remote"
    job_type:       str       = "Full-time"
    score:          int       = 0
    label:          str       = "Unknown"
    color:          str       = "gray"
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    cover_letter:   str       = ""
    reasoning:      str       = ""


# =========================================================
# LAZY CLIENTS — initialized only when first needed
# =========================================================

_tavily: Optional[TavilyClient] = None


def get_tavily() -> TavilyClient:
    global _tavily
    if _tavily is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY missing in .env")
        _tavily = TavilyClient(api_key=api_key)
    return _tavily


def get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY missing in .env")
    return ChatGoogleGenerativeAI(
        model                        = "gemini-2.0-flash",
        google_api_key               = api_key,
        temperature                  = 0.1,
        convert_system_message_to_human = True
    )


# =========================================================
# URL VALIDATOR
# =========================================================

BAD_URL_PATTERNS = [
    "linkedin.com/feed",
    "linkedin.com/company",
    "linkedin.com/search",
    "google.com/search",
    "indeed.com/jobs?",
    "glassdoor.com/Job/jobs",
    "/search?",
    "/jobs?q=",
]


def is_valid_job_url(url: str) -> bool:
    if not url:
        return False
    url_lower = url.lower()
    for pattern in BAD_URL_PATTERNS:
        if pattern in url_lower:
            return False
    return True


# =========================================================
# COMPANY EXTRACTOR
# =========================================================

def extract_company_from_title(title: str) -> str:
    """
    Extract company name from job title string.
    Handles formats like:
      'AI Engineer at Google'
      'ML Engineer - OpenAI'
      'Python Dev | Stripe'
    """
    for separator in [" at ", " - ", " | ", " @ "]:
        if separator.lower() in title.lower():
            parts = title.split(separator)
            if len(parts) >= 2:
                return parts[-1].strip()
    return "Unknown"


# =========================================================
# QUERY BUILDER
# =========================================================

def build_queries(
    user_message: str,
    resume_data:  Dict
) -> List[str]:
    """
    Build 3-5 search queries from user message + resume.
    """
    queries   = []
    titles    = resume_data.get("job_titles", [])
    skills    = resume_data.get("skills",     [])
    seniority = resume_data.get("seniority",  "mid")

    # User message is always first priority
    if user_message:
        queries.append(user_message)

    # Title-based queries
    for title in titles[:2]:
        queries.append(f"{seniority} {title} remote jobs 2025")
        queries.append(f"{title} hiring now")

    # Skill-based query
    if skills:
        top_skills = " ".join(skills[:3])
        queries.append(f"{top_skills} jobs remote 2025")

    # Deduplicate and limit
    seen    = set()
    unique  = []
    for q in queries:
        q_clean = q.lower().strip()
        if q_clean not in seen:
            seen.add(q_clean)
            unique.append(q)

    return unique[:5]


# =========================================================
# STEP 1 — SEARCH JOBS
# =========================================================

def search_jobs(
    query:       str,
    max_results: int = 8
) -> List[Dict]:
    logging.info(f"[SEARCH] Query: {query}")
    try:
        results = get_tavily().search(
            query        = query + " job hiring",
            max_results  = max_results,
            search_depth = "advanced"
        )
        jobs = []
        for r in results.get("results", []):
            url = r.get("url", "")
            if not is_valid_job_url(url):
                continue
            jobs.append({
                "title":   r.get("title", ""),
                "url":     url,
                "snippet": r.get("content", "")[:300]
            })
        logging.info(f"[SEARCH] {len(jobs)} valid results")
        return jobs
    except Exception as e:
        logging.error(f"[SEARCH] Error: {e}")
        return []


# =========================================================
# STEP 2 — DEDUPLICATE
# =========================================================

def remove_duplicate_jobs(jobs: List[Dict]) -> List[Dict]:
    """
    Remove duplicates based on URL + title combination.
    Also catches near-duplicate URLs (same job, different params).
    """
    seen        = set()
    unique_jobs = []

    for job in jobs:
        # Normalize URL — strip query params for comparison
        url_clean   = job.get("url", "").split("?")[0].lower().strip()
        title_clean = job.get("title", "").lower().strip()

        key = (title_clean, url_clean)

        if key in seen:
            continue

        seen.add(key)
        unique_jobs.append(job)

    logging.info(
        f"[DEDUP] {len(unique_jobs)} unique jobs "
        f"after deduplication"
    )
    return unique_jobs


# =========================================================
# STEP 3 — READ JOB POSTING
# =========================================================

def read_job_posting(url: str) -> str:
    logging.info(f"[READ] {url}")

    # Method 1 — Try Tavily extract if available
    try:
        client = get_tavily()
        if hasattr(client, 'extract'):
            result = client.extract(urls=[url])
            if result and result.get("results"):
                content = result["results"][0].get("raw_content","")
                if content:
                    logging.info(f"[READ] Tavily extracted {len(content)} chars")
                    return content[:6000]
    except Exception as e:
        logging.warning(f"[READ] Tavily extract failed: {e}")

    # Method 2 — Direct HTTP request fallback
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code == 200:
            text = resp.text
            # Basic HTML tag removal
            import re
            text = re.sub(r'<script[^>]*>.*?</script>',
                          '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>',
                          '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s{3,}', '\n', text)
            text = text.strip()
            if len(text) > 200:
                logging.info(f"[READ] HTTP extracted {len(text)} chars")
                return text[:6000]
    except Exception as e:
        logging.error(f"[READ] HTTP fallback failed: {e}")

    # Method 3 — Use Tavily search on the URL as last resort
    try:
        results = get_tavily().search(
            query    = url,
            max_results = 1,
            search_depth = "basic"
        )
        for r in results.get("results", []):
            content = r.get("content", "")
            if len(content) > 100:
                logging.info(f"[READ] Tavily search fallback: {len(content)} chars")
                return content[:6000]
    except Exception as e:
        logging.error(f"[READ] All methods failed for {url}: {e}")

    return ""


# =========================================================
# STEP 4 — EXTRACT JOB REQUIREMENTS VIA GEMINI
# =========================================================

JOB_EXTRACTION_PROMPT = """You are a job requirement extractor.
Extract structured requirements from the job description below.
Return ONLY valid JSON. No markdown. No explanation.

JOB DESCRIPTION:
{text}

Return exactly this JSON:
{{
  "required_skills": ["skill1", "skill2"],
  "nice_skills":     ["skill3"],
  "required_years":  3,
  "required_level":  "mid",
  "job_type":        "Full-time",
  "location":        "Remote"
}}

Rules:
- required_skills: max 15 technical skills explicitly mentioned
- nice_skills: skills listed as preferred or bonus
- required_years: integer, 0 if not mentioned
- required_level: junior / mid / senior / lead
- job_type: Full-time / Contract / Part-time
- location: Remote / Hybrid / On-site / city name
"""


def extract_job_requirements(job_text: str) -> Dict:
    # If no job text extracted use snippet from search
    if not job_text or len(job_text.strip()) < 100:
        logging.warning("[EXTRACT] Job text too short — using defaults")
        return {
            "required_skills": [],
            "nice_skills":     [],
            "required_years":  0,
            "required_level":  "junior",
            "job_type":        "Full-time",
            "location":        "Remote"
        }
    try:
        llm    = get_llm()
        prompt = JOB_EXTRACTION_PROMPT.format(text=job_text[:4000])
        response = llm.invoke(prompt)
        raw = response.content.strip()
        raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$',     '', raw, flags=re.MULTILINE)
        data = json.loads(raw)
        return {
            "required_skills": data.get("required_skills", []),
            "nice_skills":     data.get("nice_skills",     []),
            "required_years":  int(data.get("required_years", 0)),
            "required_level":  data.get("required_level",  "junior"),
            "job_type":        data.get("job_type",  "Full-time"),
            "location":        data.get("location",      "Remote")
        }
    except Exception as e:
        logging.error(f"[EXTRACT] Error: {e}")
        return {
            "required_skills": [],
            "nice_skills":     [],
            "required_years":  0,
            "required_level":  "junior",
            "job_type":        "Full-time",
            "location":        "Remote"
        }


# =========================================================
# STEP 5 — GENERATE COVER LETTER
# =========================================================

COVER_LETTER_PROMPT = """Write a short professional cover letter.

Candidate summary:  {summary}
Candidate skills:   {skills}
Applying for:       {job_title} at {company}
Key requirements:   {requirements}

Rules:
- Exactly 4 sentences
- Professional but warm tone
- Reference 2-3 specific matching skills
- No fake experience or placeholders
- No "Dear Sir/Madam" — use "Dear Hiring Manager"
"""


def generate_cover_letter(
    summary:      str,
    skills:       List[str],
    job_title:    str,
    company:      str,
    requirements: List[str]
) -> str:
    try:
        llm    = get_llm()
        prompt = COVER_LETTER_PROMPT.format(
            summary      = summary      or "Experienced software developer",
            skills       = ", ".join(skills[:10]),
            job_title    = job_title,
            company      = company,
            requirements = ", ".join(requirements[:10])
        )
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        logging.error(f"[COVER] Error: {e}")
        return (
            f"Dear Hiring Manager, I am excited to apply for "
            f"the {job_title} position at {company}. "
            f"My experience with {', '.join(skills[:2])} "
            f"aligns well with your requirements. "
            f"I look forward to discussing this opportunity."
        )


# =========================================================
# PROCESS SINGLE JOB — called in parallel
# =========================================================

def process_single_job(
    job:         Dict,
    resume_data: Dict
) -> Optional[Dict]:
    try:
        # Read full job posting
        job_text = read_job_posting(job["url"])

        # If reading fails use snippet from search as fallback
        if not job_text or len(job_text.strip()) < 100:
            job_text = job.get("snippet", "")
            logging.warning(
                f"[PROCESS] Using snippet fallback for: {job['url']}"
            )

        # If still nothing skip this job
        if not job_text or len(job_text.strip()) < 50:
            logging.warning(f"[PROCESS] Skipping — no content: {job['url']}")
            return None

        # Extract requirements
        extracted = extract_job_requirements(job_text)

        # If Gemini returned empty skills
        # try to extract from snippet directly
        if not extracted["required_skills"]:
            snippet   = job.get("snippet", "")
            title     = job.get("title",   "")
            combined  = f"{title} {snippet}"
            # Simple keyword match from title+snippet
            tech_keywords = [

        # =========================
        # Programming Languages
        # =========================
            "python", "java", "javascript", "typescript", "c", "c++",
            "c#", "go", "golang", "rust", "php", "ruby", "kotlin",
            "swift", "scala", "r", "perl", "matlab", "bash",

            # =========================
            # Frontend
            # =========================
            "html", "html5", "css", "css3", "sass", "scss",
            "tailwind", "bootstrap", "material ui",
            "react", "nextjs", "next.js", "redux",
            "vue", "vuejs", "angular", "svelte",
            "jquery", "ajax", "webpack", "vite",

            # =========================
            # Backend
            # =========================
            "nodejs", "node.js", "express", "expressjs",
            "django", "flask", "fastapi",
            "spring", "spring boot", "hibernate",
            "asp.net", ".net", "laravel",
            "rest api", "graphql", "microservices",

            # =========================
            # Databases
            # =========================
            "sql", "mysql", "postgresql", "sqlite",
            "mongodb", "redis", "firebase",
            "oracle", "mariadb", "dynamodb",
            "supabase", "cassandra", "neo4j",

            # =========================
            # DevOps / Cloud
            # =========================
            "docker", "kubernetes", "jenkins",
            "github actions", "gitlab ci/cd",
            "terraform", "ansible",
            "aws", "azure", "gcp",
            "aws s3", "aws ec2", "aws lambda",
            "azure openai", "aws bedrock",
            "cloud computing", "devops",
            "ci/cd", "linux", "ubuntu", "nginx",

            # =========================
            # Version Control
            # =========================
            "git", "github", "gitlab", "bitbucket",

            # =========================
            # Data Science
            # =========================
            "numpy", "pandas", "matplotlib",
            "seaborn", "plotly", "scipy",
            "statistics", "data analysis",
            "data visualization",
            "exploratory data analysis",
            "feature engineering",
            "data cleaning",
            "power bi", "tableau", "excel",

            # =========================
            # Machine Learning
            # =========================
            "machine learning",
            "supervised learning",
            "unsupervised learning",
            "reinforcement learning",
            "classification",
            "regression",
            "clustering",
            "decision trees",
            "random forest",
            "xgboost",
            "lightgbm",
            "catboost",
            "adaboost",
            "svm",
            "knn",
            "naive bayes",
            "linear regression",
            "logistic regression",
            "scikit-learn",
            "mlflow",

            # =========================
            # Deep Learning
            # =========================
            "deep learning", "ann", "cnn", "rnn",
            "lstm", "gru", "dnn",
            "transformers",
            "transfer learning",
            "computer vision",
            "opencv",
            "tensorflow",
            "keras",
            "pytorch",
            "huggingface",
            "efficientnet",
            "resnet",
            "yolo",
            "object detection",
            "image classification",

            # =========================
            # NLP
            # =========================
            "nlp",
            "natural language processing",
            "text classification",
            "sentiment analysis",
            "named entity recognition",
            "tokenization",
            "text preprocessing",
            "word2vec",
            "tf-idf",
            "bert",
            "roberta",
            "distilbert",
            "spacy",
            "nltk",
            "gensim",

            # =========================
            # Generative AI / LLM
            # =========================
            "generative ai",
            "gen ai",
            "large language models",
            "llm",
            "gpt",
            "chatgpt",
            "claude",
            "gemini",
            "llama",
            "mistral",
            "prompt engineering",
            "fine tuning",
            "rag",
            "retrieval augmented generation",
            "embeddings",
            "vector database",
            "semantic search",
            "reranking",

            # =========================
            # Agentic AI
            # =========================
            "agentic ai",
            "ai agents",
            "autonomous agents",
            "multi agent systems",
            "langchain",
            "langgraph",
            "crewai",
            "autogen",
            "semantic kernel",
            "tool calling",
            "function calling",
            "memory systems",
            "planning agents",

            # =========================
            # Vector Databases
            # =========================
            "pinecone",
            "faiss",
            "chromadb",
            "weaviate",
            "qdrant",
            "milvus",

            # =========================
            # OCR / Document AI
            # =========================
            "ocr",
            "tesseract",
            "document ai",
            "azure document intelligence",
            "pdf parsing",
            "document processing",

            # =========================
            # APIs & Deployment
            # =========================
            "api development",
            "restful apis",
            "deployment",
            "vercel",
            "netlify",
            "render",
            "heroku",
            "streamlit",

            # =========================
            # Cybersecurity
            # =========================
            "cybersecurity",
            "ethical hacking",
            "network security",
            "penetration testing",
            "wireshark",
            "metasploit",

            # =========================
            # Mobile Development
            # =========================
            "android",
            "flutter",
            "react native",
            "ios",

            # =========================
            # IoT
            # =========================
            "iot",
            "arduino",
            "raspberry pi",
            "sensors",
            "microcontrollers",

            # =========================
            # Software Engineering
            # =========================
            "oop",
            "data structures",
            "algorithms",
            "system design",
            "design patterns",
            "agile",
            "scrum",
            "jira",

            # =========================
            # Testing
            # =========================
            "pytest",
            "unit testing",
            "selenium",
            "postman",

            # =========================
            # Misc
            # =========================
            "web scraping",
            "beautifulsoup",
            "selenium automation",
            "automation testing",
            "multithreading",
            "socket programming"
            ]
            found = [
                kw for kw in tech_keywords
                if kw.lower() in combined.lower()
            ]
            if found:
                extracted["required_skills"] = found
                logging.info(
                    f"[PROCESS] Keyword fallback found: {found}"
                )

        # Score against candidate
        score_result = score_job(
            candidate_skills = resume_data.get("skills",           []),
            required_skills  = extracted["required_skills"],
            nice_skills      = extracted["nice_skills"],
            candidate_years  = resume_data.get("years_experience", 0),
            required_years   = extracted["required_years"],
            candidate_level  = resume_data.get("seniority",       "mid"),
            required_level   = extracted["required_level"],
            remote_friendly  = True,
            use_ai_reasoning = False
        )

        company = extract_company_from_title(job.get("title", ""))

        return {
            "title":          job["title"],
            "company":        company,
            "url":            job["url"],
            "location":       extracted["location"],
            "job_type":       extracted["job_type"],
            "score":          score_result["score"],
            "label":          score_result["label"],
            "color":          score_result["color"],
            "matched_skills": score_result["matched_skills"],
            "missing_skills": score_result["missing_skills"],
            "reasoning":      score_result.get("reasoning", ""),
            "cover_letter":   ""   # generated on demand only to reduce api calls and costs (improve performance)
            }


    except Exception as e:
        logging.error(f"[PROCESS] Job error: {e}")
        return None


# =========================================================
# MAIN ENGINE
# =========================================================

def run_job_hunter(
    user_message: str,
    resume_data:  Dict
) -> List[Dict]:

    logging.info("=" * 55)
    logging.info("AI JOB HUNTER — STARTED")
    logging.info("=" * 55)

    # ── Phase 1: Build queries ─────────────────────────────
    queries = build_queries(user_message, resume_data)
    logging.info(f"[ENGINE] Queries: {queries}")

    # ── Phase 2: Search all queries ────────────────────────
    all_raw_jobs = []
    for query in queries:
        results = search_jobs(query)
        all_raw_jobs.extend(results)

    if not all_raw_jobs:
        logging.warning("[ENGINE] No jobs found from search")
        return []

    # ── Phase 3: Deduplicate ───────────────────────────────
    unique_jobs = remove_duplicate_jobs(all_raw_jobs)

    # ── Phase 4: Limit to top 8 before processing ─────────
    jobs_to_process = unique_jobs[:8]
    logging.info(f"[ENGINE] Processing {len(jobs_to_process)} jobs")

    # ── Phase 5: Parallel processing ──────────────────────
    raw_results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                process_single_job, job, resume_data
            ): job
            for job in jobs_to_process
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    raw_results.append(result)
            except Exception as e:
                logging.error(f"[ENGINE] Thread error: {e}")

    # ── Phase 6: Validate with Pydantic ───────────────────
    validated_results = []
    for job in raw_results:
        try:
            validated = JobResult(**job)
            validated_results.append(validated.model_dump())
        except ValidationError as e:
            logging.warning(f"[ENGINE] Validation failed: {e}")
            continue

    # ── Phase 7: Filter low scores ─────────────────────────
    filtered = [
        j for j in validated_results
        if j["score"] >= 30
    ]

    # ── Phase 8: Sort by score descending ─────────────────
    filtered.sort(key=lambda x: x["score"], reverse=True)

    final = filtered[:6]

    logging.info("=" * 55)
    logging.info(f"AI JOB HUNTER — DONE | {len(final)} jobs returned")
    logging.info("=" * 55)

    return final


# =========================================================
# WRAPPER — keeps app.py interface consistent
# =========================================================

def run_agent(
    user_message: str,
    resume_data:  Dict = None
) -> List[Dict]:
    """
    Entry point called by app.py.
    Wraps run_job_hunter with safe defaults.
    """
    resume_data = resume_data or {}
    return run_job_hunter(user_message, resume_data)


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    sample_resume = {
        "skills":           ["python", "flask", "machine learning",
                             "mongodb", "docker", "langchain"],
        "job_titles":       ["AI Engineer", "ML Engineer"],
        "seniority":        "mid",
        "years_experience": 3,
        "education":        "B.Tech Computer Science",
        "summary":          "3 years Python ML engineer"
    }

    results = run_agent(
        user_message = "Find remote AI Engineer jobs",
        resume_data  = sample_resume
    )

    print(json.dumps(results, indent=2))