import os
import json
import re
import logging
from typing import List, Dict, Any

import pdfplumber
import spacy
import google.generativeai as genai

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# =========================================================
# ENV + MODEL SETUP
# =========================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

_gemini = genai.GenerativeModel("gemini-2.0-flash")
_nlp    = spacy.load("en_core_web_md")


# =========================================================
# PYDANTIC SCHEMA
# =========================================================

class ResumeSchema(BaseModel):
    skills:           List[str]
    job_titles:       List[str]
    years_experience: int
    seniority:        str
    education:        str
    summary:          str


# =========================================================
# SKILL NORMALIZATION MAP
# =========================================================

SKILL_NORMALIZATION = {
    "js":               "javascript",
    "node":             "nodejs",
    "node.js":          "nodejs",
    "py":               "python",
    "postgres":         "postgresql",
    "mongo":            "mongodb",
    "machine-learning": "machine learning",
    "deep-learning":    "deep learning",
    "gen ai":           "generative ai",
    "llms":             "llm",
    "ts":               "typescript",
    "k8s":              "kubernetes",
    "tf":               "tensorflow",
}

FALLBACK_SKILLS = [

    # =========================
    # Programming Languages
    # =========================
    "python", "java", "javascript", "typescript", "c", "c++",
    "c#", "go", "golang", "rust", "php", "ruby", "kotlin",
    "swift", "scala", "r", "perl", "matlab", "bash", "shell scripting",

    # =========================
    # Frontend Development
    # =========================
    "html", "html5", "css", "css3", "sass", "scss",
    "tailwind css", "bootstrap", "material ui",
    "react", "nextjs", "next.js", "redux",
    "vue", "vuejs", "angular", "svelte",
    "jquery", "ajax", "webpack", "vite",

    # =========================
    # Backend Development
    # =========================
    "nodejs", "node.js", "express", "expressjs",
    "django", "flask", "fastapi",
    "spring boot", "spring", "hibernate",
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
    # DevOps & Cloud
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
    "data visualization", "exploratory data analysis",
    "feature engineering", "data cleaning",
    "power bi", "tableau", "excel",

    # =========================
    # Machine Learning
    # =========================
    "machine learning", "supervised learning",
    "unsupervised learning", "reinforcement learning",
    "classification", "regression",
    "clustering", "decision trees",
    "random forest", "xgboost",
    "lightgbm", "catboost",
    "adaboost", "svm", "knn",
    "naive bayes", "linear regression",
    "logistic regression",
    "scikit-learn", "mlflow",

    # =========================
    # Deep Learning
    # =========================
    "deep learning", "ann", "cnn", "rnn",
    "lstm", "gru", "dnn",
    "transformers", "transfer learning",
    "computer vision", "opencv",
    "tensorflow", "keras", "pytorch",
    "huggingface", "hugging face",
    "efficientnet", "resnet", "yolo",
    "object detection", "image classification",

    # =========================
    # NLP
    # =========================
    "nlp", "natural language processing",
    "text classification", "sentiment analysis",
    "named entity recognition",
    "tokenization", "text preprocessing",
    "word2vec", "tf-idf", "bert",
    "roberta", "distilbert",
    "spacy", "nltk", "gensim",

    # =========================
    # Generative AI / LLM
    # =========================
    "generative ai", "gen ai",
    "large language models", "llm",
    "gpt", "chatgpt", "claude",
    "gemini", "llama", "mistral",
    "prompt engineering",
    "fine tuning", "rag",
    "retrieval augmented generation",
    "embeddings", "vector database",
    "semantic search", "reranking",
    "hallucination mitigation",

    # =========================
    # Agentic AI
    # =========================
    "agentic ai", "ai agents",
    "autonomous agents",
    "multi agent systems",
    "langchain", "langgraph",
    "crewai", "autogen",
    "semantic kernel",
    "tool calling", "function calling",
    "memory systems",
    "planning agents",

    # =========================
    # Vector Databases
    # =========================
    "pinecone", "faiss",
    "chromadb", "weaviate",
    "qdrant", "milvus",

    # =========================
    # OCR & Document AI
    # =========================
    "ocr", "tesseract",
    "document ai",
    "azure document intelligence",
    "pdf parsing",
    "document processing",

    # =========================
    # APIs & Deployment
    # =========================
    "api development",
    "restful apis",
    "fastapi", "flask api",
    "deployment", "vercel",
    "netlify", "render",
    "heroku", "streamlit",

    # =========================
    # Cybersecurity
    # =========================
    "cybersecurity", "ethical hacking",
    "network security", "penetration testing",
    "wireshark", "metasploit",

    # =========================
    # Mobile Development
    # =========================
    "android", "flutter",
    "react native", "ios",

    # =========================
    # IoT
    # =========================
    "iot", "arduino",
    "raspberry pi",
    "sensors", "microcontrollers",

    # =========================
    # Software Engineering
    # =========================
    "oop", "data structures",
    "algorithms", "system design",
    "design patterns",
    "agile", "scrum", "jira",

    # =========================
    # Testing
    # =========================
    "pytest", "unit testing",
    "selenium", "postman",

    # =========================
    # Misc
    # =========================
    "web scraping", "beautifulsoup",
    "selenium automation",
    "automation testing",
    "multithreading",
    "socket programming"
]


# =========================================================
# PROMPT TEMPLATE
# =========================================================

PROMPT_TEMPLATE = """You are an ATS resume parser.
Extract structured information from the resume below.
Return ONLY valid JSON. No markdown. No explanation.

Resume:
{text}

Return exactly this JSON:
{{
  "skills": ["skill1", "skill2"],
  "job_titles": ["title1", "title2"],
  "years_experience": 0,
  "seniority": "junior",
  "education": "degree name",
  "summary": "short profile summary"
}}

Rules:
- skills: max 25, no duplicates, normalize abbreviations
  (js→javascript, py→python, node→nodejs, k8s→kubernetes)
- job_titles: 2-3 most suitable roles based on experience
- years_experience: integer only, 0 if fresher
- seniority: junior(0-2yr) mid(3-5yr) senior(6+yr)
- education: highest degree only, one line
- summary: max 20 words describing candidate profile
"""


# =========================================================
# STEP 1 — PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
        return ""
    return text.strip()


# =========================================================
# STEP 2 — SPACY TEXT CLEANING
# =========================================================

def clean_resume_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    # Remove noise before SpaCy
    raw_text = re.sub(r'\S+@\S+',                     '', raw_text)
    raw_text = re.sub(r'http\S+|www\.\S+',            '', raw_text)
    raw_text = re.sub(r'\+?\d[\d\s\-().]{8,}\d',      '', raw_text)
    raw_text = re.sub(r'\b(page\s*\d+|pg\s*\d+)\b',  '', raw_text,
                      flags=re.IGNORECASE)
    raw_text = re.sub(r'[^\w\s\.,:/|()\-]',           ' ', raw_text)
    raw_text = re.sub(r'\s{2,}',                      ' ', raw_text)

    # SpaCy entity filtering
    doc = _nlp(raw_text[:50000])

    skip_entities = {"PERSON", "GPE", "LOC", "MONEY", "CARDINAL"}
    clean_tokens  = []

    for token in doc:
        if token.is_space:
            continue
        if token.ent_type_ in skip_entities:
            continue
        if token.is_punct and token.text not in {'.', ',', '/', ':'}:
            continue
        clean_tokens.append(token.text)

    clean_text = " ".join(clean_tokens)
    clean_text = re.sub(r'\s{2,}', ' ', clean_text).strip()

    # Cap at 3000 chars to save Gemini tokens
    return clean_text[:3000]


# =========================================================
# STEP 3 — SECTION EXTRACTION
# =========================================================

def extract_resume_sections(text: str) -> str:
    """
    Try to extract key sections.
    Falls back to full clean_text if sections are too short.
    """
    lower_text = text.lower()

    patterns = {
        "skills":     r'(?:skills|technical skills|core competencies)'
                      r'(.*?)(?=experience|projects|education|$)',
        "experience": r'(?:experience|work experience)'
                      r'(.*?)(?=projects|education|skills|$)',
        "projects":   r'(?:projects|personal projects)'
                      r'(.*?)(?=experience|education|skills|$)',
        "education":  r'(?:education|academics)'
                      r'(.*?)(?=experience|projects|skills|$)',
    }

    sections = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, lower_text, re.DOTALL)
        sections[key] = match.group(1).strip() if match else ""

    combined = "\n".join([
        sections.get("skills",     ""),
        sections.get("experience", ""),
        sections.get("projects",   ""),
        sections.get("education",  ""),
    ]).strip()

    # Only use section extraction if it captured enough content
    # Otherwise fall back to full clean text
    if len(combined) > 500:
        return combined[:3000]

    return text[:3000]


# =========================================================
# STEP 4 — SKILL NORMALIZATION
# =========================================================

def normalize_skills(skills: List[str]) -> List[str]:
    normalized = []
    for skill in skills:
        skill = skill.lower().strip()
        skill = SKILL_NORMALIZATION.get(skill, skill)
        if skill and skill not in normalized:
            normalized.append(skill)
    return normalized[:25]


# =========================================================
# STEP 5 — GEMINI EXTRACTION
# =========================================================

def extract_with_gemini(llm_input: str) -> Dict[str, Any]:
    try:
        prompt   = PROMPT_TEMPLATE.format(text=llm_input)
        response = _gemini.generate_content(
            prompt,
            generation_config={
                "temperature":         0.1,
                "response_mime_type":  "application/json"
            }
        )

        raw         = response.text.strip()
        parsed_json = json.loads(raw)

        # Pydantic validation
        validated   = ResumeSchema(**parsed_json)
        parsed_data = validated.model_dump()

        # Normalize skills after Gemini
        parsed_data["skills"] = normalize_skills(
            parsed_data.get("skills", [])
        )

        return parsed_data

    except ValidationError as e:
        logging.error(f"Pydantic validation failed: {e}")
        return fallback_parser(llm_input)

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON from Gemini: {e}")
        return fallback_parser(llm_input)

    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return fallback_parser(llm_input)


# =========================================================
# FALLBACK PARSER
# =========================================================

def fallback_parser(text: str) -> Dict[str, Any]:
    text_lower      = text.lower()
    detected_skills = [s for s in FALLBACK_SKILLS if s in text_lower]
    detected_skills = normalize_skills(detected_skills)

    return {
        "skills":           detected_skills,
        "job_titles":       ["Software Engineer"],
        "years_experience": 0,
        "seniority":        "junior",
        "education":        "Not extracted",
        "summary":          "Resume parsed using fallback method"
    }


# =========================================================
# MAIN FUNCTION — called by app.py
# =========================================================

def parse_resume(file_path: str) -> Dict[str, Any]:

    logging.info("[Parser] Step 1 — Extracting PDF text...")
    raw_text = extract_pdf_text(file_path)

    if not raw_text:
        return {"error": "Could not extract text. Use a text-based PDF."}

    logging.info("[Parser] Step 2 — Cleaning with SpaCy...")
    clean_text = clean_resume_text(raw_text)

    if not clean_text:
        return {"error": "Resume text cleaning failed."}

    logging.info("[Parser] Step 3 — Extracting sections...")
    llm_input = extract_resume_sections(clean_text)

    logging.info("[Parser] Step 4 — Analyzing with Gemini...")
    parsed_data = extract_with_gemini(llm_input)

    return {
        # Raw data
        "raw_text":          raw_text,
        "clean_text":        clean_text,

        # Structured output
        "skills":            parsed_data.get("skills",           []),
        "job_titles":        parsed_data.get("job_titles",       ["Software Engineer"]),
        "years_experience":  parsed_data.get("years_experience", 0),
        "seniority":         parsed_data.get("seniority",        "junior"),
        "education":         parsed_data.get("education",        ""),
        "summary":           parsed_data.get("summary",          ""),

        # JSON strings for DB storage
        "skills_json":       json.dumps(parsed_data.get("skills",     [])),
        "titles_json":       json.dumps(parsed_data.get("job_titles",  []))
    }


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":
    result = parse_resume("sample_resume.pdf")
    print(json.dumps(result, indent=2))