import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from backend.database import db


# =========================================================
# USER MODEL
# =========================================================

class User(db.Model):
    __tablename__ = 'users'

    id            = db.Column(db.Integer,     primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(150), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)

    # Relationships
    resumes       = db.relationship('Resume', backref='user', lazy=True)
    searches      = db.relationship('Search', backref='user', lazy=True)

    # ── Password helpers ──────────────────────────────────

    def set_password(self, plain_password: str) -> None:
        """Hash and store password — never store plain text."""
        self.password_hash = generate_password_hash(plain_password)

    def check_password(self, plain_password: str) -> bool:
        """Verify password against stored hash."""
        return check_password_hash(self.password_hash, plain_password)

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "name":       self.name,
            "email":      self.email,
            "created_at": self.created_at.isoformat()
        }


# =========================================================
# RESUME MODEL
# =========================================================

class Resume(db.Model):
    __tablename__ = 'resumes'

    id               = db.Column(db.Integer,     primary_key=True)
    user_id          = db.Column(db.Integer,     db.ForeignKey('users.id'),
                                 nullable=False)
    filename         = db.Column(db.String(255), nullable=False)
    raw_text         = db.Column(db.Text,        nullable=False)
    clean_text       = db.Column(db.Text)
    skills           = db.Column(db.Text)        # JSON string
    job_titles       = db.Column(db.Text)        # JSON string
    seniority        = db.Column(db.String(50))
    years_experience = db.Column(db.Integer,     default=0)
    education        = db.Column(db.Text)
    summary          = db.Column(db.Text)
    uploaded_at      = db.Column(db.DateTime,    default=datetime.utcnow)

    # Relationships
    searches         = db.relationship('Search', backref='resume', lazy=True)

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "user_id":          self.user_id,
            "filename":         self.filename,
            "skills":           json.loads(self.skills     or '[]'),
            "job_titles":       json.loads(self.job_titles or '[]'),
            "seniority":        self.seniority,
            "years_experience": self.years_experience,
            "education":        self.education,
            "summary":          self.summary,
            "uploaded_at":      self.uploaded_at.isoformat()
        }


# =========================================================
# SEARCH MODEL
# =========================================================

class Search(db.Model):
    __tablename__ = 'searches'

    id            = db.Column(db.Integer,     primary_key=True)
    user_id       = db.Column(db.Integer,     db.ForeignKey('users.id'),
                              nullable=False)
    resume_id     = db.Column(db.Integer,     db.ForeignKey('resumes.id'),
                              nullable=True)
    search_query          = db.Column(db.String(500), nullable=False)
    results_count = db.Column(db.Integer,     default=0)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)

    # Relationships
    jobs          = db.relationship('Job', backref='search', lazy=True)

    def to_dict(self) -> dict:
        return {
            "id":            self.id,
            "user_id":       self.user_id,
            "query":         self.search_query ,
            "results_count": self.results_count,
            "created_at":    self.created_at.isoformat()
        }


# =========================================================
# JOB MODEL
# =========================================================

class Job(db.Model):
    __tablename__ = 'jobs'

    id              = db.Column(db.Integer,      primary_key=True)
    search_id       = db.Column(db.Integer,      db.ForeignKey('searches.id'),
                                nullable=False)
    title           = db.Column(db.String(255))
    company         = db.Column(db.String(255))
    url             = db.Column(db.String(1000))
    location        = db.Column(db.String(255),  default='Remote')
    job_type        = db.Column(db.String(100),  default='Full-time')
    score           = db.Column(db.Integer,      default=0)
    score_label     = db.Column(db.String(50))
    score_color     = db.Column(db.String(20))
    matched_skills  = db.Column(db.Text)         # JSON string
    missing_skills  = db.Column(db.Text)         # JSON string
    cover_letter    = db.Column(db.Text)
    reasoning       = db.Column(db.Text)
    raw_description = db.Column(db.Text)
    status          = db.Column(db.String(50),   default='found')
    created_at      = db.Column(db.DateTime,     default=datetime.utcnow)

    # Relationships
    tracker         = db.relationship('TrackerItem', backref='job',
                                      uselist=False, lazy=True)

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "search_id":      self.search_id,
            "title":          self.title,
            "company":        self.company,
            "url":            self.url,
            "location":       self.location,
            "job_type":       self.job_type,
            "score":          self.score,
            "score_label":    self.score_label,
            "score_color":    self.score_color,
            "matched_skills": json.loads(self.matched_skills or '[]'),
            "missing_skills": json.loads(self.missing_skills or '[]'),
            "cover_letter":   self.cover_letter,
            "reasoning":      self.reasoning,
            "status":         self.status,
            "created_at":     self.created_at.isoformat()
        }


# =========================================================
# TRACKER MODEL
# =========================================================

class TrackerItem(db.Model):
    __tablename__ = 'tracker'

    id         = db.Column(db.Integer,    primary_key=True)
    job_id     = db.Column(db.Integer,    db.ForeignKey('jobs.id'),
                           nullable=False)
    status     = db.Column(db.String(50), default='saved')
    notes      = db.Column(db.Text)
    updated_at = db.Column(db.DateTime,   default=datetime.utcnow,
                           onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        job_data = self.job.to_dict() if self.job else {}
        return {
            "id":         self.id,
            "job_id":     self.job_id,
            "status":     self.status,
            "notes":      self.notes,
            "updated_at": self.updated_at.isoformat(),
            "job":        job_data
        }