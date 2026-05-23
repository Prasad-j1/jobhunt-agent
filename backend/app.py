import os
import json
import logging
from functools import wraps
from datetime import datetime
from flask import send_from_directory

from flask import (
    Flask, request, jsonify,
    session, redirect, url_for,
    render_template
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# =========================================================
# APP INIT
# =========================================================

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(
    __name__,
    template_folder = '../frontend/templates',
    static_folder   = '../frontend/static'
)

app.config['SECRET_KEY']            = os.getenv(
    'FLASK_SECRET_KEY', 'dev-secret-change-in-prod'
)
app.config['MAX_CONTENT_LENGTH']    = 16 * 1024 * 1024  # 16MB
app.config['SESSION_COOKIE_HTTPONLY']  = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

CORS(app, supports_credentials=True)

# =========================================================
# DATABASE INIT
# =========================================================

from backend.database import init_db, db
init_db(app)

from backend.models import User, Resume, Search, Job, TrackerItem

# =========================================================
# UPLOAD FOLDER
# =========================================================

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__), '..', 'uploads'
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def allowed_file(filename: str) -> bool:
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() == 'pdf'
    )


def success_response(data: dict, status: int = 200):
    return jsonify({"success": True,  **data}), status


def error_response(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


# =========================================================
# AUTH DECORATOR
# =========================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            # API request → return JSON error
            if request.path.startswith('/api/'):
                return error_response("Unauthorized. Please login.", 401)
            # Page request → redirect to login
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


def get_current_user() -> User:
    """Get logged in user from session."""
    return User.query.get(session['user_id'])


# =========================================================
# PAGE ROUTES
# =========================================================

@app.route('/')
def login_page():
    """Serve login/register page."""
    if 'user_id' in session:
        return redirect(url_for('dashboard_page'))
    return render_template('auth.html', page='auth')


@app.route('/dashboard')
@login_required
def dashboard_page():
    """Serve main dashboard (protected)."""
    return render_template('dashboard.html', page='dashboard')


# =========================================================
# AUTH ROUTES
# =========================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """
    Register a new user account.
    Body: { name, email, password }
    """
    data     = request.get_json(silent=True)
    if not data:
        return error_response("Invalid request body.")

    name     = data.get('name',     '').strip()
    email    = data.get('email',    '').strip().lower()
    password = data.get('password', '').strip()

    # ── Validation ────────────────────────────────────────
    if not name:
        return error_response("Name is required.")
    if len(name) < 2:
        return error_response("Name must be at least 2 characters.")
    if not email or '@' not in email:
        return error_response("Valid email is required.")
    if not password:
        return error_response("Password is required.")
    if len(password) < 6:
        return error_response("Password must be at least 6 characters.")

    # ── Check duplicate ───────────────────────────────────
    existing = User.query.filter_by(email=email).first()
    if existing:
        return error_response("Email already registered. Please login.")

    # ── Create user ───────────────────────────────────────
    try:
        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # Auto login after register
        session['user_id'] = user.id
        session['name']    = user.name
        session.permanent  = True

        logging.info(f"[AUTH] New user registered: {email}")

        return success_response({
            "message": f"Welcome to AI Job Hunter, {user.name}!",
            "user":    user.to_dict()
        }, 201)

    except Exception as e:
        db.session.rollback()
        logging.error(f"[AUTH] Register error: {e}")
        return error_response("Registration failed. Try again.", 500)


@app.route('/api/auth/login', methods=['POST'])
def login():
    """
    Login with email and password.
    Body: { email, password }
    """
    data     = request.get_json(silent=True)
    if not data:
        return error_response("Invalid request body.")

    email    = data.get('email',    '').strip().lower()
    password = data.get('password', '').strip()

    if not email or not password:
        return error_response("Email and password are required.")

    # ── Find user ─────────────────────────────────────────
    user = User.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        return error_response("Invalid email or password.", 401)

    # ── Set session ───────────────────────────────────────
    session['user_id'] = user.id
    session['name']    = user.name
    session.permanent  = True

    logging.info(f"[AUTH] User logged in: {email}")

    # ── Fetch resume status ───────────────────────────────
    resume = Resume.query.filter_by(
        user_id=user.id
    ).order_by(Resume.uploaded_at.desc()).first()

    return success_response({
        "message": f"Welcome back, {user.name}!",
        "user":    user.to_dict(),
        "has_resume": resume is not None,
        "resume":  resume.to_dict() if resume else None
    })


@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    """Clear session and logout."""
    name = session.get('name', 'User')
    session.clear()
    logging.info(f"[AUTH] User logged out: {name}")
    return success_response({"message": "Logged out successfully."})


@app.route('/api/auth/me', methods=['GET'])
@login_required
def me():
    """
    Get current logged in user profile.
    Also returns resume + stats.
    """
    user = get_current_user()
    if not user:
        session.clear()
        return error_response("User not found.", 404)

    # Resume
    resume = Resume.query.filter_by(
        user_id=user.id
    ).order_by(Resume.uploaded_at.desc()).first()

    # Stats
    total_searches = Search.query.filter_by(user_id=user.id).count()
    total_jobs     = Job.query.join(Search).filter(
        Search.user_id == user.id
    ).count()
    total_applied  = TrackerItem.query.join(Job).join(Search).filter(
        Search.user_id == user.id,
        TrackerItem.status == 'applied'
    ).count()
    total_saved    = TrackerItem.query.join(Job).join(Search).filter(
        Search.user_id == user.id,
        TrackerItem.status == 'saved'
    ).count()

    return success_response({
        "user":       user.to_dict(),
        "has_resume": resume is not None,
        "resume":     resume.to_dict() if resume else None,
        "stats": {
            "total_searches": total_searches,
            "total_jobs":     total_jobs,
            "total_applied":  total_applied,
            "total_saved":    total_saved
        }
    })


# =========================================================
# RESUME ROUTES
# =========================================================

@app.route('/api/resume/upload', methods=['POST'])
@login_required
def upload_resume():
    """
    Upload and parse a resume PDF.
    Form data: resume (file)
    """
    if 'resume' not in request.files:
        return error_response("No file uploaded.")

    file = request.files['resume']

    if not file or not file.filename:
        return error_response("No file selected.")

    if not allowed_file(file.filename):
        return error_response("Only PDF files are allowed.")

    user = get_current_user()

    # ── Save file temporarily ─────────────────────────────
    filename  = secure_filename(file.filename)
    file_path = os.path.join(
        UPLOAD_FOLDER,
        f"user_{user.id}_{filename}"
    )
    file.save(file_path)

    try:
        # ── Parse resume ──────────────────────────────────
        from backend.resume_parser import parse_resume
        parsed = parse_resume(file_path)

        if "error" in parsed:
            return error_response(parsed["error"])

        # ── Save to DB ────────────────────────────────────
        resume = Resume(
            user_id          = user.id,
            filename         = filename,
            raw_text         = parsed["raw_text"],
            clean_text       = parsed.get("clean_text", ""),
            skills           = parsed["skills_json"],
            job_titles       = parsed["titles_json"],
            seniority        = parsed["seniority"],
            years_experience = parsed["years_experience"],
            education        = parsed.get("education", ""),
            summary          = parsed.get("summary",   "")
        )
        db.session.add(resume)
        db.session.commit()

        logging.info(
            f"[RESUME] Uploaded for user {user.id}: "
            f"{len(parsed['skills'])} skills found"
        )

        return success_response({
            "message":   "Resume uploaded and parsed successfully!",
            "resume_id": resume.id,
            "resume":    resume.to_dict()
        }, 201)

    except Exception as e:
        db.session.rollback()
        logging.error(f"[RESUME] Upload error: {e}")
        return error_response(f"Resume processing failed: {str(e)}", 500)

    finally:
        # Always delete temp file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route('/api/resume/current', methods=['GET'])
@login_required
def get_current_resume():
    """Get the most recent resume for logged in user."""
    user   = get_current_user()
    resume = Resume.query.filter_by(
        user_id=user.id
    ).order_by(Resume.uploaded_at.desc()).first()

    if not resume:
        return error_response("No resume uploaded yet.", 404)

    return success_response({"resume": resume.to_dict()})


# =========================================================
# SEARCH ROUTES
# =========================================================

@app.route('/api/search', methods=['POST'])
@login_required
def search_jobs_route():
    """
    Run AI job search.
    Body: { query, resume_id (optional) }
    """
    data  = request.get_json(silent=True)
    if not data:
        return error_response("Invalid request body.")

    query = data.get('query', '').strip()
    if not query:
        return error_response("Search query is required.")

    user = get_current_user()

    # ── Get resume data ───────────────────────────────────
    resume      = Resume.query.filter_by(
        user_id=user.id
    ).order_by(Resume.uploaded_at.desc()).first()

    resume_data = None
    if resume:
        resume_data = {
            "skills":           json.loads(resume.skills     or '[]'),
            "job_titles":       json.loads(resume.job_titles or '[]'),
            "seniority":        resume.seniority        or "mid",
            "years_experience": resume.years_experience or 0,
            "education":        resume.education        or "",
            "summary":          resume.summary          or ""
        }

    # ── Create search record ──────────────────────────────
    search_record = Search(
        user_id   = user.id,
        resume_id = resume.id if resume else None,
        search_query      = query
    )
    db.session.add(search_record)
    db.session.commit()

    try:
        # ── Run agent ─────────────────────────────────────
        logging.info(
            f"[SEARCH] User {user.id} searching: {query}"
        )
        from backend.agent import run_agent
        jobs = run_agent(
            user_message = query,
            resume_data  = resume_data
        )

        if not jobs:
            return error_response(
                "No jobs found. Try a different search query.", 404
            )

        # ── Save jobs to DB ───────────────────────────────
        saved_jobs = []
        for j in jobs:
            job = Job(
                search_id       = search_record.id,
                title           = j.get('title',          ''),
                company         = j.get('company',        ''),
                url             = j.get('url',            ''),
                location        = j.get('location',   'Remote'),
                job_type        = j.get('job_type', 'Full-time'),
                score           = j.get('score',           0),
                score_label     = j.get('label',          ''),
                score_color     = j.get('color',        'gray'),
                matched_skills  = json.dumps(
                    j.get('matched_skills', [])
                ),
                missing_skills  = json.dumps(
                    j.get('missing_skills', [])
                ),
                cover_letter    = j.get('cover_letter',   ''),
                reasoning       = j.get('reasoning',      ''),
                raw_description = j.get('description',    '')
            )
            db.session.add(job)
            db.session.flush()      # get job.id before commit
            j['id'] = job.id
            saved_jobs.append(j)

        # ── Update search count ───────────────────────────
        search_record.results_count = len(saved_jobs)
        db.session.commit()

        logging.info(
            f"[SEARCH] Completed: {len(saved_jobs)} jobs saved"
        )

        return success_response({
            "message":   f"Found {len(saved_jobs)} matching jobs!",
            "jobs":      saved_jobs,
            "search_id": search_record.id,
            "total":     len(saved_jobs)
        })

    except Exception as e:
        db.session.rollback()
        search_record.results_count = 0
        db.session.commit()
        logging.error(f"[SEARCH] Error: {e}")
        return error_response(f"Search failed: {str(e)}", 500)


@app.route('/api/jobs', methods=['GET'])
@login_required
def get_jobs():
    """
    Get jobs by search_id or latest 50 for user.
    Query params: search_id (optional)
    """
    user      = get_current_user()
    search_id = request.args.get('search_id', type=int)

    if search_id:
        # Verify search belongs to user
        search = Search.query.filter_by(
            id=search_id, user_id=user.id
        ).first()
        if not search:
            return error_response("Search not found.", 404)
        jobs = Job.query.filter_by(
            search_id=search_id
        ).order_by(Job.score.desc()).all()
    else:
        jobs = Job.query.join(Search).filter(
            Search.user_id == user.id
        ).order_by(Job.score.desc()).limit(50).all()

    return success_response({
        "jobs":  [j.to_dict() for j in jobs],
        "total": len(jobs)
    })

@app.route('/jobs')
@login_required
def jobs_page():
    return render_template('jobs.html')


@app.route('/tracker')
@login_required
def tracker_page():
    return render_template('tracker.html')


@app.route('/history')
@login_required
def history_page():
    return render_template('history.html')
# =========================================================
# TRACKER ROUTES
# =========================================================

@app.route('/api/tracker', methods=['GET'])
@login_required
def get_tracker():
    """Get all tracker items for logged in user."""
    user  = get_current_user()
    items = TrackerItem.query.join(Job).join(Search).filter(
        Search.user_id == user.id
    ).order_by(TrackerItem.updated_at.desc()).all()

    # Group by status
    grouped = {"saved": [], "applied": [], "rejected": []}
    for item in items:
        status = item.status
        if status in grouped:
            grouped[status].append(item.to_dict())

    return success_response({
        "tracker": grouped,
        "total":   len(items)
    })


@app.route('/api/tracker/update', methods=['POST'])
@login_required
def update_tracker():
    """
    Add or update job tracker status.
    Body: { job_id, status, notes (optional) }
    """
    data   = request.get_json(silent=True)
    if not data:
        return error_response("Invalid request body.")

    job_id = data.get('job_id')
    status = data.get('status', '').strip().lower()
    notes  = data.get('notes', '').strip()

    if not job_id:
        return error_response("job_id is required.")

    valid_statuses = ['saved', 'applied', 'rejected']
    if status not in valid_statuses:
        return error_response(
            f"Status must be one of: {', '.join(valid_statuses)}"
        )

    user = get_current_user()

    # Verify job belongs to this user
    job = Job.query.join(Search).filter(
        Job.id        == job_id,
        Search.user_id == user.id
    ).first()

    if not job:
        return error_response("Job not found.", 404)

    try:
        # Update job status
        job.status = status

        # Upsert tracker item
        item = TrackerItem.query.filter_by(job_id=job_id).first()
        if item:
            item.status     = status
            item.notes      = notes
            item.updated_at = datetime.utcnow()
        else:
            item = TrackerItem(
                job_id = job_id,
                status = status,
                notes  = notes
            )
            db.session.add(item)

        db.session.commit()

        logging.info(
            f"[TRACKER] Job {job_id} → {status} "
            f"by user {user.id}"
        )

        return success_response({
            "message": f"Job marked as {status}.",
            "item":    item.to_dict()
        })

    except Exception as e:
        db.session.rollback()
        logging.error(f"[TRACKER] Update error: {e}")
        return error_response("Tracker update failed.", 500)


# =========================================================
# HISTORY ROUTES
# =========================================================

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """Get search history for logged in user."""
    user     = get_current_user()
    searches = Search.query.filter_by(
        user_id=user.id
    ).order_by(Search.created_at.desc()).limit(20).all()

    return success_response({
        "searches": [s.to_dict() for s in searches],
        "total":    len(searches)
    })


@app.route('/api/history/clear', methods=['DELETE'])
@login_required
def clear_history():
    """Clear all search history for logged in user."""
    user = get_current_user()
    try:
        searches = Search.query.filter_by(user_id=user.id).all()
        for search in searches:
            # Delete associated jobs and tracker items
            for job in search.jobs:
                TrackerItem.query.filter_by(job_id=job.id).delete()
                db.session.delete(job)
            db.session.delete(search)

        db.session.commit()
        logging.info(f"[HISTORY] Cleared for user {user.id}")
        return success_response({"message": "History cleared."})

    except Exception as e:
        db.session.rollback()
        logging.error(f"[HISTORY] Clear error: {e}")
        return error_response("Failed to clear history.", 500)


# =========================================================
# COVER LETTER ROUTES
# =========================================================

@app.route('/api/cover-letter/<int:job_id>', methods=['GET'])
@login_required
def get_cover_letter(job_id: int):
    """Get cover letter for a specific job."""
    user = get_current_user()

    job = Job.query.join(Search).filter(
        Job.id         == job_id,
        Search.user_id == user.id
    ).first()

    if not job:
        return error_response("Job not found.", 404)

    return success_response({
        "job_id":       job.id,
        "job_title":    job.title,
        "company":      job.company,
        "cover_letter": job.cover_letter
    })


@app.route('/api/cover-letter/<int:job_id>', methods=['PUT'])
@login_required
def update_cover_letter(job_id: int):
    """
    Update cover letter for a job.
    Body: { cover_letter }
    """
    user = get_current_user()
    data = request.get_json(silent=True)

    if not data:
        return error_response("Invalid request body.")

    cover_letter = data.get('cover_letter', '').strip()
    if not cover_letter:
        return error_response("Cover letter content is required.")

    job = Job.query.join(Search).filter(
        Job.id         == job_id,
        Search.user_id == user.id
    ).first()

    if not job:
        return error_response("Job not found.", 404)

    try:
        job.cover_letter = cover_letter
        db.session.commit()
        return success_response({
            "message":      "Cover letter updated.",
            "cover_letter": job.cover_letter
        })
    except Exception as e:
        db.session.rollback()
        logging.error(f"[COVER] Update error: {e}")
        return error_response("Update failed.", 500)


@app.route('/api/cover-letter/<int:job_id>/generate', methods=['POST'])
@login_required
def generate_cover_letter_route(job_id: int):
    """
    Generate cover letter on demand when user clicks button.
    Only called when user explicitly requests it.
    """
    user = get_current_user()

    job = Job.query.join(Search).filter(
        Job.id         == job_id,
        Search.user_id == user.id
    ).first()

    if not job:
        return error_response("Job not found.", 404)

    # If already generated return cached version
    if job.cover_letter and len(job.cover_letter.strip()) > 50:
        logging.info(f"[COVER] Returning cached letter for job {job_id}")
        return success_response({
            "cover_letter": job.cover_letter,
            "job_title":    job.title,
            "company":      job.company,
            "cached":       True
        })

    # Get resume data for personalization
    resume = Resume.query.filter_by(
        user_id=user.id
    ).order_by(Resume.uploaded_at.desc()).first()

    resume_data = {}
    if resume:
        resume_data = {
            "skills":    json.loads(resume.skills     or '[]'),
            "summary":   resume.summary               or "",
            "seniority": resume.seniority             or "junior"
        }

    try:
        from backend.agent import generate_cover_letter

        matched = json.loads(job.matched_skills or '[]')
        missing = json.loads(job.missing_skills or '[]')
        requirements = matched + missing

        cover_letter = generate_cover_letter(
            summary      = resume_data.get("summary", ""),
            skills       = resume_data.get("skills",  []),
            job_title    = job.title   or "this position",
            company      = job.company or "your company",
            requirements = requirements[:10]
        )

        # Cache it in DB so next click is instant
        job.cover_letter = cover_letter
        db.session.commit()

        logging.info(f"[COVER] Generated for job {job_id}")

        return success_response({
            "cover_letter": cover_letter,
            "job_title":    job.title,
            "company":      job.company,
            "cached":       False
        })

    except Exception as e:
        logging.error(f"[COVER] Generation error: {e}")
        return error_response(f"Cover letter generation failed: {str(e)}", 500)

# =========================================================
# STATS ROUTE
# =========================================================

@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats():
    """Get dashboard stats for logged in user."""
    user = get_current_user()

    total_searches = Search.query.filter_by(
        user_id=user.id
    ).count()

    total_jobs = Job.query.join(Search).filter(
        Search.user_id == user.id
    ).count()

    avg_score_result = db.session.query(
        db.func.avg(Job.score)
    ).join(Search).filter(
        Search.user_id == user.id
    ).scalar()

    avg_score = round(avg_score_result or 0)

    total_applied = TrackerItem.query.join(Job).join(Search).filter(
        Search.user_id    == user.id,
        TrackerItem.status == 'applied'
    ).count()

    total_saved = TrackerItem.query.join(Job).join(Search).filter(
        Search.user_id    == user.id,
        TrackerItem.status == 'saved'
    ).count()

    total_rejected = TrackerItem.query.join(Job).join(Search).filter(
        Search.user_id    == user.id,
        TrackerItem.status == 'rejected'
    ).count()

    return success_response({
        "stats": {
            "total_searches":  total_searches,
            "total_jobs":      total_jobs,
            "avg_score":       avg_score,
            "total_applied":   total_applied,
            "total_saved":     total_saved,
            "total_rejected":  total_rejected
        }
    })


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return error_response("Endpoint not found.", 404)
    return redirect(url_for('login_page'))


@app.errorhandler(405)
def method_not_allowed(e):
    return error_response("Method not allowed.", 405)


@app.errorhandler(413)
def file_too_large(e):
    return error_response("File too large. Maximum size is 16MB.", 413)


@app.errorhandler(500)
def server_error(e):
    logging.error(f"[SERVER] 500 error: {e}")
    return error_response("Internal server error.", 500)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status":    "healthy",
        "version":   "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    })


# =========================================================
# RUN
# =========================================================

if __name__ == '__main__':
    app.run(
        debug = os.getenv('FLASK_DEBUG', 'True') == 'True',
        port  = 5000,
        host  = '0.0.0.0'
    )