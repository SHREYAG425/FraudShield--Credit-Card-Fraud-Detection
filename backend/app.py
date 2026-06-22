"""
=============================================================
  FraudShield — Flask Backend (app.py)
=============================================================
  Run:   python app.py
  URL:   http://localhost:5000

  API Endpoints:
    POST /api/auth/register
    POST /api/auth/login

    POST /api/transactions/predict       ← ML prediction
    GET  /api/transactions/mine          ← user: own history
    GET  /api/transactions/all           ← analyst/admin only
    GET  /api/transactions/queue         ← analyst/admin only
    POST /api/transactions/<id>/review   ← analyst/admin only

    GET  /api/dashboard/stats            ← analyst/admin only
    GET  /api/analytics                  ← analyst/admin only
    GET  /api/model/info                 ← all roles

    GET  /api/admin/users                ← admin only
    POST /api/admin/users/<id>/toggle    ← admin only
    POST /api/admin/users/<id>/role      ← admin only
    GET  /api/admin/logs                 ← admin only
=============================================================
"""

import os
import json
import pickle
import sqlite3
import secrets
import numpy as np
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, g, render_template
try:
    from flask_cors import CORS
except ImportError:
    class CORS:  # fallback if flask-cors not installed
        def __init__(self, *a, **kw): pass
from werkzeug.security import generate_password_hash, check_password_hash

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
ML_DIR       = BASE_DIR.parent / 'ml' / 'artifacts'
DB_PATH      = BASE_DIR / 'fraudshield.db'

# ─────────────────────────────────────────────
# FLASK SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ─────────────────────────────────────────────
# DATABASE — SQLite (zero config)
# Creates fraudshield.db automatically
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'user',
            is_active     INTEGER NOT NULL DEFAULT 1,
            created_at    TEXT    DEFAULT (datetime('now')),
            last_login    TEXT
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id                TEXT    PRIMARY KEY,
            user_id           INTEGER NOT NULL,
            amount            REAL    NOT NULL,
            txn_type          TEXT    NOT NULL,
            merchant_category TEXT    NOT NULL,
            location_risk     TEXT    NOT NULL,
            hour_of_day       INTEGER NOT NULL,
            card_velocity     TEXT    NOT NULL,
            card_last4        TEXT,
            card_holder       TEXT,
            description       TEXT,
            v1  REAL, v2  REAL, v3  REAL, v4  REAL,
            v5  REAL, v6  REAL, v7  REAL, v8  REAL,
            v9  REAL, v10 REAL, v11 REAL, v12 REAL,
            v13 REAL, v14 REAL, v15 REAL, v16 REAL,
            v17 REAL, v18 REAL, v19 REAL, v20 REAL,
            v21 REAL, v22 REAL, v23 REAL, v24 REAL,
            v25 REAL, v26 REAL, v27 REAL, v28 REAL,
            log_amount        REAL,
            ml_prediction     TEXT    NOT NULL,
            fraud_probability REAL    NOT NULL,
            review_status     TEXT    NOT NULL DEFAULT 'open',
            reviewed_by       INTEGER,
            analyst_note      TEXT,
            reviewed_at       TEXT,
            submitted_at      TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id)     REFERENCES users(id),
            FOREIGN KEY (reviewed_by) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER,
            action     TEXT NOT NULL,
            detail     TEXT,
            ip_address TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()

    # Seed 3 default users
    default_users = [
        ('Admin User',   'admin@fraudshield.com',  generate_password_hash('admin123'),   'admin'),
        ('Bank Analyst', 'analyst@bank.com',        generate_password_hash('analyst123'), 'analyst'),
        ('Test User',    'user@test.com',            generate_password_hash('user123'),    'user'),
    ]
    for u in default_users:
        try:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?,?,?,?)", u
            )
        except sqlite3.IntegrityError:
            pass  # already exists
    conn.commit()
    conn.close()
    print("[DB] Database ready at:", DB_PATH)

def rows_to_list(rows):
    return [dict(r) for r in rows]

def add_log(user_id, action, detail=None):
    conn = get_db()
    conn.execute(
        "INSERT INTO audit_logs (user_id, action, detail, ip_address) VALUES (?,?,?,?)",
        (user_id, action, detail, request.remote_addr)
    )
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# ML MODEL — load once at startup
# ─────────────────────────────────────────────
MODEL  = None
SCALER = None
META   = None

def load_ml_model():
    global MODEL, SCALER, META
    model_path  = ML_DIR / 'model.pkl'
    scaler_path = ML_DIR / 'scaler.pkl'
    meta_path   = ML_DIR / 'model_meta.json'

    if not model_path.exists():
        print(f"[ML] ERROR: model.pkl not found at {model_path}")
        print("[ML] Please run:  cd ml && python train_model.py")
        return False

    with open(model_path,  'rb') as f: MODEL  = pickle.load(f)
    with open(scaler_path, 'rb') as f: SCALER = pickle.load(f)
    with open(meta_path)         as f: META   = json.load(f)
    print(f"[ML] Model loaded: {META['model_type']}")
    print(f"[ML] Features: {META['feature_names']}")
    return True

# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# Builds the exact same feature vector used in training
# ─────────────────────────────────────────────
TYPE_ENC     = {'online': 0, 'pos': 1, 'atm': 2, 'transfer': 3, 'international': 4}
MERCHANT_ENC = {'retail': 0, 'food': 1, 'travel': 2, 'electronics': 3,
                'luxury': 4, 'crypto': 5, 'gambling': 6, 'healthcare': 7}
VELOCITY_ENC = {'low': 0, 'medium': 1, 'high': 2}
LOCATION_ENC = {'low': 0, 'medium': 1, 'high': 2}

def predict_fraud(amount, txn_type, merchant, location, hour, velocity, v_input=None):
    """
    Build feature vector → scale → predict with ML model.
    Returns: (prediction_label, fraud_probability, v_features_list)
    """
    import math

    # Generate or use provided V1-V28 features
    if v_input and len(v_input) >= 28:
        v = [float(x) for x in v_input[:28]]
    else:
        # Auto-generate V features based on transaction risk signals
        # (mimics what real PCA features look like for each risk level)
        v = list(np.random.randn(28) * 0.3)  # base noise

        # Apply known fraud signal patterns from the dataset
        if txn_type == 'international':
            v[0]  -= 3.0   # V1 strongly negative in fraud
            v[2]  -= 2.5   # V3 negative
            v[6]  -= 4.0   # V7 very negative
            v[16] -= 3.0   # V17 negative
        if merchant in ('crypto', 'gambling'):
            v[0]  -= 2.5
            v[3]  += 2.5   # V4 elevated
            v[11] -= 3.5   # V12 negative
            v[13] -= 3.0   # V14 negative
        if amount > 1000:
            v[0]  -= 1.5
            v[9]  -= 2.0   # V10 negative
        if hour < 5 or hour > 22:  # odd hours
            v[0]  -= 1.5
            v[6]  -= 2.0
        if location == 'high':
            v[0]  -= 2.0
            v[15] -= 2.5   # V16
            v[16] -= 3.0   # V17
        if velocity == 'high':
            v[0]  -= 1.5
            v[6]  -= 2.0

    log_amount = math.log1p(amount)

    # Build feature dict matching training order
    feat_dict = {f'V{i+1}': v[i] for i in range(28)}
    
    feat_dict['Time'] = hour
    feat_dict['Hour'] = hour
    feat_dict['Amount'] = amount

    # Order exactly as training
    feature_names = META['feature_names']
    X = np.array([[feat_dict[f] for f in feature_names]])
    X_scaled = SCALER.transform(X)

    prob       = float(MODEL.predict_proba(X_scaled)[0][1])
    prediction = 'Fraudulent' if prob >= META.get('threshold', 0.5) else 'Genuine'

    return prediction, round(prob, 4), v


# ─────────────────────────────────────────────
# SESSION STORE (in-memory tokens)
# For production: use JWT or Redis
# ─────────────────────────────────────────────
SESSIONS = {}   # token → user_id

def create_session(user_id):
    token = secrets.token_hex(32)
    SESSIONS[token] = user_id
    return token

# ─────────────────────────────────────────────
# AUTH DECORATORS
# ─────────────────────────────────────────────
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
        uid   = SESSIONS.get(token)
        if not uid:
            return jsonify({'error': 'Unauthorized — please login'}), 401
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        conn.close()
        if not user or not user['is_active']:
            return jsonify({'error': 'Unauthorized'}), 401
        g.user = dict(user)
        return f(*args, **kwargs)
    return decorated

def require_role(*roles):
    def decorator(f):
        @wraps(f)
        @require_auth
        def decorated(*args, **kwargs):
            if g.user['role'] not in roles:
                return jsonify({'error': f'Access denied. Required role: {roles}'}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

TXN_COUNTER = [10000]  # auto-increment transaction IDs



@app.route("/")
def index():
    return render_template("index.html")

# ═════════════════════════════════════════════
# ROUTES — AUTH
# ═════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    d     = request.json or {}
    name  = d.get('name', '').strip()
    email = d.get('email', '').strip().lower()
    pwd   = d.get('password', '')
    role  = d.get('role', 'user')

    if not name or not email or not pwd:
        return jsonify({'error': 'Name, email and password are required'}), 400
    if len(pwd) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if role not in ('user', 'analyst'):
        role = 'user'  # cannot self-register as admin

    conn = get_db()
    if conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
        conn.close()
        return jsonify({'error': 'Email already registered'}), 409

    conn.execute(
        "INSERT INTO users (name, email, password_hash, role) VALUES (?,?,?,?)",
        (name, email, generate_password_hash(pwd), role)
    )
    conn.commit()
    uid = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()['id']
    conn.close()

    add_log(uid, 'register', f'New {role} account registered')
    token = create_session(uid)
    return jsonify({'token': token, 'user': {'id': uid, 'name': name, 'email': email, 'role': role}}), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    d     = request.json or {}
    email = d.get('email', '').strip().lower()
    pwd   = d.get('password', '')

    conn = get_db()
    row  = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row or not check_password_hash(row['password_hash'], pwd):
        conn.close()
        return jsonify({'error': 'Invalid email or password'}), 401
    if not row['is_active']:
        conn.close()
        return jsonify({'error': 'Account has been deactivated. Contact admin.'}), 403

    conn.execute("UPDATE users SET last_login=datetime('now') WHERE id=?", (row['id'],))
    conn.commit()
    conn.close()

    add_log(row['id'], 'login', f'Login from {request.remote_addr}')
    token = create_session(row['id'])
    return jsonify({
        'token': token,
        'user': {'id': row['id'], 'name': row['name'], 'email': row['email'], 'role': row['role']}
    })


# ═════════════════════════════════════════════
# ROUTES — TRANSACTIONS
# ═════════════════════════════════════════════

@app.route('/api/transactions/predict', methods=['POST'])
@require_auth
def predict():
    if not MODEL:
        return jsonify({'error': 'ML model not loaded. Run train_model.py first.'}), 503

    d = request.json or {}
    required = ['amount', 'txn_type', 'merchant_category', 'location_risk', 'hour_of_day', 'card_velocity']
    for field in required:
        if field not in d:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    try:
        amount = float(d['amount'])
        if amount <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({'error': 'Amount must be a positive number'}), 400

    v_input = d.get('v_features')  # optional list of 28 PCA values

    prediction, prob, v = predict_fraud(
        amount   = amount,
        txn_type = d['txn_type'],
        merchant = d['merchant_category'],
        location = d['location_risk'],
        hour     = int(d['hour_of_day']),
        velocity = d['card_velocity'],
        v_input  = v_input,
    )

   
    txn_id = f"TXN-{secrets.token_hex(4)}"

    # Build V columns for DB insert
    v_vals = [v[i] if i < len(v) else 0.0 for i in range(28)]

    conn = get_db()
    conn.execute("""
        INSERT INTO transactions (
            id, user_id, amount, txn_type, merchant_category,
            location_risk, hour_of_day, card_velocity,
            card_last4, card_holder, description,
            v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,
            v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,
            log_amount, ml_prediction, fraud_probability, review_status
        ) VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?
        )
    """, (
        txn_id, g.user['id'], amount, d['txn_type'], d['merchant_category'],
        d['location_risk'], int(d['hour_of_day']), d['card_velocity'],
        d.get('card_last4', '****'), d.get('card_holder', ''), d.get('description', ''),
        *v_vals,
        float(np.log1p(amount)),
        prediction, prob,
        'open' if prediction == 'Fraudulent' else 'cleared'
    ))
    conn.commit()
    conn.close()

    add_log(g.user['id'], 'predict', f"{txn_id} → {prediction} ({prob*100:.1f}%)")

    return jsonify({
        'transaction_id'   : txn_id,
        'prediction'       : prediction,
        'fraud_probability': prob,
        'is_fraud'         : prediction == 'Fraudulent',
        'amount'           : amount,
        'submitted_at'     : datetime.now().isoformat(),
    })


@app.get('/api/transactions/mine')
@require_auth
def my_transactions():
    conn = get_db()
    rows = conn.execute("""
        SELECT t.*, u.name AS reviewer_name
        FROM transactions t
        LEFT JOIN users u ON t.reviewed_by = u.id
        WHERE t.user_id = ?
        ORDER BY t.submitted_at DESC
    """, (g.user['id'],)).fetchall()
    conn.close()
    return jsonify(rows_to_list(rows))


@app.get('/api/transactions/all')
@require_role('analyst', 'admin')
def all_transactions():
    verdict  = request.args.get('verdict', '')
    status   = request.args.get('status', '')
    txn_type = request.args.get('txn_type', '')
    search   = request.args.get('search', '')

    query  = """
        SELECT t.*, u.name AS user_name, rv.name AS reviewer_name
        FROM transactions t
        JOIN users u ON t.user_id = u.id
        LEFT JOIN users rv ON t.reviewed_by = rv.id
        WHERE 1=1
    """
    params = []
    if verdict:  query += " AND t.ml_prediction = ?";   params.append(verdict)
    if status:   query += " AND t.review_status = ?";   params.append(status)
    if txn_type: query += " AND t.txn_type = ?";        params.append(txn_type)
    if search:
        query += " AND (t.id LIKE ? OR u.name LIKE ? OR CAST(t.amount AS TEXT) LIKE ?)"
        params += [f'%{search}%', f'%{search}%', f'%{search}%']
    query += " ORDER BY t.submitted_at DESC LIMIT 1000"

    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify(rows_to_list(rows))


@app.get('/api/transactions/queue')
@require_role('analyst', 'admin')
def fraud_queue():
    conn = get_db()
    rows = conn.execute("""
        SELECT t.*, u.name AS user_name
        FROM transactions t
        JOIN users u ON t.user_id = u.id
        WHERE t.ml_prediction = 'Fraudulent' AND t.review_status = 'open'
        ORDER BY t.fraud_probability DESC
    """).fetchall()
    conn.close()
    return jsonify(rows_to_list(rows))

@app.route('/api/transactions/<txn_id>/review', methods=['POST'])
@require_role('analyst', 'admin')
def review_transaction(txn_id):
    d      = request.json or {}
    action = d.get('action', '')
    note   = d.get('note', '')

    valid = ('investigating', 'confirmed_fraud', 'cleared')
    if action not in valid:
        return jsonify({'error': f'action must be one of {valid}'}), 400

    conn = get_db()
    txn  = conn.execute("SELECT * FROM transactions WHERE id=?", (txn_id,)).fetchone()
    if not txn:
        conn.close()
        return jsonify({'error': 'Transaction not found'}), 404

    conn.execute("""
        UPDATE transactions
        SET review_status=?, reviewed_by=?, analyst_note=?, reviewed_at=datetime('now')
        WHERE id=?
    """, (action, g.user['id'], note, txn_id))
    conn.commit()
    conn.close()

    add_log(g.user['id'], 'review', f"{txn_id} marked as {action}")
    return jsonify({'success': True, 'transaction_id': txn_id, 'action': action})


# ═════════════════════════════════════════════
# ROUTES — DASHBOARD & ANALYTICS
# ═════════════════════════════════════════════

@app.get('/api/dashboard/stats')
@require_role('analyst', 'admin')
def dashboard_stats():
    conn   = get_db()
    total  = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud  = conn.execute("SELECT COUNT(*) FROM transactions WHERE ml_prediction='Fraudulent'").fetchone()[0]
    open_q = conn.execute("SELECT COUNT(*) FROM transactions WHERE ml_prediction='Fraudulent' AND review_status='open'").fetchone()[0]
    conf   = conn.execute("SELECT COUNT(*) FROM transactions WHERE review_status='confirmed_fraud'").fetchone()[0]
    clr    = conn.execute("SELECT COUNT(*) FROM transactions WHERE review_status='cleared' AND ml_prediction='Fraudulent'").fetchone()[0]
    avg_p  = conn.execute("SELECT AVG(fraud_probability) FROM transactions WHERE ml_prediction='Fraudulent'").fetchone()[0] or 0

    recent = rows_to_list(conn.execute("""
        SELECT t.id, u.name AS user_name, t.amount, t.txn_type,
               t.merchant_category, t.fraud_probability, t.review_status, t.submitted_at
        FROM transactions t
        JOIN users u ON t.user_id = u.id
        WHERE t.ml_prediction = 'Fraudulent'
        ORDER BY t.submitted_at DESC LIMIT 8
    """).fetchall())
    conn.close()

    return jsonify({
        'total'          : total,
        'fraud'          : fraud,
        'genuine'        : total - fraud,
        'open_queue'     : open_q,
        'confirmed'      : conf,
        'cleared_fp'     : clr,
        'avg_fraud_prob' : round(avg_p * 100, 1),
        'fraud_rate'     : round(fraud / total * 100, 2) if total else 0,
        'recent_fraud'   : recent,
    })


@app.get('/api/analytics')
@require_role('analyst', 'admin')
def analytics():
    conn = get_db()

    by_merchant = rows_to_list(conn.execute("""
        SELECT merchant_category,
               COUNT(*) AS total,
               SUM(CASE WHEN ml_prediction='Fraudulent' THEN 1 ELSE 0 END) AS fraud
        FROM transactions GROUP BY merchant_category
    """).fetchall())

    by_type = rows_to_list(conn.execute("""
        SELECT txn_type,
               COUNT(*) AS total,
               SUM(CASE WHEN ml_prediction='Fraudulent' THEN 1 ELSE 0 END) AS fraud
        FROM transactions GROUP BY txn_type
    """).fetchall())

    by_hour = rows_to_list(conn.execute("""
        SELECT hour_of_day AS hour,
               COUNT(*) AS total,
               SUM(CASE WHEN ml_prediction='Fraudulent' THEN 1 ELSE 0 END) AS fraud
        FROM transactions GROUP BY hour_of_day ORDER BY hour_of_day
    """).fetchall())

    by_amount = rows_to_list(conn.execute("""
        SELECT
          CASE WHEN amount < 50   THEN '<$50'
               WHEN amount < 200  THEN '$50-200'
               WHEN amount < 500  THEN '$200-500'
               WHEN amount < 2000 THEN '$500-2K'
               ELSE '>$2K'
          END AS bucket,
          COUNT(*) AS total,
          SUM(CASE WHEN ml_prediction='Fraudulent' THEN 1 ELSE 0 END) AS fraud
        FROM transactions GROUP BY bucket
    """).fetchall())

    top_fraud = rows_to_list(conn.execute("""
        SELECT t.id, u.name AS user_name, t.amount,
               t.merchant_category, t.location_risk,
               t.fraud_probability, t.review_status
        FROM transactions t
        JOIN users u ON t.user_id = u.id
        WHERE t.ml_prediction = 'Fraudulent'
        ORDER BY t.fraud_probability DESC LIMIT 10
    """).fetchall())

    conn.close()
    return jsonify({
        'by_merchant': by_merchant,
        'by_type'    : by_type,
        'by_hour'    : by_hour,
        'by_amount'  : by_amount,
        'top_fraud'  : top_fraud,
    })


@app.get('/api/model/info')
@require_auth
def model_info():
    if not META:
        return jsonify({'error': 'Model not loaded'}), 503
    return jsonify(META)


# ═════════════════════════════════════════════
# ROUTES — ADMIN
# ═════════════════════════════════════════════

@app.get('/api/admin/users')
@require_role('admin')
def get_users():
    conn = get_db()
    rows = rows_to_list(conn.execute("""
        SELECT u.id, u.name, u.email, u.role, u.is_active,
               u.created_at, u.last_login,
               COUNT(t.id) AS txn_count,
               SUM(CASE WHEN t.ml_prediction='Fraudulent' THEN 1 ELSE 0 END) AS fraud_count
        FROM users u
        LEFT JOIN transactions t ON u.id = t.user_id
        GROUP BY u.id
        ORDER BY u.created_at
    """).fetchall())
    conn.close()
    return jsonify(rows)

@app.route('/api/admin/users/<int:uid>/toggle', methods=['POST'])
@require_role('admin')
def toggle_user(uid):
    if uid == g.user['id']:
        return jsonify({'error': 'Cannot deactivate your own account'}), 400

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    new_status = 0 if user['is_active'] else 1
    conn.execute("UPDATE users SET is_active=? WHERE id=?", (new_status, uid))
    conn.commit()
    conn.close()

    action = 'activate' if new_status else 'deactivate'
    add_log(g.user['id'], action, f"User {user['name']} ({user['email']})")
    return jsonify({'success': True, 'is_active': bool(new_status)})


@app.route('/api/admin/users/<int:uid>/role', methods=['POST'])
@require_role('admin')
def change_role(uid):
    d        = request.json or {}
    new_role = d.get('role', '')
    if new_role not in ('admin', 'analyst', 'user'):
        return jsonify({'error': 'Invalid role'}), 400

    conn = get_db()
    conn.execute("UPDATE users SET role=? WHERE id=?", (new_role, uid))
    conn.commit()
    conn.close()

    add_log(g.user['id'], 'change_role', f"User {uid} → {new_role}")
    return jsonify({'success': True})


@app.get('/api/admin/logs')
@require_role('admin')
def get_logs():
    limit = min(int(request.args.get('limit', 100)), 500)
    conn  = get_db()
    rows  = rows_to_list(conn.execute("""
        SELECT l.*, u.name AS user_name
        FROM audit_logs l
        LEFT JOIN users u ON l.user_id = u.id
        ORDER BY l.created_at DESC
        LIMIT ?
    """, (limit,)).fetchall())
    conn.close()
    return jsonify(rows)


# ═════════════════════════════════════════════
# START
# ═════════════════════════════════════════════
if __name__ == '__main__':
    print("\n🛡  FraudShield — Flask Backend")
    print("="*40)
    init_db()
    if not load_ml_model():
        print("\n⚠  WARNING: Could not load ML model.")
        print("   Run this first:  cd ../ml && python train_model.py\n")
    else:
        print("[ML] Model ready for predictions")
    print("\n[SERVER] Starting on http://localhost:5000")
    print("[SERVER] Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000)                                 