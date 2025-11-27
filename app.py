"""
Flask app + training script in one file with internal CSS.
"""

from flask import Flask, request, render_template_string, jsonify, send_from_directory
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Try to import matplotlib; if unavailable, fallback to SVG
_have_plt = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    _have_plt = False

MODEL_PATH = 'linear_model_hours_score.joblib'
PLOT_PNG = 'static/regression_plot.png'
PLOT_SVG = 'static/regression_plot.svg'
SAMPLE_CSV = 'sample_hours_scores.csv'

# ------------------ Data & Training ------------------
def create_synthetic_df(n=100, seed=42):
    rng = np.random.RandomState(seed)
    hours = rng.uniform(0.5, 10, n)
    scores = 5 + 9 * hours + rng.normal(0, 5, n)
    return pd.DataFrame({'Hours': np.round(hours, 2), 'Score': np.round(scores, 2)})

def train_model(df):
    X = df[['Hours']].values
    y = df['Score'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {'model': model, 'mse': mse, 'r2': r2}

def train_and_save_model(force_retrain=False):
    if os.path.exists(MODEL_PATH) and not force_retrain:
        try:
            params = joblib.load(MODEL_PATH)
            if isinstance(params, dict) and 'model' in params:
                df = pd.read_csv(SAMPLE_CSV) if os.path.exists(SAMPLE_CSV) else create_synthetic_df()
                params['df'] = df
                return params
        except Exception:
            pass
    df = create_synthetic_df()
    params = train_model(df)
    params['df'] = df
    joblib.dump(params, MODEL_PATH)
    df.to_csv(SAMPLE_CSV, index=False)
    return params

# ------------------ Plot helpers ------------------
def save_png_plot(df, model, out_path=PLOT_PNG):
    plt.figure(figsize=(8,5))
    plt.scatter(df['Hours'], df['Score'], s=36, alpha=0.85)
    x_line = np.linspace(df['Hours'].min()-0.5, df['Hours'].max()+0.5, 200).reshape(-1,1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, linewidth=2, label='Regression line')
    plt.title('Hours studied vs Exam Score')
    plt.xlabel('Hours studied')
    plt.ylabel('Exam score')
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_svg_plot(df, model, out_path=PLOT_SVG, width=700, height=320):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    min_x = float(df['Hours'].min()-0.5)
    max_x = float(df['Hours'].max()+0.5)
    try:
        pred_min = float(model.predict([[min_x]])[0])
        pred_max = float(model.predict([[max_x]])[0])
    except Exception:
        pred_min, pred_max = df['Score'].min(), df['Score'].max()
    min_y = float(min(df['Score'].min(), pred_min)-5)
    max_y = float(max(df['Score'].max(), pred_max)+5)
    def tx(x): return 40 + (x-min_x)/(max_x-min_x)*(width-80)
    def ty(y): return 20 + (max_y-y)/(max_y-min_y)*(height-60)
    points_svg = '\n'.join(
        f'<circle cx="{tx(row.Hours):.1f}" cy="{ty(row.Score):.1f}" r="3" fill="#1f78b4" opacity="0.9" />'
        for row in df.itertuples()
    )
    x0, x1 = min_x, max_x
    try:
        y0 = float(model.predict([[x0]])[0])
        y1 = float(model.predict([[x1]])[0])
    except Exception:
        y0, y1 = df['Score'].min(), df['Score'].max()
    line_svg = f'<line x1="{tx(x0):.1f}" y1="{ty(y0):.1f}" x2="{tx(x1):.1f}" y2="{ty(y1):.1f}" stroke="#f59e0b" stroke-width="2" />'
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#ffffff" rx="8" />
  <line x1="40" y1="{height-30}" x2="{width-40}" y2="{height-30}" stroke="#ddd" />
  <line x1="40" y1="20" x2="40" y2="{height-30}" stroke="#ddd" />
  {line_svg}
  {points_svg}
  <text x="20" y="14" font-family="Arial,Helvetica,sans-serif" font-size="12" fill="#374151">Hours studied vs Exam Score</text>
  <text x="{(width/2):.1f}" y="{(height-8):.1f}" font-family="Arial" font-size="11" fill="#6b7280" text-anchor="middle">Hours</text>
  <text x="12" y="{(height/2):.1f}" font-family="Arial" font-size="11" fill="#6b7280" text-anchor="middle" transform="rotate(-90 12,{(height/2):.1f})">Score</text>
</svg>
'''
    with open(out_path, 'w', encoding='utf-8') as f: f.write(svg)

# ------------------ Flask App ------------------
app = Flask(__name__)
_PARAMS = train_and_save_model()
_MODEL = _PARAMS['model']
_MSE = _PARAMS['mse']
_R2 = _PARAMS['r2']
_DF = _PARAMS['df']

if _have_plt:
    try: save_png_plot(_DF, _MODEL, PLOT_PNG); plot_url = '/' + PLOT_PNG
    except Exception: save_svg_plot(_DF, _MODEL, PLOT_SVG); plot_url = '/' + PLOT_SVG
else: save_svg_plot(_DF, _MODEL, PLOT_SVG); plot_url = '/' + PLOT_SVG

INDEX_HTML = '''
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Score Predictor</title>
<style>
body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px}
.container{max-width:950px;margin:0 auto;background:#fff;padding:20px;border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,0.15)}
h1{margin:0 0 12px;font-size:26px;color:#1f2937}
.grid{display:grid;grid-template-columns:1fr 360px;gap:18px}
.card{background:#f9fafb;padding:16px;border-radius:10px;border:1px solid #e5e7eb;transition:all 0.3s ease}
.card:hover{border-color:#d1d5db;box-shadow:0 4px 12px rgba(0,0,0,0.08)}
label{display:block;font-size:12px;color:#4b5563;margin-bottom:8px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px}
input[type=range]{width:100%;height:6px;border-radius:3px;background:#e5e7eb;-webkit-appearance:none;appearance:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:18px;height:18px;border-radius:50%;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);cursor:pointer;box-shadow:0 2px 8px rgba(102,126,234,0.4)}
input[type=range]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);cursor:pointer;border:none;box-shadow:0 2px 8px rgba(102,126,234,0.4)}
input[type=number]{width:80px;padding:8px 12px;border-radius:6px;border:1.5px solid #d1d5db;font-size:14px;font-weight:600;background:#fff;transition:all 0.2s}
input[type=number]:focus{outline:none;border-color:#667eea;box-shadow:0 0 0 3px rgba(102,126,234,0.1)}
.btn{display:inline-block;padding:8px 12px;border-radius:8px;background:#667eea;color:#fff;border:none;cursor:pointer;font-weight:600;transition:all 0.3s;margin-right:6px}
.btn:hover{background:#764ba2;transform:translateY(-1px);box-shadow:0 4px 12px rgba(102,126,234,0.3)}
.btn-secondary{background:#e5e7eb;color:#374151}
.btn-secondary:hover{background:#d1d5db}
.pred-section{margin-top:16px;padding:16px;background:linear-gradient(135deg,#f0f4ff 0%,#f5f0ff 100%);border-radius:10px;border-left:4px solid #667eea}
.pred-label{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
.pred-value{font-size:48px;font-weight:700;color:#667eea;margin:0;display:inline}
.pred-percent{font-size:24px;color:#764ba2;font-weight:600}
.progress-bar{width:100%;height:12px;background:#e5e7eb;border-radius:6px;overflow:hidden;margin-top:12px}
.progress-fill{height:100%;background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);border-radius:6px;transition:width 0.4s ease;box-shadow:0 0 8px rgba(102,126,234,0.3)}
.pred-unit{font-size:13px;color:#6b7280;margin-top:10px;font-weight:600}
.meta{font-size:12px;color:#6b7280;margin-top:12px;line-height:1.6}
.plot{width:100%;height:auto;border-radius:8px;border:1px solid #e5e7eb}
/* Error message styling */
.error-msg{display:none;margin-top:10px;padding:8px 10px;border-radius:8px;background:#fff5f5;border:1px solid #fecaca;color:#b91c1c;font-size:13px}
.error-msg strong{margin-right:6px}
</style>
</head>
<body>
<div class="container">
<h1>Student Score Predictor (Linear Regression)</h1>
<p style="color:#64748b;margin-top:0">Enter number of study hours to get a predicted exam score. Model metrics shown on the right.</p>
<div class="grid">
<div>
<div class="card">
<h3 style="margin:0 0 12px;font-size:16px;color:#1f2937">Prediction Input</h3>
<label for="hoursRange">Hours studied (0-12)</label>
<input id="hoursRange" type="range" min="0" max="12" step="0.1" value="7.5" oninput="syncHours(this.value)">
<div style="display:flex;gap:8px;margin-top:10px;align-items:center">
<input id="hoursNumber" type="number" min="0" max="12" step="0.1" value="7.5" onchange="syncRange(this.value)">
<button class="btn" onclick="getPrediction()">⚡ Predict</button>
<button class="btn btn-secondary" onclick="resetForm()">Reset</button>
</div>
<!-- Error message placeholder -->
<div id="errorMsg" class="error-msg" role="status" aria-live="polite" style="display:none"><strong>Error</strong><span id="errorText"></span></div>
<div class="pred-section">
<div class="pred-label">Predicted Score (%)</div>
<div style="margin-bottom:8px"><span class="pred-value" id="predicted">--</span><span class="pred-percent">%</span></div>
<div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
<div class="pred-unit" id="predRange"></div>
</div>
<div class="meta" style="margin-top:12px;padding-top:12px;border-top:1px solid #e5e7eb">
<strong>Model Formula:</strong><br>Score = {{ intercept }} + {{ coef }} × Hours
</div>
</div>
<div class="card" style="margin-top:12px">
<h3 style="margin:0 0 8px">Dataset sample</h3>
<div style="max-height:160px;overflow:auto">
<table style="width:100%;border-collapse:collapse;font-size:13px">
<thead style="color:#475569"><tr><th style="text-align:left;padding:6px">Hours</th><th style="text-align:left;padding:6px">Score</th></tr></thead>
<tbody>
{% for row in rows %}
<tr style="border-top:1px solid #f1f5f9"><td style="padding:6px">{{ row.Hours }}</td><td style="padding:6px">{{ row.Score }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>
</div>
</div>
<div>
<div class="card">
<h3 style="margin:0 0 8px">Model & Plot</h3>
<img src="{{ plot_url }}" alt="regression plot" class="plot">
<div class="meta">Test MSE: {{ '%.4f' % mse }} &nbsp; R²: {{ '%.4f' % r2 }}</div>
</div>
</div>
</div>
</div>
<script>
function syncHours(v){
const val=Number(v).toFixed(1);
document.getElementById('hoursNumber').value=val;
getPrediction();
}
function syncRange(v){
const val=Number(v);
document.getElementById('hoursRange').value=val;
getPrediction();
}
function resetForm(){
document.getElementById('hoursRange').value=7.5;
document.getElementById('hoursNumber').value=7.5;
document.getElementById('predicted').innerText='--';
document.getElementById('progressFill').style.width='0%';
document.getElementById('predRange').innerText='';
    // hide any visible error
    const errEl = document.getElementById('errorMsg');
    const errText = document.getElementById('errorText');
    if (errEl) { errEl.style.display = 'none'; if (errText) errText.innerText = ''; }
    getPrediction();
}
async function getPrediction(){
    const hours = Number(document.getElementById('hoursNumber').value);
    const errEl = document.getElementById('errorMsg');
    const errText = document.getElementById('errorText');
    if (errEl) { errEl.style.display = 'none'; if (errText) errText.innerText = ''; }

    if (isNaN(hours) || hours < 0 || hours > 12) {
        if (errEl && errText) {
            errText.innerText = 'Please enter a valid number of hours (0–12).';
            errEl.style.display = 'block';
        }
        document.getElementById('predicted').innerText = '--';
        document.getElementById('progressFill').style.width = '0%';
        return;
    }

    try {
        const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ hours }) });
        if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            const msg = body && body.error ? body.error : `Server returned ${res.status}`;
            if (errEl && errText) { errText.innerText = msg; errEl.style.display = 'block'; }
            document.getElementById('predicted').innerText = '--';
            document.getElementById('progressFill').style.width = '0%';
            return;
        }
        const j = await res.json();
        if (j && typeof j.predicted === 'number') {
            // Convert raw score to percentage (cap to 0-100)
            const scoreVal = j.predicted;
            let percentage = Math.min(100, Math.max(0, (scoreVal / 120) * 100));
            percentage = Math.round(percentage);
            document.getElementById('predicted').innerText = percentage;
            document.getElementById('progressFill').style.width = percentage + '%';
            let feedback = '';
            if (percentage >= 90) feedback = '🏆 Outstanding!';
            else if (percentage >= 80) feedback = '🎉 Excellent!';
            else if (percentage >= 70) feedback = '✅ Very Good';
            else if (percentage >= 60) feedback = '👍 Good';
            else if (percentage >= 50) feedback = '📈 Fair';
            else feedback = '💪 Keep studying!';
            document.getElementById('predRange').innerText = feedback + ' (Raw: ' + j.predicted.toFixed(1) + ')';
            // hide error if any
            if (errEl) errEl.style.display = 'none';
        } else {
            if (errEl && errText) { errText.innerText = 'Invalid response from server.'; errEl.style.display = 'block'; }
            document.getElementById('predicted').innerText = '--';
            document.getElementById('progressFill').style.width = '0%';
        }
    } catch (err) {
        if (errEl && errText) { errText.innerText = 'Request failed: ' + (err.message || err); errEl.style.display = 'block'; }
        document.getElementById('predicted').innerText = '--';
        document.getElementById('progressFill').style.width = '0%';
    }
}
window.addEventListener('load',()=>getPrediction());
</script>
</body>
</html>
'''

@app.route('/')
def index():
    df = _DF if _DF is not None else create_synthetic_df()
    rows = df.head(8).to_dict(orient='records')
    return render_template_string(
        INDEX_HTML,
        rows=rows,
        intercept=float(getattr(_MODEL, 'intercept_', 0.0)),
        coef=float(getattr(_MODEL, 'coef_', [0.0])[0]),
        mse=_MSE,
        r2=_R2,
        plot_url=plot_url
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    try:
        hours = float(data.get('hours', 0))
        pred = float(_MODEL.predict([[hours]])[0])
        return jsonify({'predicted': pred})
    except Exception as e:
        return jsonify({'error': f'prediction failed: {str(e)}'}), 400

@app.route('/health')
def health():
    return jsonify({'status':'ok','has_matplotlib':_have_plt,'mse':_MSE,'r2':_R2})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__=='__main__':
    debug = os.environ.get('FLASK_DEBUG','1')=='1'
    print(f"Starting app (matplotlib available={_have_plt}), debug={debug}")
    app.run(debug=debug)
