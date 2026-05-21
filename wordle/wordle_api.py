"""
Wordle Game — REST API (Max 8 Guesses)
=========================================
    python3 wordle_api.py → http://localhost:5001

Client protocol:
    1. POST /api/new_game         → {"game_id": "..."}
    2. POST /api/guess            → {"colors": [...], "solved": bool, "failed": bool, "guess_num": N}
    3. POST /api/result (optional)→ {"solved": bool, "num_guesses": N, "penalty": N}
"""
import random, uuid, os
from flask import Flask, request, jsonify, render_template_string

# ── Inlined: no external dependencies needed ──
_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_DIR, 'wordle_words.txt')) as f:
    ANSWERS = [w.strip() for w in f if len(w.strip()) == 5]

def letter_colors(guess, secret):
    g, s = list(guess), list(secret)
    c = ['gray'] * 5
    for i in range(5):
        if g[i] == s[i]: c[i] = 'green'; s[i] = None
    for i in range(5):
        if c[i] == 'green': continue
        if g[i] in s: c[i] = 'yellow'; s[s.index(g[i])] = None
    return c

app = Flask(__name__)
MAX_GUESSES = 8
FAIL_PENALTY = 1000
games = {}


@app.route('/api/new_game', methods=['POST'])
def new_game():
    gid = uuid.uuid4().hex[:8]
    secret = random.choice(ANSWERS)
    games[gid] = {
        "secret": secret,
        "candidates": list(ANSWERS),
        "guesses": [],
        "finished": False,
        "solved": False,
        "failed": False,
    }
    return jsonify({"game_id": gid})


@app.route('/api/guess', methods=['POST'])
def guess():
    data = request.get_json(force=True, silent=True) or {}
    gid = data.get("game_id")
    g = data.get("guess", "").strip().lower()
    if gid not in games:
        return jsonify({"error": "invalid game_id"}), 400
    game = games[gid]
    if game["finished"]:
        return jsonify({"error": "game already finished"}), 400
    if len(g) != 5 or not g.isalpha():
        return jsonify({"error": "guess must be 5 letters"}), 400

    secret = game["secret"]
    colors = letter_colors(g, secret)
    game["guesses"].append({"word": g, "colors": colors})
    color_key = tuple(colors)
    game["candidates"] = [c for c in game["candidates"]
                          if tuple(letter_colors(g, c)) == color_key]
    guess_num = len(game["guesses"])

    if g == secret:
        game["finished"] = True
        game["solved"] = True
    elif len(game["candidates"]) <= 1:
        # Only one left, auto-play it as the last guess
        last = game["candidates"][0]
        game["guesses"].append({"word": last, "colors": letter_colors(last, secret)})
        game["finished"] = True
        game["solved"] = (last == secret)
    elif guess_num >= MAX_GUESSES:
        game["finished"] = True
        game["failed"] = True

    return jsonify({
        "guess": g,
        "colors": colors,
        "guess_num": guess_num,
        "remaining": len(game["candidates"]),
        "solved": game["solved"],
        "failed": game["failed"],
        "finished": game["finished"],
        "max_guesses": MAX_GUESSES,
    })


@app.route('/api/result/<gid>')
def result(gid):
    if gid not in games:
        return jsonify({"error": "not found"}), 404
    g = games[gid]
    return jsonify({
        "solved": g["solved"],
        "failed": g["failed"],
        "num_guesses": len(g["guesses"]),
        "penalty": FAIL_PENALTY if g["failed"] else len(g["guesses"]),
    })


WATCH_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="1"><title>Wordle Watch</title>
<style>
body{font-family:Georgia,serif;max-width:500px;margin:20px auto;text-align:center;background:#121213;color:#fff}
.grid{display:grid;grid-template-columns:repeat(5,56px);gap:5px;justify-content:center;margin:10px 0}
.cell{width:56px;height:56px;border:2px solid #3a3a3c;display:flex;align-items:center;justify-content:center;font-size:1.8rem;font-weight:bold;text-transform:uppercase}
.cell.green{background:#538d4e;border-color:#538d4e}
.cell.yellow{background:#b59f3b;border-color:#b59f3b}
.cell.gray{background:#3a3a3c;border-color:#3a3a3c}
.info{color:#888;font-size:.9rem;margin:4px 0}
</style></head><body>
<h2>Wordle Solver</h2>
<div class="info">{{remaining}} candidates</div>
<div class="grid">
{% for row in range(6) %}
  {% if row < guesses|length %}
    {% for i in range(5) %}
      <div class="cell {{guesses[row].colors[i]}}">{{guesses[row].word[i].upper()}}</div>
    {% endfor %}
  {% else %}
    {% for i in range(5) %}<div class="cell"></div>{% endfor %}
  {% endif %}
{% endfor %}
</div>
<div class="info">Game: {{gid}} | Auto-refreshes</div>
</body></html>"""

@app.route('/game/<gid>')
def watch(gid):
    if gid not in games:
        return "Game not found. POST /api/new_game first.", 404
    g = games[gid]
    return render_template_string(WATCH_HTML, gid=gid,
        guesses=g["guesses"],
        remaining=len(g["candidates"]))


@app.route('/games')
def summary():
    rows = []
    for gid, g in sorted(games.items()):
        ng = len(g["guesses"])
        status = "✅" if g.get("solved") else "❌" if g.get("failed") else "⏳"
        rows.append(f"<tr><td><a href='/game/{gid}'>{gid}</a></td>"
                    f"<td>{status}</td><td>{ng}</td></tr>")
    html = ("""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="2"><title>All Games</title>
<style>body{font-family:Georgia,serif;max-width:600px;margin:20px auto;
background:#121213;color:#fff;text-align:center}
table{margin:10px auto;border-collapse:collapse}
td,th{padding:6px 16px;border-bottom:1px solid #3a3a3c}
a{color:#538d4e}</style></head><body>
<h2>Wordle Games</h2><table><tr><th>Game</th><th>Result</th><th>Guesses</th></tr>"""
            + ''.join(rows) + "</table></body></html>")
    return html


if __name__ == '__main__':
    print(f"Wordle API — {len(ANSWERS)} words, max {MAX_GUESSES} guesses")
    print("POST /api/new_game   POST /api/guess   GET /api/result/<id>")
    print("Watch: http://localhost:5001/game/<id>")
    print("Summary: http://localhost:5001/games")
    app.run(debug=False, port=5001)
