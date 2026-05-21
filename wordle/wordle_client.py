"""
Wordle Client Runner — Shared Game Loop
=========================================
Handles all API communication. You only implement next_guess(history).

    history = [("crane", ["green","gray","gray","yellow","gray"]), ...]
    return "your_guess"   # 5-letter string
"""
import requests, os

API = "http://localhost:5001"

# Load word list for local use
_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_DIR, 'wordle_words.txt')) as f:
    ALL_WORDS = [w.strip() for w in f if len(w.strip()) == 5]


def run_one(next_guess_fn, verbose=False, open_browser=False, delay=0.0):
    """Play one game using the given strategy function."""
    r = requests.post(f"{API}/api/new_game")
    gid = r.json()["game_id"]
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:5001/game/{gid}")
        __import__('time').sleep(0.5)  # let browser load the empty grid
    if verbose: print(f"Game {gid}  →  http://localhost:5001/game/{gid}")

    history = []  # list of (guess, result_list)
    while True:
        g = next_guess_fn(history)
        r = requests.post(f"{API}/api/guess", json={"game_id": gid, "guess": g})
        d = r.json()
        history.append((g, d["colors"]))
        if verbose:
            disp = ' '.join({'green':'G','yellow':'Y','gray':'.'}[c] for c in d["colors"])
            print(f"  {g.upper()} → {disp}  (#{d['guess_num']})")

        if d["finished"]:
            break
        if open_browser:
            __import__('time').sleep(0.8)  # let browser refresh between guesses

    r = requests.get(f"{API}/api/result/{gid}")
    return r.json()


def benchmark(strategy_fn, n=50, verbose=False, web=False):
    """Play n games and report stats."""
    if web:
        import webbrowser
        webbrowser.open("http://localhost:5001/games")
    solved = 0
    guesses_dist = {}
    for i in range(n):
        res = run_one(strategy_fn, open_browser=False)  # summary page, not per-game
        if web: __import__('time').sleep(1.5)
        ng = res["num_guesses"]
        ng = res["num_guesses"]
        if res["solved"]:
            solved += 1
            guesses_dist[ng] = guesses_dist.get(ng, 0) + 1
        print(f"\r  Game {i+1}/{n}", end="", flush=True)
    print()
    print(f"Games: {n}  |  Solved: {solved}  |  Failed: {n-solved}")
    avg = sum(k*v for k,v in guesses_dist.items()) / max(solved, 1)
    print(f"Avg guesses (solved): {avg:.2f}")
    print(f"Distribution: {dict(sorted(guesses_dist.items()))}")
