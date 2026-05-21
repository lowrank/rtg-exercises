# Wordle Solver

Self-contained Wordle game API + solver clients.

## Files

| File | Purpose |
|---|---|
| `wordle_api.py` | Flask server — self-contained |
| `wordle_client.py` | Game loop runner — handles API, you write `next_guess(history) |
| `wordle_random.py` | Random baseline — pure random strings |
| `wordle_words.txt` | 1000 clean 5-letter words |
| `requirements.txt` | `flask`, `requests` |

## Quick Start

```bash
pip install -r requirements.txt
python3 wordle_api.py &          # → http://localhost:5001
python3 wordle_random.py --games 50 --web
```

## Game Rules

- 5-letter secret from 1000-word list (your client can access)
- Max 8 guesses per game
- Failed → penalty = 1000 (for benchmark scoring)
- Feedback: `green` = correct position, `yellow` = wrong position, `gray` = not in word

## Writing a Custom Strategy

Make a copy of ``wordle_random.py``, and write a function `next_guess(history)` and pass it to `run_one()` or `benchmark()` from `wordle_client.py`:

```python
from wordle_client import run_one, benchmark

def my_strategy(history):
    # history = [("crane", ["green","gray","gray","yellow","gray"]), ...]
    return "adieu"  # your logic here

run_one(my_strategy, verbose=True)           # play 1 game
benchmark(my_strategy, n=50)                 # benchmark 50 games
benchmark(my_strategy, n=5, web=True)        # + open browser
```

## API Endpoints

### `POST /api/new_game`

Start a new game → `{"game_id": "abc12345"}`

### `POST /api/guess`

Submit a guess → `{"colors": [...], "solved": bool, "failed": bool, "finished": bool, "guess_num": N, "remaining": N}`

### `GET /api/result/<game_id>`

Final result → `{"solved": bool, "num_guesses": N, "penalty": N}`. Penalty = 1000 if failed.

### `GET /game/<gid>`

Watch page — auto-refreshing grid view of a single game.

### `GET /games`

Summary page — all games with ✅/❌ status and guess counts.
