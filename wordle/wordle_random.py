"""
Wordle Random Guesser — Baseline Strategy
============================================
Completely random 5-letter strings. For API demonstration only.
"""
import argparse, random, string
from wordle_client import benchmark, run_one


def next_guess(history):
    """Ignore history, return a random 5-letter string."""
    print("History:", history)
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(5))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=0)
    p.add_argument("--web", action="store_true")
    args = p.parse_args()
    if args.games:
        benchmark(next_guess, n=args.games, web=args.web)
    else:
        res = run_one(next_guess, verbose=True, open_browser=args.web)
        print(f"Solved: {res['solved']}  Guesses: {res['num_guesses']}")
