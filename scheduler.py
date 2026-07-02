"""Spaced repetition — the SM-2 algorithm (the one Anki is based on).

After you review a card you rate how well you recalled it (quality 0-5):
    5 perfect · 4 correct after hesitation · 3 correct but hard
    2 wrong but familiar · 1 wrong · 0 total blackout

From that rating + the card's history we compute WHEN to show it again. Easy cards
drift far into the future; ones you fail come back tomorrow. That schedule is what
makes people come back daily — and it's memory ChatGPT simply doesn't keep.
"""

from datetime import datetime, timedelta


def sm2(quality: int, repetitions: int, ease: float, interval: int):
    """Return (repetitions, ease, interval_days, due_date) after a review."""
    quality = max(0, min(5, int(quality)))

    if quality < 3:
        # failed recall — reset the streak, see it again tomorrow
        repetitions = 0
        interval = 1
    else:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = round(interval * ease)
        repetitions += 1

    # adjust how "easy" the card is (never below 1.3)
    ease = ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease = max(1.3, ease)

    due_date = datetime.utcnow() + timedelta(days=interval)
    return repetitions, ease, interval, due_date
