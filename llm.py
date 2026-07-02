"""The 'generation' half of RAG — everything that talks to an AI model.

Provider-agnostic: the app auto-picks based on which key is in your .env —
  GEMINI_API_KEY    -> Google Gemini  (free tier)
  ANTHROPIC_API_KEY -> Claude
Embeddings/retrieval are local and unaffected either way.

Grounding rule everywhere: the model may use ONLY the provided notes. If the notes
don't contain the answer, it must say so (never invent). That honesty is the whole
pitch vs. a general chatbot.

Study generators (quiz / flashcards / summary / grading) need a key; `/ask` still
works without one (it returns the retrieved chunks) — see generate_answer.
"""

import os
import re
import json
import config


class NoAPIKey(Exception):
    """Raised by study features that can't run without an API key."""


class AIError(Exception):
    """A friendly, user-facing message when the AI provider call fails."""


def _friendly_error(e: Exception) -> str:
    """Turn a raw provider exception into a short, helpful message."""
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg or "rate limit" in msg or "413" in msg:
        m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"'?(\d+)s'?", msg)
        wait = f" (wait ~{int(float(m.group(1)))}s)" if m else ""
        return (f"All AI keys are busy on their free tier right now{wait}. "
                "Try again shortly, or add another GROQ_API_KEY / GEMINI_API_KEY to your .env "
                "(comma-separate multiple keys) to add capacity.")
    if "PERMISSION_DENIED" in msg or "API_KEY_INVALID" in msg or "401" in msg or "403" in msg:
        return f"{provider_name()} rejected the API key — check it in your .env file."
    return f"{provider_name()} error: {msg[:200]}"


def _keys(env: str) -> list[str]:
    """A provider can hold several keys, comma-separated (rotate across free accounts)."""
    return [k.strip() for k in os.getenv(env, "").split(",") if k.strip()]


def _backends() -> list[dict]:
    """Ordered pool of (provider, model, key) to try — every key you've supplied.

    We try each Groq key (primary model then lighter fallbacks), then each Gemini
    key, then Claude. On a rate-limit/quota/too-large error we advance to the next
    entry, so one exhausted free tier transparently rolls over to another.
    """
    pool = []
    for key in _keys("GROQ_API_KEY"):
        pool.append({"provider": "groq", "model": config.GROQ_MODEL, "key": key})
        for m in getattr(config, "GROQ_FALLBACKS", []):
            pool.append({"provider": "groq", "model": m, "key": key})
    for key in _keys("GEMINI_API_KEY"):
        pool.append({"provider": "gemini", "model": config.GEMINI_MODEL, "key": key})
        for m in getattr(config, "GEMINI_FALLBACKS", []):
            pool.append({"provider": "gemini", "model": m, "key": key})
    for key in _keys("ANTHROPIC_API_KEY"):
        pool.append({"provider": "claude", "model": config.CLAUDE_MODEL, "key": key})
    return pool


def _provider() -> str | None:
    pool = _backends()
    return pool[0]["provider"] if pool else None


def has_key() -> bool:
    return bool(_backends())


def provider_name() -> str:
    return {"groq": "Groq", "gemini": "Gemini", "claude": "Claude"}.get(_provider(), "no model")


def _as_list(data, keys: list[str]) -> list:
    """Normalise a JSON payload to a list (JSON-mode models wrap arrays in an object)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in keys:
            if isinstance(data.get(k), list):
                return data[k]
        for v in data.values():           # fall back to the first list value present
            if isinstance(v, list):
                return v
    return []


# ---------- low-level completion (routes to the active provider) ----------
def _raw_gemini(model: str, key: str, system: str, user: str, max_tokens: int, json_out: bool) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_out else None,
    )
    resp = client.models.generate_content(model=model, contents=user, config=cfg)
    return resp.text or ""


def _raw_groq(model: str, key: str, system: str, user: str, max_tokens: int, json_out: bool) -> str:
    from groq import Groq
    client = Groq(api_key=key)
    kwargs = {"response_format": {"type": "json_object"}} if json_out else {}
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        **kwargs)
    return resp.choices[0].message.content or ""


def _raw_claude(model: str, key: str, system: str, user: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system, messages=[{"role": "user", "content": user}])
    return next(b.text for b in resp.content if b.type == "text")


def _recoverable(msg: str) -> bool:
    """True for errors where trying the next key/provider makes sense."""
    return any(t in msg for t in ("UNAVAILABLE", "503", "overloaded", "RESOURCE_EXHAUSTED",
                                   "429", "413", "too large", "tokens per minute", "rate limit",
                                   "RateLimit"))


def _run_backend(b: dict, system, user, max_tokens, json_out) -> str:
    if b["provider"] == "groq":
        return _raw_groq(b["model"], b["key"], system, user, max_tokens, json_out)
    if b["provider"] == "gemini":
        return _raw_gemini(b["model"], b["key"], system, user, max_tokens, json_out)
    return _raw_claude(b["model"], b["key"], system, user, max_tokens)


def _complete(system: str, user: str, max_tokens: int = 1024, json_out: bool = False) -> str:
    """Try each key/provider in the pool, rolling over on rate-limit/quota/too-large."""
    pool = _backends()
    if not pool:
        raise NoAPIKey()
    last = None
    for b in pool:
        try:
            return _run_backend(b, system, user, max_tokens, json_out)
        except Exception as e:
            last = e
            if _recoverable(str(e)):
                continue          # this key/model is tapped out — try the next in the pool
            raise AIError(_friendly_error(e)) from e   # hard error (bad key etc.)
    raise AIError(_friendly_error(last))


def _parse_json(raw: str):
    """Best-effort JSON parse: strips ```json fences, then falls back to the outermost {}/[]."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        raw = raw[4:] if raw.lstrip().startswith("json") else raw
        raw = raw.strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = min([i for i in (raw.find("["), raw.find("{")) if i != -1], default=-1)
        end = max(raw.rfind("]"), raw.rfind("}"))
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise


def _complete_json(system: str, user: str, max_tokens: int = 2048):
    """Completion that must return JSON. Retries once if the model returns bad JSON."""
    system = system + " Respond with ONLY valid JSON, no prose, no code fences."
    last_err = None
    for _ in range(2):
        raw = _complete(system, user, max_tokens, json_out=True)
        try:
            return _parse_json(raw)
        except json.JSONDecodeError as e:
            last_err = e   # transient — ask again once
    raise AIError("The model returned malformed data. Please try again.") from last_err


# ---------- 1) grounded Q&A (works without a key) ----------
SYSTEM_PROMPT = (
    "You are a study assistant. Answer the user's question using ONLY the context "
    "from their notes below. Each context block is tagged with its source like "
    "[filename, p.3] — when you use a fact, cite its source inline, e.g. (lecture1.pdf, p.3). "
    "If notes from different documents relate, compare them. If the answer is not in "
    "the notes, say so honestly. Be clear and concise."
)


def generate_answer(question: str, labeled_context: list[str]) -> str:
    """Build a grounded prompt from labeled chunks and ask the model to answer."""
    if not has_key():
        return (
            "⚠️ No API key set, so I can't generate an AI answer yet — but retrieval "
            "works! See the most relevant chunks from your notes below. Add a "
            "GEMINI_API_KEY (free) or ANTHROPIC_API_KEY to a .env file to enable AI answers."
        )
    context = "\n\n---\n\n".join(labeled_context)
    user = f"Context from my notes:\n\n{context}\n\nQuestion: {question}"
    try:
        return _complete(SYSTEM_PROMPT, user)
    except AIError as e:
        return f"⚠️ {e}"


def generate_chat(question: str, labeled_context: list[str], history: list[dict]) -> str:
    """Grounded answer that also sees the recent conversation (for follow-ups)."""
    if not has_key():
        return (
            "⚠️ No API key set — add a GROQ_API_KEY (free) to a .env file to chat with your notes. "
            "Retrieval still works; see the sources below."
        )
    context = "\n\n---\n\n".join(labeled_context)
    convo = ""
    for m in history[-6:]:                      # keep the last few turns for context
        who = "Student" if m.get("role") == "user" else "Assistant"
        convo += f"{who}: {m.get('content', '')}\n"
    user = (
        f"Context from my notes:\n\n{context}\n\n"
        f"Conversation so far:\n{convo}\nStudent: {question}"
    )
    try:
        return _complete(SYSTEM_PROMPT, user)
    except AIError as e:
        return f"⚠️ {e}"


# ---------- 2) quiz generator (MCQs) ----------
def generate_quiz(labeled_context: list[str], num: int = 5) -> list[dict]:
    """Return [{question, options[4], answer_index, explanation}] built ONLY from the notes."""
    if not has_key():
        raise NoAPIKey()
    context = "\n\n---\n\n".join(labeled_context)
    system = (
        "You are a quiz author. Create multiple-choice questions using ONLY the "
        "provided notes — never test facts that aren't in them."
    )
    user = (
        f"From these notes, write {num} multiple-choice questions. Each must have exactly "
        "4 options with ONE correct. Return a JSON array of objects with keys: "
        '"question" (string), "options" (array of 4 strings), "answer_index" (0-3), '
        '"explanation" (short, why the answer is right, grounded in the notes).\n\n'
        f"Notes:\n{context}"
    )
    data = _complete_json(system, user, max_tokens=3000)
    return _as_list(data, ["questions", "quiz", "mcqs"])


# ---------- 3) flashcard generator ----------
def generate_flashcards(labeled_context: list[str], num: int = 10) -> list[dict]:
    """Return high-quality active-recall cards built ONLY from the notes.

    Each card: {question, answer, difficulty, topic, est_time, source_page, hint, explanation}.
    Context blocks are tagged [filename, p.N] so the model can fill in source_page.
    """
    if not has_key():
        raise NoAPIKey()
    context = "\n\n---\n\n".join(labeled_context)
    system = (
        "You are an expert study-flashcard author in the style of GeeksforGeeks. "
        "Create high-quality flashcards using ONLY the provided notes (never invent facts)."
    )
    user = (
        f"Create {num} flashcards from these notes. Follow ALL rules strictly:\n"
        "1. One concept per card. No duplicate cards.\n"
        "2. Use ACTIVE RECALL questions. Never keyword-only cards. No yes/no questions.\n"
        "3. Questions must be clear and specific.\n"
        "4. Answers UNDER 40 words. Use markdown bullets ('- ') for lists; code/SQL in backticks.\n"
        "5. Vary card TYPES when appropriate: definition, comparison, scenario, code, "
        "fill-in-the-blank, and application-based.\n"
        "6. Prioritise concepts most likely to appear in EXAMS and INTERVIEWS.\n"
        "Each card is a JSON object with keys:\n"
        '  "question" (string), "answer" (string, <40 words),\n'
        '  "topic" (short label), "difficulty" ("Easy"|"Medium"|"Hard"),\n'
        '  "est_time" (integer seconds to recall, 10-60),\n'
        '  "source_page" (integer page from the [file, p.N] tags this came from),\n'
        '  "hint" (short optional nudge, empty string if none),\n'
        '  "explanation" (one short sentence on why the answer is correct).\n'
        "Return ONLY a JSON array of these objects.\n\n"
        f"Notes:\n{context}"
    )
    data = _complete_json(system, user, max_tokens=4500)
    return _as_list(data, ["flashcards", "cards"])


# ---------- 4) cheat-sheet / summary ----------
def generate_summary(labeled_context: list[str]) -> str:
    """A compact, well-structured revision cheat-sheet for a document."""
    if not has_key():
        raise NoAPIKey()
    context = "\n\n---\n\n".join(labeled_context)
    system = (
        "You are a study assistant making a revision cheat-sheet. Use ONLY the notes. "
        "Output tight markdown: a one-line overview, then key points as bullets, "
        "and bold the most important terms."
    )
    return _complete(system, f"Summarise these notes into a cheat-sheet:\n\n{context}", max_tokens=1500)


# ---------- 5) 'test me' answer grading ----------
def grade_answer(question: str, user_answer: str, labeled_context: list[str]) -> dict:
    """Grade the student's free-text answer against the notes.

    Returns {score: 0-5, verdict: str, feedback: str, ideal_answer: str}.
    """
    if not has_key():
        raise NoAPIKey()
    context = "\n\n---\n\n".join(labeled_context)
    system = (
        "You are a fair examiner. Grade the student's answer ONLY against the provided "
        "notes (the notes are the source of truth). Be encouraging but honest."
    )
    user = (
        f"Question: {question}\n\nStudent's answer: {user_answer}\n\n"
        f"Reference notes:\n{context}\n\n"
        'Return JSON with keys: "score" (integer 0-5, how well the answer matches the notes), '
        '"verdict" (one of "correct", "partially correct", "incorrect"), '
        '"feedback" (2-3 sentences, what was right/missing), '
        '"ideal_answer" (the correct answer from the notes).'
    )
    return _complete_json(system, user, max_tokens=1200)
