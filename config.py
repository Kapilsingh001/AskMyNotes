"""All the tunable settings live here in one place."""

# --- Chunking: how we cut notes into pieces ---
CHUNK_SIZE = 200       # ~words per chunk (a chunk is one "unit" we embed & retrieve)
CHUNK_OVERLAP = 40     # words shared between neighbouring chunks (keeps context across cuts)

# --- Retrieval: how many chunks we send to the AI ---
TOP_K = 4              # how many most-relevant chunks to feed the model per question

# --- Embedding model (runs LOCALLY, free, no API key) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, 384-dim vectors

# --- Which AI generates the answers ---
# The app auto-picks a provider by which key is in your .env (in this priority):
#   GROQ_API_KEY    -> Groq (fast, generous free tier — used if present)
#   GEMINI_API_KEY  -> Google Gemini (free tier, small daily quota)
#   ANTHROPIC_API_KEY -> Claude (paid)
GROQ_MODEL = "llama-3.3-70b-versatile" # used if you have a GROQ_API_KEY (fast, high quality, free)
# If the primary Groq model is rate-limited (429), fall back to a lighter model
# with a much higher free per-minute limit:
GROQ_FALLBACKS = ["llama-3.1-8b-instant"]
CLAUDE_MODEL = "claude-opus-4-8"       # used if you have an ANTHROPIC_API_KEY
GEMINI_MODEL = "gemini-2.5-flash-lite" # used if you have a GEMINI_API_KEY (best free-tier daily quota)
# If the primary model is overloaded (503) or out of quota (429), auto-fall back to these:
GEMINI_FALLBACKS = ["gemini-2.5-flash", "gemini-flash-latest"]
