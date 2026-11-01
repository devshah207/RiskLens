"""Private local Phase 7 credential configuration check. No network calls."""
import os
from risklens.research.config import load_project_env


def main():
    path = load_project_env()
    print("Project .env:", "present" if path.is_file() else "missing")
    for key in ("GROQ_API_KEY", "PINECONE_API_KEY"):
        print(key, "configured" if os.environ.get(key) else "missing")
    print("PINECONE_INDEX_NAME:", os.environ.get("PINECONE_INDEX_NAME", "missing"))
    if not all(os.environ.get(k) for k in ("GROQ_API_KEY", "PINECONE_API_KEY")):
        raise SystemExit("ERROR: Missing API credentials. Never paste them in chat or logs.")
    print("PASS: local credentials readable; no external calls performed")


if __name__ == "__main__":
    main()
