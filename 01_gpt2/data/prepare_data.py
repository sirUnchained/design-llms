import asyncio
import aiohttp
import hashlib
import json
import sys
import time
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from typing import List, Optional, Dict, Tuple

# Force line-buffered (flushed on every newline) stdout. When output isn't
# attached to a live terminal -- redirected to a file, piped through another
# tool, run inside some wrapper/runner -- Python silently switches stdout to
# block buffering, so print() lines sit in memory and only appear once the
# buffer fills or the process exits. That's what made progress look
# "missing" (it was actually just delayed and dumped all at once at the
# end). Reconfiguring here guarantees every print() is flushed immediately,
# in any environment.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass  # very old Python without reconfigure(); flush=True calls below still work

# If True, every fetch attempt logs its own line (✅/🚫/⛔/etc). With large
# URL lists (thousands+) this floods the console/log and buries the actual
# progress signal, so it's off by default -- turn it on for debugging a
# small batch. Progress is always shown separately regardless of this flag.
VERBOSE_PER_URL_LOGS = False

# How often to print a plain-text progress line (every N completed URLs).
# This does NOT rely on carriage-return redraws (unlike a tqdm bar), so it
# shows up reliably in redirected output, log files, or any non-interactive
# runner -- not just a live terminal.
PROGRESS_LOG_EVERY = 50

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
OUTPUT_FILE = "llm_dataset.jsonl"
REQUEST_TIMEOUT = 15
MAX_CONCURRENT = 5  # global concurrency cap across all domains
MAX_RETRIES = 3
RETRY_BACKOFF = 2
USER_AGENT = "LLM Dataset Builder/1.0 (+https://myproject.org/bot)"  # honest UA
TOKENIZER_NAME = "gpt2"
MAX_CONTENT_BYTES = 5_000_000  # skip/truncate anything larger than ~5MB of HTML
DEFAULT_CRAWL_DELAY = 1.0  # seconds, used when robots.txt gives no crawl-delay
COUNT_TOKENS = False

# -------------------------------------------------------------------
# Tokenization (post-scrape)
# -------------------------------------------------------------------
# After scraping finishes, the resulting JSONL is tokenized straight to a
# uint16 binary file next to it -- this is now the ONLY place tokenization
# happens. Training (main.py / src/training/train.py) never tokenizes; it
# just expects this .bin to already exist and fails fast with instructions
# if it doesn't. Run this script once whenever your dataset changes.
TOKENIZE_TO_BIN = True
TOKENIZE_CHUNK_CHARS = 50_000_000  # ~50MB of text per chunk while tokenizing

# License allowlist (URLs and keywords).
#
# Deliberately restricted to CC0 / Public Domain and CC-BY. These are the only
# tiers with no conditions beyond (for CC-BY) attribution -- no share-alike,
# no non-commercial, no no-derivatives. CC-BY-SA is intentionally excluded:
# its share-alike clause could obligate you to release derivative works
# (arguably including a model trained on the data) under the same license,
# which isn't "do anything you want."
#
# NOTE: matching "cc-by" as a substring would also match "cc-by-sa" and
# "cc-by-nc", so each accepted variant is listed explicitly rather than
# relying on a short prefix.
ALLOWED_LICENSES = {
    "cc0",
    "public domain",
    "creativecommons.org/publicdomain/zero/",
}

# CC-BY variants matched as exact license strings/URLs, not substrings, so
# CC-BY-SA / CC-BY-NC / CC-BY-ND don't accidentally slip through.
ALLOWED_LICENSE_URL_PREFIXES = {
    "creativecommons.org/licenses/by/",  # CC-BY only, any version
}
ALLOWED_LICENSE_EXACT_PHRASES = {
    "cc-by",
    "cc by",
    "attribution 4.0",
    "attribution 3.0",
}
# Phrases that, if present alongside a match above, disqualify it (catches
# "CC-BY-SA" and "CC-BY-NC" being loosely matched by "cc-by").
DISQUALIFYING_PHRASES = {
    "-sa",
    " sa",
    "sharealike",
    "share-alike",
    "-nc",
    " nc",
    "noncommercial",
    "non-commercial",
    "-nd",
    " nd",
    "noderivatives",
    "no derivatives",
}

FILTER_BY_LICENSE = True

# Text-based license "guesses" (no explicit meta/link tag found) are inherently
# unreliable -- a page can mention "public domain" in a footer, a quote, or in
# reference to something else entirely. Mislabeling license data is a real risk
# for a training set, so heuristic matches are excluded by default. Flip this on
# only if you plan to manually review flagged documents before use.
ALLOW_HEURISTIC_LICENSE = False

# Respect AI-specific opt-out signals in addition to robots.txt Disallow rules
# (X-Robots-Tag header and <meta name="robots"> content).
RESPECT_AI_OPT_OUT = True
AI_OPT_OUT_TOKENS = {"noai", "noimageai", "noindex"}


# -------------------------------------------------------------------
# robots.txt handling (via stdlib robotparser -- correctly handles
# Allow/Disallow precedence, wildcards, and per-agent groups)
# -------------------------------------------------------------------
class RobotsCache:
    def __init__(self):
        self._cache: Dict[str, Tuple[RobotFileParser, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, domain: str, session: aiohttp.ClientSession):
        async with self._lock:
            if domain in self._cache:
                return self._cache[domain]

        robots_url = f"https://{domain}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)

        try:
            async with session.get(robots_url, timeout=5) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    rp.parse(text.splitlines())
                else:
                    # No robots.txt (or inaccessible) -- per convention, treat as
                    # "no restrictions" rather than failing closed or open blindly.
                    rp.parse([])
        except Exception:
            rp.parse([])

        delay = rp.crawl_delay(USER_AGENT) or rp.crawl_delay("*") or DEFAULT_CRAWL_DELAY

        async with self._lock:
            self._cache[domain] = (rp, delay)
        return self._cache[domain]


robots_cache = RobotsCache()


# -------------------------------------------------------------------
# Per-domain throttling: serializes all requests (incl. retries) to the
# same host so crawl-delay is actually honored, even with many concurrent
# tasks in flight for other domains.
# -------------------------------------------------------------------
class DomainThrottle:
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def get_lock(self, domain: str) -> asyncio.Lock:
        async with self._registry_lock:
            if domain not in self._locks:
                self._locks[domain] = asyncio.Lock()
            return self._locks[domain]


domain_throttle = DomainThrottle()


# -------------------------------------------------------------------
# License detection
# -------------------------------------------------------------------
def detect_license(soup: BeautifulSoup, url: str) -> Tuple[Optional[str], bool]:
    """Returns (license_string, verified). verified=True means an explicit
    machine-readable signal was found (meta/link tag or known-domain rule),
    as opposed to a heuristic guess from body text."""

    meta = soup.find("meta", attrs={"name": "dcterms.license"})
    if meta and meta.get("content"):
        return meta["content"], True

    meta = soup.find("meta", attrs={"property": "cc:license"})
    if meta and meta.get("content"):
        return meta["content"], True

    link = soup.find("link", rel="license")
    if link and link.get("href"):
        return link["href"], True

    if "wikipedia.org" in url:
        return "https://creativecommons.org/licenses/by-sa/4.0/", True

    # Project Gutenberg: every ebook page carries their standard license
    # stating the underlying text is public domain in the US (their own
    # header/footer boilerplate and trademark terms are not the content
    # itself). Treated as verified rather than relying on the generic
    # "public domain" body-text heuristic.
    if "gutenberg.org" in url:
        return "Public Domain (Project Gutenberg)", True

    body = soup.get_text().lower()
    if "creative commons attribution" in body:
        return "CC-BY (heuristic)", False
    if "public domain" in body:
        return "Public Domain (heuristic)", False

    return None, False


def is_license_allowed(license_str: Optional[str], verified: bool) -> bool:
    if not license_str:
        return False
    if not verified and not ALLOW_HEURISTIC_LICENSE:
        return False

    license_lower = license_str.lower()

    # Disqualify first: catches "cc-by-sa", "cc-by-nc", "cc-by-nd" etc. even
    # though they contain "cc-by" / the by/ URL prefix as a substring.
    if any(bad in license_lower for bad in DISQUALIFYING_PHRASES):
        return False

    if any(allowed in license_lower for allowed in ALLOWED_LICENSES):
        return True
    if any(prefix in license_lower for prefix in ALLOWED_LICENSE_URL_PREFIXES):
        return True
    if any(phrase in license_lower for phrase in ALLOWED_LICENSE_EXACT_PHRASES):
        return True

    return False


# -------------------------------------------------------------------
# AI opt-out / indexing signals
# -------------------------------------------------------------------
def has_ai_opt_out(soup: BeautifulSoup, headers) -> bool:
    if not RESPECT_AI_OPT_OUT:
        return False

    header_value = headers.get("X-Robots-Tag", "").lower()
    if any(tok in header_value for tok in AI_OPT_OUT_TOKENS):
        return True

    meta = soup.find("meta", attrs={"name": "robots"})
    if meta and meta.get("content"):
        content = meta["content"].lower()
        if any(tok in content for tok in AI_OPT_OUT_TOKENS):
            return True

    return False


# -------------------------------------------------------------------
# Text extraction
# -------------------------------------------------------------------
def extract_main_text(soup: BeautifulSoup) -> Optional[str]:
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
    ):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="content")
        or soup.find("div", id="content")
        or soup.find("body")
    )

    if not main:
        return None

    parts = []
    for elem in main.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        text = elem.get_text(strip=True)
        if len(text) > 30:
            parts.append(text)

    if not parts:
        return None

    full = "\n\n".join(parts)
    full = "\n".join(line.strip() for line in full.splitlines() if line.strip())
    return full if len(full) > 200 else None


def content_hash(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------
# Write one record as a JSON line (append)
# -------------------------------------------------------------------
def append_jsonl(filepath: str, record: dict):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------------------------------------------------------------
# Asynchronous scraper
# -------------------------------------------------------------------
def vlog(msg: str):
    """Per-URL detail logging -- gated behind VERBOSE_PER_URL_LOGS so it
    doesn't bury the progress line on large URL lists. Set the flag to True
    to see every ✅/🚫/⛔/etc. outcome as it happens."""
    if VERBOSE_PER_URL_LOGS:
        print(msg)


async def scrape_one(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    seen_hashes: set,
    hashes_lock: asyncio.Lock,
    output_file: str,
    write_lock: asyncio.Lock,
) -> bool:
    """Fetches, filters and (on success) immediately appends one record to
    output_file. Returns True/False instead of the record itself, since the
    record is written straight to disk and never held in memory."""
    parsed = urlparse(url)
    domain = parsed.netloc

    rp, delay = await robots_cache.get(domain, session)
    if not rp.can_fetch(USER_AGENT, url):
        vlog(f"⛔ {url} – disallowed by robots.txt")
        return False

    domain_lock = await domain_throttle.get_lock(domain)

    async with semaphore, domain_lock:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(
                    url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
                ) as resp:
                    if resp.status == 429:
                        retry_after = resp.headers.get("Retry-After")
                        wait = (
                            float(retry_after)
                            if retry_after
                            else RETRY_BACKOFF**attempt
                        )
                        vlog(f"⏳ {url} – 429 rate limited, waiting {wait}s")
                        await asyncio.sleep(wait)
                        continue

                    if resp.status != 200:
                        vlog(
                            f"⚠️ {url} – HTTP {resp.status} (attempt {attempt}/{MAX_RETRIES})"
                        )
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(RETRY_BACKOFF**attempt)
                        continue

                    content_type = resp.headers.get("Content-Type", "")
                    if (
                        "text/html" not in content_type
                        and "application/xhtml" not in content_type
                    ):
                        vlog(
                            f"🚫 {url} – non-HTML content-type: {content_type or 'unknown'}"
                        )
                        return False

                    content_length = resp.headers.get("Content-Length")
                    if content_length and int(content_length) > MAX_CONTENT_BYTES:
                        vlog(f"🚫 {url} – too large ({content_length} bytes), skipping")
                        return False

                    html = await resp.text()
                    if len(html) > MAX_CONTENT_BYTES:
                        html = html[:MAX_CONTENT_BYTES]

                    soup = BeautifulSoup(html, "html.parser")

                    if has_ai_opt_out(soup, resp.headers):
                        vlog(f"🚫 {url} – AI/indexing opt-out signal present")
                        return False

                    lic, verified = detect_license(soup, url)
                    if FILTER_BY_LICENSE and not is_license_allowed(lic, verified):
                        vlog(
                            f"🚫 {url} – license not allowed: {lic} (verified={verified})"
                        )
                        return False

                    text = extract_main_text(soup)
                    if not text:
                        vlog(f"📭 {url} – no text extracted")
                        return False

                    h = content_hash(text)
                    async with hashes_lock:
                        if h in seen_hashes:
                            vlog(f"♻️ {url} – duplicate content, skipping")
                            return False
                        seen_hashes.add(h)

                    record = {
                        "url": url,
                        "license": lic,
                        "license_verified": verified,
                        "fetched_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "content_hash": h,
                        "text": text,
                    }

                    # Write straight to disk as soon as it's ready -- results
                    # are never accumulated in memory.
                    async with write_lock:
                        append_jsonl(output_file, record)

                    vlog(
                        f"✅ {url} – {len(text)} chars, license: {lic} (verified={verified})"
                    )
                    return True

            except asyncio.TimeoutError:
                vlog(f"⌛ {url} – timeout ({attempt}/{MAX_RETRIES})")
            except Exception as e:
                vlog(f"❌ {url} – {type(e).__name__}: {e}")

            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BACKOFF**attempt)

        vlog(f"💀 {url} – failed after {MAX_RETRIES} attempts")
        return False


# -------------------------------------------------------------------
# Load already-scraped URLs/hashes from an existing JSONL file, so re-runs
# resume instead of re-downloading and re-storing what's already there.
# -------------------------------------------------------------------
def load_existing_records(output_file: str) -> Tuple[set, set]:
    existing_urls: set = set()
    existing_hashes: set = set()

    if not os.path.exists(output_file):
        return existing_urls, existing_hashes

    with open(output_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ {output_file}:{line_num} – skipping malformed JSON line")
                continue
            if "url" in record:
                existing_urls.add(record["url"])
            if "content_hash" in record:
                existing_hashes.add(record["content_hash"])

    return existing_urls, existing_hashes


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
async def scrape_urls(urls: List[str], output_file: str):
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    # Resume support: never re-download a URL that's already in the file,
    # and never re-store content whose hash is already present.
    existing_urls, seen_hashes = load_existing_records(output_file)
    to_fetch = [u for u in unique if u not in existing_urls]
    skipped = len(unique) - len(to_fetch)

    print(
        f"📋 {len(unique)} unique URLs ({skipped} already in '{output_file}', "
        f"{len(to_fetch)} to fetch; max concurrency={MAX_CONCURRENT}, "
        f"per-domain requests serialized to respect crawl-delay)",
        flush=True,
    )

    if not to_fetch:
        print("🎉 Nothing new to fetch.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    hashes_lock = asyncio.Lock()
    write_lock = asyncio.Lock()

    # Append to the existing file rather than truncating it -- previously
    # scraped records are preserved across runs.
    if not os.path.exists(output_file):
        open(output_file, "w", encoding="utf-8").close()

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 2)
    count = 0
    completed = 0
    total = len(to_fetch)
    start_time = time.monotonic()

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            scrape_one(
                session,
                url,
                semaphore,
                seen_hashes,
                hashes_lock,
                output_file,
                write_lock,
            )
            for url in to_fetch
        ]
        # as_completed (not gather) so progress reflects URLs as they finish,
        # regardless of which order they land in. We print a plain text line
        # every PROGRESS_LOG_EVERY completions (not a carriage-return bar),
        # so it shows up reliably in redirected output, logs, or any
        # non-interactive runner -- not just a live terminal.
        for coro in asyncio.as_completed(tasks):
            ok = await coro
            completed += 1
            if ok:
                count += 1

            if completed % PROGRESS_LOG_EVERY == 0 or completed == total:
                elapsed = time.monotonic() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta_sec = (total - completed) / rate if rate > 0 else 0
                pct = completed / total * 100
                print(
                    f"📊 Progress: {completed}/{total} ({pct:.1f}%) – "
                    f"{count} saved – {rate:.1f} url/s – "
                    f"ETA {eta_sec / 60:.1f} min",
                    flush=True,
                )

    print(
        f"\n🎉 Done. {count} new documents appended to '{output_file}' (JSONL, one record per line)."
    )


def read_urls_from_file(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


# -------------------------------------------------------------------
# Tokenization: JSONL -> uint16 binary, single streaming pass
# -------------------------------------------------------------------
def get_bin_path(jsonl_path: str) -> str:
    """
    ## Derive the tokenized binary path for a given JSONL dataset path.

    Keeps the tokenized cache next to the dataset file, same name, `.bin`
    extension. So `./data/llm_dataset.jsonl` maps to `./data/llm_dataset.bin`.
    This must match `get_bin_path` in `src/data/dataset.py` exactly, since
    that's where training looks for the file this function produces.

    ---

    Args:
        jsonl_path (str): Path to the scraped `.jsonl` dataset file.

    Returns:
        str: Path to the corresponding `.bin` tokenized file.
    """
    root, _ = os.path.splitext(jsonl_path)
    return root + f"_{TOKENIZER_NAME}_tokenizer" + ".bin"


def _iter_jsonl_text_chunks(jsonl_path: str, chunk_chars: int):
    """
    ## Yield bounded-size text chunks from a scraped JSONL dataset.

    Reads the JSONL file line by line, extracting the `"text"` field of
    each record and accumulating it into a buffer, yielding once the buffer
    reaches `chunk_chars`. Only one chunk's worth of text is held in memory
    at a time, so this scales to arbitrarily large JSONL files.

    ---

    Args:
        jsonl_path (str): Path to the scraped `.jsonl` dataset file.
        chunk_chars (int): Number of characters to accumulate per yielded chunk.

    Yields:
        str: Successive chunks of concatenated record text, each up to
            ~`chunk_chars` long.
    """
    buf = ""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            buf += record["text"] + "\n"
            if len(buf) >= chunk_chars:
                yield buf
                buf = ""
    if buf:
        yield buf


def tokenize_to_bin(
    jsonl_path: str,
    out_path: str,
    tokenizer_name: str = TOKENIZER_NAME,
    chunk_chars: int = TOKENIZE_CHUNK_CHARS,
) -> int:
    """
    ## Tokenize a scraped JSONL dataset into a binary token file, in a single pass.

    Reads the dataset in bounded-size text chunks (see
    `_iter_jsonl_text_chunks`), encodes each chunk with tiktoken, and appends
    the resulting `uint16` token ids straight onto the end of `out_path` as
    raw bytes. A plain file handle in append-binary mode grows on disk as we
    write, so there's no need to know the total token count up front and no
    need for a second counting pass -- peak RAM stays bounded by
    `chunk_chars` regardless of dataset size, and the corpus is tokenized
    exactly once.

    The resulting file has the same on-disk layout a `numpy.memmap` of dtype
    `uint16` would produce, so training reads it back with
    `np.memmap(out_path, dtype=np.uint16, mode="r")` -- see
    `MemmapGPTDataset` / `ensure_bin_dataset` in `src/data/dataset.py`.

    `uint16` is safe for the GPT-2 tokenizer since `vocab_size` (50257) fits
    under 65536, and it halves storage compared to `int64`.

    This is meant to be run once, as part of this data-prep script, never
    during training.

    ---

    Args:
        jsonl_path (str):
            Path to the scraped `.jsonl` dataset (as produced by `scrape_urls`).
        out_path (str):
            Path where the resulting binary token file will be written.
        tokenizer_name (str, optional):
            Name of the tiktoken encoding to use. Default is `TOKENIZER_NAME`.
        chunk_chars (int, optional):
            Number of characters to accumulate before encoding and writing a
            chunk. Default is `TOKENIZE_CHUNK_CHARS`.

    Returns:
        int: Total number of tokens written to `out_path`. `0` if
            `jsonl_path` doesn't exist or contains no records.
    """
    if not os.path.exists(jsonl_path):
        print(f"⚠️ {jsonl_path} not found, skipping tokenization.")
        return 0

    import tiktoken
    import numpy as np

    enc = tiktoken.get_encoding(tokenizer_name)
    total_len = 0

    print(f"🔤 Tokenizing '{jsonl_path}' -> '{out_path}' ...", flush=True)

    with open(out_path, "wb") as out_f:
        for chunk in _iter_jsonl_text_chunks(jsonl_path, chunk_chars):
            ids = enc.encode_ordinary(chunk)
            arr = np.array(ids, dtype=np.uint16)
            out_f.write(arr.tobytes())
            total_len += arr.size

    if total_len:
        print(f"✅ Wrote {total_len:,} tokens to '{out_path}'")
    else:
        print(
            f"⚠️ '{jsonl_path}' produced 0 tokens (empty file?), '{out_path}' is empty."
        )

    return total_len


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        print(f"Usage: python prepare_data.py urls.txt [output.jsonl]")
        sys.exit(1)

    url_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else OUTPUT_FILE

    if not os.path.exists(url_file):
        print(f"File not found: {url_file}")
        sys.exit(1)

    urls = read_urls_from_file(url_file)
    if not urls:
        print("No URLs found.")
        sys.exit(1)

    asyncio.run(scrape_urls(urls, output_file))

    if TOKENIZE_TO_BIN:
        bin_path = get_bin_path(output_file)
        tokenize_to_bin(output_file, bin_path)
