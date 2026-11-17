# LinkedIn Post Automation Agent

Discover emerging technical projects, generate a short LinkedIn description with Gemini, check its claims against source text, and capture a screenshot. Optionally publish using a local LinkedIn browser session.

## Setup

Use Python 3.10 or newer. Run from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install requests beautifulsoup4 python-dotenv google-genai ddgs playwright Pillow
python -m playwright install chromium
```

The repository includes an empty `.env`. Before adding credentials, keep your local changes out of normal Git staging:

```powershell
git update-index --skip-worktree .env
```

Then edit `.env` locally:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
GITHUB_TOKEN=
GITHUB_MAX_AGE_DAYS=30
GITHUB_MIN_STARS=20
GITHUB_MAX_STARS=5000
```

`GITHUB_TOKEN` is optional. You can also set `GEMINI_MODEL` to override the default model. Never commit a populated `.env` or the `linkedin_session/` folder. The skip-worktree setting is local to each clone; it does not encrypt or protect secrets outside Git staging.

## Generate a post

```powershell
python post_agent2.py
```

Choose GitHub, Web, Both, or enter a tool name or URL. You can also supply a name or URL directly:

```powershell
python post_agent2.py https://github.com/owner/repository
```

GitHub discovery defaults to repositories created within 30 days, pushed within 14 days, and having 20-5,000 stars. Web discovery collects Hacker News, Hugging Face papers and models, Google News, and Reddit/blog search results. A balanced shortlist gives each source representation before Gemini chooses a topic and explains why.

The agent fetches the source, extracts facts, writes three short body paragraphs and a link (at most 120 words), and reviews the draft against the original text. Unsupported drafts are retried up to three times. This model-assisted review improves grounding but does not guarantee factual accuracy.

Generated files:

- `posts/<tool>_post.txt` and `post.txt`
- `posts/ss_<tool>.png` and, when capture succeeds, `screenshot.png`

## Optional LinkedIn publishing

At the agent's final prompt, choose `N` to keep the files locally. Choosing `y` opens the browser and submits the post automatically after login if needed.

For standalone browser review:

```powershell
python linkedin_publisher.py --post posts/example_post.txt --image posts/ss_example.png --manual
```

The `--manual` option pauses for terminal confirmation before submission. Use matching post/image paths. To initialize login separately:

```powershell
python linkedin_publisher.py --login
```

Browser authentication is stored locally in `linkedin_session/`. Never share that folder.

## Current limitations

- `posted.json` records selected URLs before generation succeeds; it is selection history, not confirmed publishing history.
- Article links are not automatically mapped to repositories discussed inside them.
- Publication dates are not consistently available for web results.
- The publisher currently reports success after clicking Post; verify the result in LinkedIn.
- Dependencies are unpinned; the repository does not yet provide a reproducible lockfile.
