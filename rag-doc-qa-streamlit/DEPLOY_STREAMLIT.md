# Deploying to Streamlit Community Cloud

This folder is ready to push to GitHub and deploy directly — unzip it straight
into the root of your repo (don't nest it inside another folder).

## 1. Push to GitHub
```
git init
git add .
git commit -m "Streamlit-ready RAG Document Q&A"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 2. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "Create app" → "Use existing repo".
3. Select your repo, branch `main`, and **main file path: `app.py`**.
4. (Optional) Set a custom subdomain in "App URL".
5. **IMPORTANT — set the Python version manually.** Open **"Advanced settings"**
   before deploying and pick **Python 3.11** from the dropdown.
   `runtime.txt` is included in this repo as a hint, but Streamlit Community
   Cloud has a known bug where it's sometimes silently ignored, defaulting to
   the newest Python (3.13/3.14) instead — which breaks several of this
   project's pinned dependencies (pydantic-core, Pillow, torch) since they
   don't have prebuilt wheels for that version yet.
6. **Python version can't be changed after first deploy.** If your app is
   already deployed and stuck on the wrong Python version, you must delete
   the app (⋮ menu → Delete app) and recreate it, picking Python 3.11 in
   Advanced settings this time. You'll need to re-add your `GROQ_API_KEY`
   secret afterward since deleting wipes it.

## 3. Add your API key
1. Open your deployed app → click the **⋮** menu → **Settings** → **Secrets**.
2. Paste:
   ```
   GROQ_API_KEY = "your_actual_key_here"
   ```
3. Save — the app reboots automatically with the key available.

## Notes
- This app uses ChromaDB's `EphemeralClient` (in-memory), so it does NOT rely on
  persistent disk storage. This is exactly what you want on Streamlit Cloud, since
  the filesystem there is also ephemeral and resets on every reboot/redeploy.
- Each browser session gets its own isolated vector store — uploaded docs are never
  shared between users and disappear on refresh, by design.
- `app.py` reads the API key from `st.secrets` first, falling back to environment
  variables — so the exact same code also works unmodified on Hugging Face Spaces.
- The leftover `src/streamlit_app.py` default-template file from the original zip
  has been removed — `app.py` is the only entry point you need.
