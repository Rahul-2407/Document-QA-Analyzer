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

Python version is already pinned via `runtime.txt` (3.11), so you shouldn't need
to touch "Advanced settings" — this avoids the torch/numpy build errors that
happen on newer Python versions (3.13/3.14) with these older pinned packages.

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
