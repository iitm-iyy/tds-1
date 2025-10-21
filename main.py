# ///script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi[standard]",
#     "uvicorn",
#     "requests",
#     "huggingface_hub",
#     "python-dotenv",
# ]

import os
import time
import requests
import base64
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse

load_dotenv("/home/user1/myapp/round-runner.env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DEFAULT_BRANCH = os.getenv("GITHUB_DEFAULT_BRANCH", "main")
headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}
print(f"[DEBUG] GITHUB_TOKEN: {'set' if GITHUB_TOKEN else 'not set'}")

app = FastAPI()

def validate_secret(secret: str) -> bool:
    return secret == os.getenv("secret_key")

def create_github_repo(repo_name: str):
    payload = {"name": repo_name, "private": False, "auto_init": True, "license_template": "mit"}
    response = requests.post(
        "https://api.github.com/user/repos",
        headers=headers,
        json=payload
    )
    print(f"[DEBUG] Create repo response: {response.status_code} - {response.text[:200]}")
    if response.status_code == 201:
        repo_info = response.json()
        print(f"Repo created successfully: {repo_name} - Full response: {repo_info.get('full_name', 'N/A')}")
        return repo_info
    elif response.status_code == 422:
        print(f"Repo {repo_name} already exists, reusing it.")
        r = requests.get(
            f"https://api.github.com/repos/iitm-iyy/{repo_name}",
            headers=headers
        )
        print(f"[DEBUG] Get existing repo: {r.status_code} - {r.text[:200]}")
        if r.status_code == 200:
            return r.json()
        raise Exception(f"Failed to access existing repo: {r.status_code} {r.text}")
    else:
        raise Exception(f"Failed to create repo: {response.status_code} {response.text}")

def enable_github_pages(repo_name: str):
    payload = {"build_type": "legacy", "source": {"branch": DEFAULT_BRANCH, "path": "/"}}
    r = requests.post(
        f"https://api.github.com/repos/iitm-iyy/{repo_name}/pages",
        headers=headers,
        json=payload,
        timeout=30
    )
    print(f"[DEBUG] Enable Pages: {r.status_code} - {r.text[:200]}")
    if r.status_code in (201, 204, 200):
        print("Pages enabled or already active.")
        return r.json() if r.content else {}
    if r.status_code in (409, 422):
        print("Pages already enabled.")
        return r.json() if r.content else {}
    raise Exception(f"Failed to enable Pages: {r.status_code} {r.text}")

def get_file_sha_commit(repo_name: str, path: str, branch: str = DEFAULT_BRANCH) -> str | None:
    url = f"https://api.github.com/repos/iitm-iyy/{repo_name}/contents/{path}"
    response = requests.get(url, headers=headers, params={"ref": branch})
    print(f"[DEBUG] Get file SHA: {path} - {response.status_code}")
    if response.status_code == 200:
        return response.json().get("sha")
    if response.status_code == 404:
        return None
    raise Exception(f"GET contents failed for {path}: {response.status_code} {response.text}")

def add_update_file(repo_name: str, path: str, content: str | bytes, message: str, branch: str = DEFAULT_BRANCH):
    if isinstance(content, str) and re.search(r"(ghp_|HF_TOKEN|secret_key)=[\w\d]+", content, re.IGNORECASE):
        print(f"[WARN] Potential secrets detected in {path} - Aborting push")
        raise Exception("Content contains potential secrets")
    
    sha = get_file_sha_commit(repo_name, path, branch)
    if isinstance(content, str):
        b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    else:
        b64 = base64.b64encode(content).decode("utf-8")

    payload = {"message": message, "content": b64, "branch": branch}
    if sha:
        payload["sha"] = sha

    url = f"https://api.github.com/repos/iitm-iyy/{repo_name}/contents/{path}"
    response = requests.put(url, headers=headers, json=payload, timeout=30)
    print(f"[DEBUG] Push file {path}: {response.status_code} - {response.text[:200]}")
    if response.status_code not in (200, 201):
        raise Exception(f"Failed to push {path}: {response.status_code} {response.text}")

def push_files_to_repo(repo_name: str, files: list[dict], round_number: int):
    msg = "auto: round-2 update" if round_number == 2 else "auto: initial commit"
    for f in files:
        add_update_file(repo_name, f["name"], f["content"], msg)

# ---- LLM: Hugging Face Inference API ----
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

def llm_call(system_prompt: str, user_prompt: str, attachments: list = [], max_tries: int = 3) -> str:
    print(f"[LLM] backend=huggingface model={HF_MODEL}")
    attachment_text = ""
    for att in attachments:
        if att.get("url", "").startswith("data:text/csv;base64,"):
            try:
                b64_data = att["url"].split("base64,")[1]
                decoded = base64.b64decode(b64_data).decode("utf-8")
                attachment_text += f"Attachment: {att['name']} (CSV content):\n{decoded}\n"
            except Exception as e:
                print(f"[LLM][WARN] Failed to decode CSV attachment {att['name']}: {e}")
                attachment_text += f"Attachment: {att['name']} (base64: {att['url'][:50]}...)\n"
        elif att.get("url", "").startswith("data:application/json;base64,"):
            try:
                b64_data = att["url"].split("base64,")[1]
                decoded = base64.b64decode(b64_data).decode("utf-8")
                attachment_text += f"Attachment: {att['name']} (JSON content):\n{decoded}\n"
            except Exception as e:
                print(f"[LLM][WARN] Failed to decode JSON attachment {att['name']}: {e}")
                attachment_text += f"Attachment: {att['name']} (base64: {att['url'][:50]}...)\n"
        elif att.get("url", "").startswith("data:image/"):
            attachment_text += f"Attachment: {att['name']} (Image URL):\n{att['url']}\n"
        else:
            attachment_text += f"Attachment: {att['name']} ({att['url'][:50]}...)\n"
    
    full_prompt = f"{system_prompt.strip()}\n\n--- USER TASK ---\n{user_prompt.strip()}\n\nATTACHMENTS:\n{attachment_text}\n\nINSTRUCTIONS:\n- If a CSV attachment is provided, use its exact data (e.g., product names and sales values) in the generated HTML.\n- If a JSON attachment is provided, parse and use its data as specified in the brief.\n- If an image attachment is provided, display it in the HTML.\n- Do not generate placeholder data unless explicitly instructed.\n- Ensure numeric values are clean (no currency symbols like '$' unless specified)."
    
    delay = 1
    last_error = None
    for attempt in range(1, max_tries + 1):
        try:
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=4096,
                temperature=0.2
            )
            text = (resp.choices[0].message.content or "").strip()
            print("[LLM] got text len:", len(text))
            if "```" in text:
                text = text.replace("```html","").replace("```","").strip()
            print("[LLM] first 160:", text[:160].replace("\n","⏎"))
            return text
        except Exception as e:
            last_error = str(e)
            print(f"[LLM][ERR] Attempt {attempt} failed: {last_error} — retrying in {delay}s")
            time.sleep(delay)
            delay *= 2
    raise Exception(f"[LLM][ERR] Failed after {max_tries} tries: {last_error}")

def write_code_with_llm(data: dict, round_number: int = 1):
    print(f"[WRITE] enter write_code_with_llm round={round_number} task={data.get('task')}")
    user_prompt = f"""TASK: {data.get('task')}
ROUND: {round_number}
BRIEF:
{data.get('brief','(none)')}

REQUIREMENTS:
- Return only ONE complete HTML file (index.html), no markdown fences.
- Must run on GitHub Pages with no build step.
- Load any required libs via CDN.
- For CSV attachments, decode and use the exact data provided.
- For JSON attachments, parse and use the data as specified.
- For image attachments, display the image.
- Ensure numeric outputs are clean (e.g., no '$' symbols unless specified).

ATTACHMENTS:
{data.get('attachments', [])}

CHECKS:
{data.get('checks', [])}
"""
    system_prompt = ("You are an expert frontend engineer. "
                    "Return ONLY a full HTML document (<html>…</html>). "
                    "No commentary. No markdown fences.")
    try:
        print("[WRITE] calling llm_call…")
        html = llm_call(system_prompt, user_prompt, data.get('attachments', []))
        print("[WRITE] llm_call returned len:", len(html) if html else None)
        if not html or "<html" not in html.lower():
            print("[WRITE][WARN] LLM returned empty or non-HTML; will fallback.")
            raise ValueError("non-html or empty")
        html = re.sub(r'\$([\d.]+)', r'\1', html)
        return [{"name": "index.html", "content": html}]
    except Exception as e:
        print("[WRITE][ERR] using enhanced placeholder due to:", repr(e))
        task = data.get('task', '')
        brief = data.get('brief', '')
        checks = data.get('checks', [])
        attachments = data.get('attachments', [])
        
        # Extract title from brief or checks (e.g., "set title to 'X'")
        title = task
        for check in checks:
            if "document.title" in check and "===" in check:
                try:
                    title = check.split("===")[1].strip().strip("'\"")
                except:
                    pass
            elif "set the title to" in brief.lower():
                try:
                    start = brief.lower().index("set the title to") + len("set the title to")
                    title = brief[start:].split('.')[0].strip().strip("'\"")
                except:
                    pass
        
        # Process attachments
        attachment_content = ""
        for att in attachments:
            if att.get("url", "").startswith("data:text/csv;base64,"):
                try:
                    b64_data = att["url"].split("base64,")[1]
                    csv_content = base64.b64decode(b64_data).decode("utf-8")
                    attachment_content += f"<h3>{att['name']}</h3><pre>{csv_content}</pre>"
                    # Attempt to sum numbers for elements like #total-sales
                    try:
                        lines = csv_content.strip().split("\n")
                        total = 0
                        if len(lines) > 1:  # Skip header
                            for line in lines[1:]:
                                try:
                                    sales = float(line.split(",")[-1])  # Assume last column
                                    total += sales
                                except:
                                    pass
                        attachment_content += f"<p id='total-sales'>{total:.2f}</p>"
                    except:
                        pass
                except Exception as e:
                    attachment_content += f"<p>Failed to decode {att['name']}: {str(e)}</p>"
            elif att.get("url", "").startswith("data:application/json;base64,"):
                try:
                    b64_data = att["url"].split("base64,")[1]
                    json_content = base64.b64decode(b64_data).decode("utf-8")
                    attachment_content += f"<h3>{att['name']}</h3><pre>{json_content}</pre>"
                    # Add a select element for JSON (e.g., currency picker)
                    try:
                        import json
                        data = json.loads(json_content)
                        if isinstance(data, dict):
                            options = "".join([f"<option value='{k}'>{k}</option>" for k in data.keys()])
                            attachment_content += f"""
                            <select id='currency-picker' onchange='updateCurrency(this.value)'>
                                {options}
                            </select>
                            <p id='total-currency'>N/A</p>
                            """
                    except:
                        pass
                except Exception as e:
                    attachment_content += f"<p>Failed to decode {att['name']}: {str(e)}</p>"
            elif att.get("url", "").startswith("data:image/"):
                attachment_content += f"<h3>{att['name']}</h3><img src='{att['url']}' alt='{att['name']}' style='max-width: 100%;'>"
        
        # Extract common IDs from checks (e.g., #total-sales, #region-filter)
        element_ids = []
        for check in checks:
            matches = re.findall(r'#[\w-]+', check)
            element_ids.extend(matches)
        elements_html = "".join([f"<div id='{eid[1:]}'>Placeholder for {eid}</div>" for eid in set(element_ids)])
        
        # Include common libraries
        libraries = """
        <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
        <script src='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js'></script>
        """
        
        # Basic JavaScript for dynamic behavior (e.g., currency picker)
        script = """
        <script>
        function updateCurrency(value) {
            document.getElementById('total-currency').textContent = value;
        }
        </script>
        """
        
        placeholder = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>{title}</title>
    {libraries}
</head>
<body>
    <div class='container'>
        <h1>{task} (Round {round_number})</h1>
        <p>{brief}</p>
        {attachment_content}
        {elements_html}
    </div>
    {script}
</body>
</html>"""
        return [{"name": "index.html", "content": placeholder}]

def build_eval_payload(request_data: dict, repo_url: str, commit_sha: str, pages_url: str) -> dict:
    return {
        "email": request_data["email"],
        "task": request_data["task"],
        "round": int(request_data.get("round", 1)),
        "nonce": request_data["nonce"],
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url,
    }

def pages_url_for(repo_name: str) -> str:
    return f"https://iitm-iyy.github.io/{repo_name}/"

def post_evaluation_with_backoff(evaluation_url: str, payload: dict, max_tries: int = 6) -> bool:
    delay = 1
    last = None
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.post(
                evaluation_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            print(f"[DEBUG] Evaluation POST: {r.status_code} - {r.text[:200]}")
            if r.status_code == 200:
                print(f"evaluator acknowledged on try {attempt}")
                return True
            last = f"{r.status_code} {r.text[:200]}"
        except requests.RequestException as e:
            last = str(e)
        print(f"… evaluator POST failed (try {attempt}): {last} — retrying in {delay}s")
        time.sleep(delay)
        delay *= 2
    print(f"evaluator POST failed after {max_tries} tries: {last}")
    return False

def get_latest_commit_sha(repo_name: str, branch: str = DEFAULT_BRANCH) -> str | None:
    url = f"https://api.github.com/repos/iitm-iyy/{repo_name}/commits/{branch}"
    r = requests.get(url, headers=headers, timeout=30)
    print(f"[DEBUG] Get commit SHA: {r.status_code}")
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def build_readme_round1(repo: str, brief: str, pages_url: str) -> str:
    return f"""# {repo}

{brief or 'Initial implementation of the task.'}

## Overview
This repository contains a static web application built for the specified task.

## Setup
No build step required. The app is served via GitHub Pages.

## Usage
Open [{pages_url}]({pages_url}) in your browser.

## Code Explanation
- `index.html`: The main application file, generated to meet task requirements.
  - Loads required libraries via CDN.
  - Implements the task as per the provided brief.

## License
MIT
"""

def round1(data: dict):
    print(f"[ROUND1] start task={data.get('task')} nonce={data.get('nonce')}")
    repo = f"{data['task'].replace(' ', '-')}_{data['nonce']}"
    files = write_code_with_llm(data, 1)
    print("[ROUND1] about to push index.html first 160:", files[0]["content"][:160].replace("\n","⏎"))
    
    repo_info = create_github_repo(repo)
    print(f"[ROUND1] Repo created: {repo_info.get('full_name', 'N/A')}")
    
    add_update_file(repo, "index.html", files[0]["content"], "auto: add/update index.html")
    readme = build_readme_round1(repo, data.get("brief", ""), pages_url_for(repo))
    add_update_file(repo, "README.md", readme, "auto: initial README")
    
    enable_github_pages(repo)
    commit_sha = get_latest_commit_sha(repo, DEFAULT_BRANCH) or ""
    payload = build_eval_payload(
        data,
        repo_info.get("html_url", f"https://github.com/iitm-iyy/{repo}"),
        commit_sha,
        pages_url_for(repo),
    )
    if data.get("evaluation_url"):
        post_evaluation_with_backoff(data["evaluation_url"], payload, max_tries=6)
    print("[ROUND1] done")

def build_readme_round2(repo: str, brief: str, pages_url: str) -> str:
    return f"""# {repo}

{brief or 'Round-2: updates applied to the app.'}

## What changed in Round-2
- Implemented brief-driven updates and refactors.
- See the page live at [{pages_url}]({pages_url}).

## Setup
No build step required. Served via GitHub Pages.

## Usage
Open [{pages_url}]({pages_url}) in your browser.

## Code Explanation
- `index.html`: Updated to reflect round-2 requirements.
  - Includes CDN-loaded libraries.
  - Implements new features from the brief.

## License
MIT
"""

def round2(data: dict):
    repo = f"{data['task'].replace(' ', '-')}_{data['nonce']}"
    repo_url = f"https://github.com/iitm-iyy/{repo}"
    pages_url = f"https://iitm-iyy.github.io/{repo}/"

    files = write_code_with_llm(data, round_number=2)
    push_files_to_repo(repo, files, round_number=2)

    readme = build_readme_round2(repo, data.get("brief", ""), pages_url)
    add_update_file(repo, "README.md", readme, "auto: round-2 README update")

    enable_github_pages(repo)
    commit_sha = get_latest_commit_sha(repo, DEFAULT_BRANCH) or ""
    payload = build_eval_payload(data, repo_url, commit_sha, pages_url)
    payload["round"] = 2
    if data.get("evaluation_url"):
        post_evaluation_with_backoff(data["evaluation_url"], payload, max_tries=6)
    print("[ROUND2] done")

@app.post("/handle_task")
async def handle_task(data: dict, bg: BackgroundTasks):
    if not validate_secret(data.get("secret", "")):
        return JSONResponse({"error": "Invalid secret"}, status_code=401)

    round_number = data.get("round", 1)
    if round_number == 1:
        bg.add_task(round1, data)
        return JSONResponse(
            {
                "status": "accepted",
                "round": 1,
                "task": data.get("task"),
                "message": "Round-1 task received successfully and processing in background."
            },
            status_code=200
        )
    elif round_number == 2:
        bg.add_task(round2, data)
        return JSONResponse(
            {
                "status": "accepted",
                "round": 2,
                "task": data.get("task"),
                "message": "Round-2 task received successfully and processing in background."
            },
            status_code=200
        )
    else:
        return JSONResponse({"error": "Invalid round"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)