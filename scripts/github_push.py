"""
Script: github_push.py
Purpose: Commit and push updates to GitHub
Usage: python scripts/github_push.py "Your commit message"
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}")
        return False, result.stderr.strip()
    return True, result.stdout.strip()


def push_to_github(commit_message: str):
    username = os.getenv("GITHUB_USERNAME")
    token    = os.getenv("GITHUB_TOKEN")
    repo     = os.getenv("GITHUB_REPO_NAME", "Credit-Card-Customer-Segmentation")
    email    = os.getenv("GITHUB_EMAIL", f"{username}@users.noreply.github.com")

    if not username or not token:
        print("ERROR: GITHUB_USERNAME and GITHUB_TOKEN must be set in .env")
        sys.exit(1)

    remote_url = f"https://{username}:{token}@github.com/{username}/{repo}.git"
    run(f'git config user.name "{username}"')
    run(f'git config user.email "{email}"')
    run(f'git remote set-url origin {remote_url}')

    run("git add .")
    ok, out = run(f'git commit -m "{commit_message}"')
    if not ok:
        if "nothing to commit" in out.lower():
            print("Nothing new to commit.")
            return
        sys.exit(1)

    ok, out = run("git push -u origin main")
    if ok:
        print(f"Pushed to: https://github.com/{username}/{repo}")
    else:
        print(f"Push failed: {out}")


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "Update project files"
    push_to_github(msg)
