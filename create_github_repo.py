#!/usr/bin/env python3
"""
Script to create a private GitHub repository using the GitHub API.
Requires a GitHub Personal Access Token with 'repo' scope.
"""

import os
import sys
import subprocess
import getpass

def create_github_repo():
    try:
        import requests
    except ImportError:
        print("Error: requests library not found")
        print("Install with: pip install requests")
        sys.exit(1)

    print("GitHub Repository Creator")
    print("=" * 50)
    print()

    # Get GitHub credentials
    username = input("Enter your GitHub username: ").strip()
    print()
    print("You need a Personal Access Token with 'repo' scope.")
    print("Create one at: https://github.com/settings/tokens/new")
    print()
    token = getpass.getpass("Enter your GitHub Personal Access Token: ").strip()

    if not username or not token:
        print("Error: Username and token are required")
        sys.exit(1)

    # Create repository
    print()
    print("Creating private repository 'deepseek-ocr'...")

    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "name": "deepseek-ocr",
        "description": "Batch OCR processing tool using DeepSeek-OCR model via vLLM, optimized for Nvidia DGX Spark",
        "private": True,
        "auto_init": False
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 201:
        repo_data = response.json()
        print(f"✓ Repository created successfully!")
        print(f"  URL: {repo_data['html_url']}")
        print(f"  Clone URL (HTTPS): {repo_data['clone_url']}")
        print(f"  Clone URL (SSH): {repo_data['ssh_url']}")
        print()

        # Add remote and push
        ssh_url = repo_data['ssh_url']
        https_url = repo_data['clone_url']

        print("Choose remote URL type:")
        print("1. SSH (recommended if you have SSH keys configured)")
        print("2. HTTPS")
        choice = input("Enter choice (1 or 2): ").strip()

        remote_url = ssh_url if choice == "1" else https_url

        print()
        print("Adding remote and pushing...")

        # Add remote
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)

        # Rename branch to main
        subprocess.run(["git", "branch", "-M", "main"], check=True)

        # Push to GitHub
        result = subprocess.run(["git", "push", "-u", "origin", "main"],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Successfully pushed to GitHub!")
            print(f"  Repository: {repo_data['html_url']}")
        else:
            print("Error pushing to GitHub:")
            print(result.stderr)
            print()
            print("You can manually push with:")
            print(f"  git push -u origin main")

    elif response.status_code == 422:
        error_msg = response.json().get('errors', [{}])[0].get('message', '')
        if 'already exists' in error_msg.lower():
            print(f"✗ Repository 'deepseek-ocr' already exists for user '{username}'")
            print()
            print("To push to existing repository:")
            print(f"  git remote add origin git@github.com:{username}/deepseek-ocr.git")
            print("  git branch -M main")
            print("  git push -u origin main")
        else:
            print(f"✗ Error: {error_msg}")
    else:
        print(f"✗ Error creating repository (HTTP {response.status_code})")
        print(f"  {response.json().get('message', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    create_github_repo()
