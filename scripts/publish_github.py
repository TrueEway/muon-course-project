"""Publish the prepared project using an existing Git Credential Manager login.

Never prints, writes, or accepts a token as a command-line argument. Repo
creation is needed because the connected GitHub tool has no create-repo tool.
Requires --publish for any remote mutation. Normal execution only checks auth.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = 'TrueEway'
REPO = 'muon-course-project'


def credentials():
    env = dict(os.environ, GCM_INTERACTIVE='Never', GIT_TERMINAL_PROMPT='0')
    result = subprocess.run(['git', 'credential', 'fill'],
                            input=f'protocol=https\nhost=github.com\nusername={OWNER}\n\n',
                            text=True, capture_output=True, env=env, check=False)
    if result.returncode:
        raise RuntimeError('No noninteractive GitHub credential is available; sign in with Git Credential Manager.')
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not fields.get('password'):
        raise RuntimeError('Git Credential Manager returned no usable credential.')
    return fields['password']


def request(token, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request('https://api.github.com' + path, data=data,
        headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                 'User-Agent': 'muon-course-project', 'X-GitHub-Api-Version': '2022-11-28',
                 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=40) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # The response body contains API diagnostics, never request headers.
        if error.code == 404:
            return None
        message = json.loads(error.read()).get('message', 'GitHub API request failed')
        raise RuntimeError(f'GitHub API {error.code}: {message}') from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--tag', default='v1.1', help='Existing local tag to publish; older tags are never replaced')
    args = parser.parse_args()
    token = credentials()
    user = request(token, '/user')
    if user.get('login', '').lower() != OWNER.lower():
        raise RuntimeError('The local GitHub identity differs from the connected account.')
    repo = request(token, f'/repos/{OWNER}/{REPO}')
    print(json.dumps({'authenticated_account': user['login'], 'repository_exists': repo is not None}))
    if not args.publish:
        return
    if not (ROOT / 'output' / 'pdf' / 'main.pdf').exists() or not (ROOT / 'output' / 'audit.json').exists():
        raise RuntimeError('Finish and verify the report and experiments before publishing.')
    if args.tag != 'v1.0':
        import hashlib
        verification = json.loads((ROOT/'output'/'verification.json').read_text(encoding='utf-8'))
        if verification.get('vision_formal_runs') != 21 or verification.get('revision_formal_runs') != 30:
            raise RuntimeError('Complete and verify all revised experiments before publishing.')
        if verification['pdf_sha256'] != hashlib.sha256((ROOT/'output'/'pdf'/'Muon_Report.pdf').read_bytes()).hexdigest():
            raise RuntimeError('The report changed after verification.')
        visual_review = json.loads((ROOT/'output'/'visual_review.json').read_text(encoding='utf-8'))
        if visual_review.get('result') != 'passed' or visual_review.get('pdf_sha256') != verification['pdf_sha256']:
            raise RuntimeError('Render and visually review the final PDF before publishing.')
    local_tag = subprocess.check_output(['git','rev-parse',args.tag+'^{commit}'],cwd=ROOT,text=True).strip()
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if local_tag != head:
        raise RuntimeError('The selected tag does not point at the prepared HEAD commit.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit the prepared deliverables before publishing.')
    if repo is None:
        repo = request(token, '/user/repos', dict(name=REPO, private=False,
            description='Muon optimizer: convergence proofs, reproducible CPU matrix experiments and a tiny Transformer study.',
            auto_init=False))
    if repo['private']:
        raise RuntimeError('Existing repository is private; visibility will not be changed automatically.')
    remote = f'https://github.com/{OWNER}/{REPO}.git'
    current = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, capture_output=True, text=True)
    if current.returncode:
        subprocess.run(['git', 'remote', 'add', 'origin', remote], cwd=ROOT, check=True)
    elif current.stdout.strip() != remote:
        raise RuntimeError('Existing origin differs from the intended repository.')
    subprocess.run(['git', 'push', '-u', 'origin', 'main'], cwd=ROOT, check=True)
    subprocess.run(['git', 'push', 'origin', args.tag], cwd=ROOT, check=True)
    print(json.dumps({'published': repo['html_url'], 'visibility': 'public', 'tag': args.tag}))


if __name__ == '__main__':
    main()
