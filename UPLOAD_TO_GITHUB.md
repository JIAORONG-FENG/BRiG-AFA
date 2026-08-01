# Upload instructions

This directory is the public release package. Do not upload the parent research workspace.

## GitHub web interface

1. Sign in to GitHub as `JIAORONG-FENG`.
2. Create a new repository named `BRiG-AFA`.
3. Choose **Public** or **Private** as appropriate.
4. Do not initialize the repository with another README, `.gitignore`, or license.
5. Open **uploading an existing file** and upload the contents of this directory.
6. Use the initial commit message `Initial public release of BRiG-AFA`.

Before making the repository public, select a software license with all authors. No license has been assumed in this package.

## Command line, once Git is available

```bash
git init
git branch -M main
git add .
git commit -m "Initial public release of BRiG-AFA"
git remote add origin https://github.com/JIAORONG-FENG/BRiG-AFA.git
git push -u origin main
```
