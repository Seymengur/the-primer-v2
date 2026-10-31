"""The Primer v2 — GitHub Repo Oluşturma, Push ve GitHub Pages Etkinleştirme Betiği (deploy_github.py)."""
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BURASI = Path(__file__).resolve().parent
REPO_ADI = "the-primer-v2"


def git_credential_al():
    """Windows Git Credential Manager üzerinden mevcut GitHub oturumunu alır."""
    girdi = "protocol=https\nhost=github.com\n\n"
    r = subprocess.run(
        ["git", "credential", "fill"],
        input=girdi,
        capture_output=True,
        text=True,
        cwd=BURASI,
    )
    if r.returncode != 0:
        sys.exit(f"✗ Git kimlik bilgisi alınamadı:\n{r.stderr}")
    bilgiler = {}
    for satir in r.stdout.splitlines():
        if "=" in satir:
            k, v = satir.split("=", 1)
            bilgiler[k.strip()] = v.strip()
    if "password" not in bilgiler:
        sys.exit("✗ GitHub parolası/token bulunamadı.")
    return bilgiler.get("username", ""), bilgiler["password"]


def gh_api(metod: str, yol: str, token: str, veri: dict = None):
    url = f"https://api.github.com{yol}"
    govde = json.dumps(veri).encode("utf-8") if veri is not None else None
    req = urllib.request.Request(url, data=govde, method=metod)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "the-primer-v2-deployer")
    if govde is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            ham = resp.read().decode("utf-8")
            return resp.status, (json.loads(ham) if ham else {})
    except urllib.error.HTTPError as e:
        ham = e.read().decode("utf-8", errors="replace")
        try:
             return e.code, json.loads(ham)
        except Exception:
            return e.code, {"message": ham}


def main():
    # Kök dizinde de parite.json ve site.json bulunsun (hem / hem /web uyumu için)
    shutil.copy2(BURASI / "web" / "site.json", BURASI / "site.json")
    shutil.copy2(BURASI / "web" / "parite.json", BURASI / "parite.json")
    shutil.copy2(BURASI / "web" / "index.html", BURASI / "index.html")

    _, token = git_credential_al()
    kod, kullanici_bilgi = gh_api("GET", "/user", token)
    if kod != 200:
        sys.exit(f"✗ GitHub API kimlik doğrulaması başarısız ({kod}): {kullanici_bilgi}")

    kullanici = kullanici_bilgi["login"]
    tam_repo = f"{kullanici}/{REPO_ADI}"
    repo_url = f"https://github.com/{tam_repo}"
    pages_url = f"https://{kullanici.lower()}.github.io/{REPO_ADI}/"

    print(f"✓ GitHub kullanıcısı doğrulandı: {kullanici}")

    # 1. Repo var mı kontrol et, yoksa oluştur
    kod, mevcut_repo = gh_api("GET", f"/repos/{tam_repo}", token)
    if kod == 404:
        kod, yeni_repo = gh_api(
            "POST",
            "/user/repos",
            token,
            {
                "name": REPO_ADI,
                "description": "The Primer v2: Uyarlamalı Yazılım Mühendisliği Hikâyesi, ONNX Çıkarımı ve Diferansiyel Parite Testi",
                "private": False,
                "has_issues": True,
            },
        )
        if kod not in (200, 201):
            sys.exit(f"✗ GitHub reposu oluşturulamadı ({kod}): {yeni_repo}")
        print(f"✓ Public GitHub reposu oluşturuldu → {repo_url}")
    else:
        print(f"✓ Mevcut GitHub reposu bulundu → {repo_url}")

    # 2. README.md içine canlı uygulama ve GitHub repo linkini ekle, commit'le ve push et
    subprocess.run(["git", "branch", "-M", "main"], cwd=BURASI, check=True)
    subprocess.run(["git", "add", "site.json", "parite.json", "index.html"], cwd=BURASI, check=True)
    subprocess.run(["git", "commit", "-m", "chore: sync root static files for GitHub Pages"], cwd=BURASI, check=False)

    r_rem = subprocess.run(["git", "remote"], cwd=BURASI, capture_output=True, text=True)
    if "origin" in r_rem.stdout.split():
        subprocess.run(["git", "remote", "set-url", "origin", f"{repo_url}.git"], cwd=BURASI, check=True)
    else:
        subprocess.run(["git", "remote", "add", "origin", f"{repo_url}.git"], cwd=BURASI, check=True)

    subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd=BURASI, check=True)
    print(f"✓ Kodlar GitHub'a push edildi → {repo_url}")

    # 3. GitHub Pages'i etkinleştir (önce workflow, izin yoksa main branch root)
    p_kod, _ = gh_api("GET", f"/repos/{tam_repo}/pages", token)
    if p_kod == 404:
        k1, yanit1 = gh_api("POST", f"/repos/{tam_repo}/pages", token, {"build_type": "workflow"})
        if k1 in (200, 201):
            print("✓ GitHub Pages etkinleştirildi (Kaynak: GitHub Actions workflow)")
        else:
            k2, yanit2 = gh_api(
                "POST",
                f"/repos/{tam_repo}/pages",
                token,
                {"source": {"branch": "main", "path": "/"}},
            )
            if k2 in (200, 201):
                print("✓ GitHub Pages etkinleştirildi (Kaynak: main branch /)")
            else:
                print(f"⚠ GitHub Pages API uyarısı ({k1}/{k2}): {yanit1} / {yanit2}")
    else:
        print("✓ GitHub Pages zaten aktif.")

    # Workflow'u tetiklemek için küçük bir boş commit veya workflow_dispatch
    gh_api("POST", f"/repos/{tam_repo}/actions/workflows/pages.yml/dispatches", token, {"ref": "main"})

    print(f"\n==========================================================")
    print(f"1. GitHub Repo Linki     : {repo_url}")
    print(f"2. Canlı Uygulama Linki  : {pages_url}")
    print(f"==========================================================")


if __name__ == "__main__":
    main()
