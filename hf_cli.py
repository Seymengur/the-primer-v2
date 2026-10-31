"""The Primer v2 — Hugging Face Hub & GitHub Pages Dağıtım CLI'ı (hf_cli.py).

Başlangıç reposundaki (`huggingface-hello-web`) `hf_cli.py` sözleşmesini The Primer v2'nin
`model/` klasörü ve commit'e sabitlenmiş (`resolve/<commit>/model.onnx`) sürümleme yapısıyla
birebir uyumlu şekilde çalıştırır.

Komutlar:
    python hf_cli.py train          → train.py + tests/parite.py çalıştırır
    python hf_cli.py pull-cloud     → Canlı web uygulamasından buluta otomatik kaydedilen arkadaş oturumlarını data/oturumlar.csv'ye ekler
    python hf_cli.py predict        → yerel model ile örnek çıkarım yapar
    python hf_cli.py push-model     → model/ klasörünü Hugging Face Hub'a yükler ve commit SHA döner
    python hf_cli.py hub-predict    → HF Hub'daki commit'li model.onnx'i indirip çıkarım yapar
    python hf_cli.py deploy         → train + push-model + web/site.json güncellemesi
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from generate_data import veri_uret
from tests.parite import parite_hesapla_ve_dogrula
from train import MODEL_DIR, VERI_YOLU, egit_ve_paketle

BURASI = Path(__file__).resolve().parent
VARSAYILAN_AD = "the-primer-v2-onnx"
BULUT_KANALI = "the_primer_v2_seymen_oturumlar_live"


def cmd_pull_cloud(args):
    url = f"https://ntfy.sh/{BULUT_KANALI}/json?poll=1&since=all"
    print(f"☁️ Buluttaki canlı oturum verileri çekiliyor: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "the-primer-v2-cli/2.0"})
    yeni_satirlar = []
    with urllib.request.urlopen(req, timeout=15) as resp:
        for line in resp.read().decode("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                zarf = json.loads(line)
                if zarf.get("event") != "message":
                    continue
                icerik = json.loads(zarf.get("message", "{}"))
                for s in icerik.get("rows", []):
                    if "oturum_id" in s and "sahne_no" in s:
                        s["kaynak"] = "gercek"
                        yeni_satirlar.append(s)
            except Exception:
                continue

    if not VERI_YOLU.exists():
        veri_uret()

    mevcut_df = pd.read_csv(VERI_YOLU)
    mevcut_anahtarlar = set(zip(mevcut_df["oturum_id"].astype(str), mevcut_df["sahne_no"].astype(int)))

    eklenecek = []
    for s in yeni_satirlar:
        anahtar = (str(s["oturum_id"]), int(s["sahne_no"]))
        if anahtar not in mevcut_anahtarlar:
            mevcut_anahtarlar.add(anahtar)
            eklenecek.append(s)

    if eklenecek:
        ek_df = pd.DataFrame(eklenecek)[mevcut_df.columns]
        birlesik_df = pd.concat([mevcut_df, ek_df], ignore_index=True)
        birlesik_df.to_csv(VERI_YOLU, index=False)
        print(f"✓ Buluttan {len(eklenecek)} yeni gerçek sahne satırı data/oturumlar.csv dosyasına eklendi!")
        print(f"  Toplam satır: {len(birlesik_df)} (Gerçek: {int((birlesik_df['kaynak'] == 'gercek').sum())})")
    else:
        print("ℹ Bulutta henüz data/oturumlar.csv içinde olmayan yeni bir oturum bulunamadı.")

    if getattr(args, "retrain", False) and eklenecek:
        cmd_train(args)


def cmd_train(args):
    if not VERI_YOLU.exists():
        veri_uret()
    egit_ve_paketle()
    parite_hesapla_ve_dogrula()


def cmd_predict(args):
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    oturum = ort.InferenceSession(str(MODEL_DIR / "model.onnx"))
    girdi = json.loads(args.json) if args.json else {o["name"]: o["example"] for o in config["features"]}
    vektor = np.array([[float(girdi[o["name"]]) for o in config["features"]]], dtype=np.float32)
    olasilik = oturum.run(["probabilities"], {"input": vektor})[0][0]
    p_mudahale = float(olasilik[1])
    esik = float(config.get("decision_threshold", 0.55))
    print(
        json.dumps(
            {
                "input": girdi,
                "probability_intervention": round(p_mudahale, 6),
                "decision": "mudahale" if p_mudahale >= esik else "otonom",
                "proba": {
                    config["class_labels"][str(c)]: round(float(p), 4)
                    for c, p in zip(config["classes"], olasilik)
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def api_ve_kullanici(token=None):
    from huggingface_hub import HfApi, login

    if token:
        login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    try:
        kullanici = api.whoami()["name"]
    except Exception:
        sys.exit(
            "✗ HF girişi yok. Önce: .\\.venv\\Scripts\\hf.exe auth login --token hf_... "
            "(https://huggingface.co/settings/tokens adresinden Write yetkili token alın)"
        )
    return api, kullanici


def cmd_whoami(args):
    _, kullanici = api_ve_kullanici(getattr(args, "token", None))
    print(f"✓ Hugging Face girişi doğrulandı: {kullanici}")


def repo_adi(verilen, kullanici):
    if verilen:
        temiz = verilen.strip("<>")
        return temiz if "/" in temiz else f"{kullanici}/{temiz}"
    return f"{kullanici}/{VARSAYILAN_AD}"


def cmd_push_model(args):
    if not (MODEL_DIR / "model.onnx").exists():
        sys.exit("✗ Model bulunamadı. Önce: python train.py")
    api, kullanici = api_ve_kullanici(getattr(args, "token", None))
    repo = repo_adi(args.repo, kullanici)
    api.create_repo(repo, repo_type="model", private=args.private, exist_ok=True)
    commit_info = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=MODEL_DIR,
        commit_message="v2.0: The Primer v2 Sokratik Mentor ONNX modeli",
    )
    commit_sha = getattr(commit_info, "oid", None) or api.repo_info(repo, repo_type="model").sha
    pinned_url = f"https://huggingface.co/{repo}/resolve/{commit_sha}/model.onnx"

    site_json_yolu = BURASI / "web" / "site.json"
    site_verisi = {
        "model_repo": repo,
        "commit": commit_sha,
        "pinned_model_url": pinned_url,
        "pinned_config_url": f"https://huggingface.co/{repo}/resolve/{commit_sha}/config.json",
        "local_fallback": True,
    }
    site_json_yolu.write_text(json.dumps(site_verisi, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    release_yolu = BURASI / "RELEASE.md"
    if release_yolu.exists():
        metin = release_yolu.read_text(encoding="utf-8")
        metin = metin.replace("<kullanici>/the-primer-v2-onnx", repo).replace("<commit_sha>", commit_sha)
        release_yolu.write_text(metin, encoding="utf-8")

    print(f"✓ Model reposu yayınlandı → https://huggingface.co/{repo}")
    print(f"✓ Sabitlenmiş (commit-pinned) ONNX URL:\n  {pinned_url}")
    return repo, commit_sha


def cmd_hub_predict(args):
    from huggingface_hub import hf_hub_download

    _, kullanici = api_ve_kullanici(getattr(args, "token", None))
    repo = repo_adi(args.repo, kullanici)
    revision = args.revision or "main"
    config = json.loads(
        Path(hf_hub_download(repo, "config.json", revision=revision, force_download=True)).read_text(encoding="utf-8")
    )
    oturum = ort.InferenceSession(hf_hub_download(repo, "model.onnx", revision=revision, force_download=True))
    girdi = json.loads(args.json) if args.json else {o["name"]: o["example"] for o in config["features"]}
    x = np.array([[float(girdi[o["name"]]) for o in config["features"]]], dtype=np.float32)
    olasilik = oturum.run(["probabilities"], {"input": x})[0][0]
    print(
        json.dumps(
            {
                "repo": repo,
                "revision": revision,
                "input": girdi,
                "proba": {
                    config["class_labels"][c]: round(float(p), 4)
                    for c, p in zip(config["classes"], olasilik)
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_deploy(args):
    cmd_train(args)
    cmd_push_model(args)


def main():
    p = argparse.ArgumentParser(description="The Primer v2 — Model Eğitim ve HF Hub CLI")
    alt = p.add_subparsers(dest="komut", required=True)

    alt.add_parser("train", help="Veriyi doğrula, modeli eğit ve parite.json üret").set_defaults(fn=cmd_train)

    pc = alt.add_parser("pull-cloud", help="Canlı web uygulamasından buluta gelen arkadaş verilerini data/oturumlar.csv'ye çek")
    pc.add_argument("--retrain", action="store_true", help="Yeni veri eklendiyse modeli otomatik yeniden eğit")
    pc.set_defaults(fn=cmd_pull_cloud)

    w = alt.add_parser("whoami", help="Hugging Face girişini doğrula")
    w.add_argument("--token", help="Hugging Face Write token (hf_...)")
    w.set_defaults(fn=cmd_whoami)

    pr = alt.add_parser("predict", help="Yerel ONNX modeliyle tahmin yap")
    pr.add_argument("--json", help="Girdi JSON dizgesi (boşsa config.json örneği kullanılır)")
    pr.set_defaults(fn=cmd_predict)

    for ad, fn, yardim in [
        ("push-model", cmd_push_model, "model/ klasörünü Hugging Face Hub'a yükle"),
        ("deploy", cmd_deploy, "train + push-model"),
    ]:
        s = alt.add_parser(ad, help=yardim)
        s.add_argument("--repo", help=f"HF model reposu (varsayılan: <kullanıcı>/{VARSAYILAN_AD})")
        s.add_argument("--token", help="Hugging Face Write token (hf_...)")
        s.add_argument("--private", action="store_true", help="Özel repo olarak oluştur")
        s.set_defaults(fn=fn)

    hp = alt.add_parser("hub-predict", help="Hub'daki model.onnx ile çıkarım yap")
    hp.add_argument("--repo", help=f"HF model reposu (varsayılan: <kullanıcı>/{VARSAYILAN_AD})")
    hp.add_argument("--revision", help="Commit SHA veya dal adı (varsayılan: main)")
    hp.add_argument("--token", help="Hugging Face token (hf_...)")
    hp.add_argument("--json")
    hp.set_defaults(fn=cmd_hub_predict)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
