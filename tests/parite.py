"""The Primer v2 — İkinci Tüketici (Test Kahini / Oracle) ve Diferansiyel Parite Betiği (tests/parite.py).

Aynı `model/model.onnx` artefaktını Python `onnxruntime` ile çalıştırır, 5 referans girdi için
referans olasılıkları (`tests/parite.json`) üretir ve doğrular. Web istemcisindeki (`onnxruntime-web`)
ve sunucudaki (`server/app.py`) çıkarımlar bu Oracle çıktılarıyla < 1e-4 hata toleransı dahilinde
eşleşmek zorundadır.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

BURASI = Path(__file__).resolve().parent.parent
MODEL_DIR = BURASI / "model"
PARITE_JSON = BURASI / "tests" / "parite.json"
WEB_PARITE_JSON = BURASI / "web" / "parite.json"
TOLERANS = 1e-4

REFERANS_SENARYOLAR = [
    {
        "id": "ornek_1_cirak_tereddutlu",
        "aciklama": "Kabul Örneği 1: Yüksek tereddüt ve birikimli hata yaşayan çırak (Sokratik müdahale beklenir)",
        "input": {
            "deneyim_seviyesi": 1.0,
            "sahne_zorlugu": 4.0,
            "karar_suresi_sn": 34.5,
            "secenek_degistirme_sayisi": 5.0,
            "onceki_kritik_hata_sayisi": 2.0,
            "acilan_ipucu_sayisi": 2.0,
            "mimari_risk_skoru": 0.72,
        },
    },
    {
        "id": "ornek_2_usta_otonom",
        "aciklama": "Kabul Örneği 2: Hızlı ve hatasız ilerleyen usta mühendis adayı (Otonom akış beklenir)",
        "input": {
            "deneyim_seviyesi": 3.0,
            "sahne_zorlugu": 2.0,
            "karar_suresi_sn": 6.2,
            "secenek_degistirme_sayisi": 0.0,
            "onceki_kritik_hata_sayisi": 0.0,
            "acilan_ipucu_sayisi": 0.0,
            "mimari_risk_skoru": 0.10,
        },
    },
    {
        "id": "ornek_3_kalfa_sinirda",
        "aciklama": "Kabul Örneği 3: Orta zorlukta kararsızlık yaşayan kalfa profili",
        "input": {
            "deneyim_seviyesi": 2.0,
            "sahne_zorlugu": 3.0,
            "karar_suresi_sn": 21.0,
            "secenek_degistirme_sayisi": 3.0,
            "onceki_kritik_hata_sayisi": 1.0,
            "acilan_ipucu_sayisi": 1.0,
            "mimari_risk_skoru": 0.48,
        },
    },
    {
        "id": "ornek_4_aceleci_yuksek_risk",
        "aciklama": "Hızlı tıklayan ancak kritik hatalarla yüksek teknik borç biriktirmiş oyuncu",
        "input": {
            "deneyim_seviyesi": 1.0,
            "sahne_zorlugu": 5.0,
            "karar_suresi_sn": 11.5,
            "secenek_degistirme_sayisi": 2.0,
            "onceki_kritik_hata_sayisi": 3.0,
            "acilan_ipucu_sayisi": 1.0,
            "mimari_risk_skoru": 0.85,
        },
    },
    {
        "id": "ornek_5_dengeli_kalfa_akista",
        "aciklama": "Zor sahnede kontrollü düşünen ancak hata yapmamış dengeli kalfa",
        "input": {
            "deneyim_seviyesi": 2.0,
            "sahne_zorlugu": 4.0,
            "karar_suresi_sn": 14.0,
            "secenek_degistirme_sayisi": 1.0,
            "onceki_kritik_hata_sayisi": 0.0,
            "acilan_ipucu_sayisi": 0.0,
            "mimari_risk_skoru": 0.15,
        },
    },
]


def parite_hesapla_ve_dogrula(sadece_kontrol: bool = False):
    onnx_yolu = MODEL_DIR / "model.onnx"
    config_yolu = MODEL_DIR / "config.json"
    joblib_yolu = MODEL_DIR / "model.joblib"

    if not onnx_yolu.exists() or not config_yolu.exists():
        sys.exit("✗ Model dosyaları bulunamadı. Önce `python train.py` çalıştırın.")

    config = json.loads(config_yolu.read_text(encoding="utf-8"))
    ozellik_sirasi = [f["name"] for f in config["features"]]
    esik = float(config.get("decision_threshold", 0.55))

    oturum = ort.InferenceSession(str(onnx_yolu))
    sklearn_model = joblib.load(joblib_yolu) if joblib_yolu.exists() else None

    kayitlar = []
    maks_ic_fark = 0.0

    for senaryo in REFERANS_SENARYOLAR:
        girdi = senaryo["input"]
        vektor = np.array([[float(girdi[ad]) for ad in ozellik_sirasi]], dtype=np.float32)
        onnx_cikti = oturum.run(["probabilities"], {"input": vektor})[0][0]
        p_mudahale = float(onnx_cikti[1])
        p_otonom = float(onnx_cikti[0])

        if sklearn_model is not None:
            df_girdi = pd.DataFrame([{ad: float(girdi[ad]) for ad in ozellik_sirasi}])
            sk_p = float(sklearn_model.predict_proba(df_girdi)[0][1])
            fark = abs(p_mudahale - sk_p)
            maks_ic_fark = max(maks_ic_fark, fark)
            if fark >= TOLERANS:
                sys.exit(f"✗ {senaryo['id']}: ONNX ({p_mudahale}) ile sklearn ({sk_p}) farkı {fark:.2e} >= {TOLERANS}")

        kayitlar.append({
            "id": senaryo["id"],
            "description": senaryo["aciklama"],
            "input": girdi,
            "feature_vector": [float(girdi[ad]) for ad in ozellik_sirasi],
            "oracle_probability": round(p_mudahale, 6),
            "oracle_probability_4dec": round(p_mudahale, 4),
            "oracle_Complement_probability": round(p_otonom, 6),
            "expected_decision": "mudahale" if p_mudahale >= esik else "otonom",
        })

    cikti_verisi = {
        "contract_version": config.get("version", "2.0.0"),
        "tolerance": TOLERANS,
        "decision_threshold": esik,
        "feature_order": ozellik_sirasi,
        "cases": kayitlar,
    }

    if sadece_kontrol and PARITE_JSON.exists():
        mevcut = json.loads(PARITE_JSON.read_text(encoding="utf-8"))
        for yeni_k, eski_k in zip(kayitlar, mevcut["cases"]):
            fark = abs(yeni_k["oracle_probability"] - eski_k["oracle_probability"])
            if fark >= TOLERANS:
                sys.exit(
                    f"✗ Parite uyumsuzluğu ({yeni_k['id']}): Oracle={yeni_k['oracle_probability']} "
                    f"vs Kayıtlı={eski_k['oracle_probability']} (fark={fark:.2e})"
                )
        print(f"✓ Parite doğrulaması başarılı (5/5 referans girdi, maks fark < {TOLERANS:.1e})")
        return cikti_verisi

    PARITE_JSON.parent.mkdir(parents=True, exist_ok=True)
    PARITE_JSON.write_text(json.dumps(cikti_verisi, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    WEB_PARITE_JSON.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PARITE_JSON, WEB_PARITE_JSON)

    print(f"✓ {PARITE_JSON} üretildi (5 referans senaryo, ONNX↔sklearn maks fark={maks_ic_fark:.2e} < {TOLERANS:.1e})")
    for k in kayitlar:
        print(f"  - {k['id']}: P(mudahale=1) = {k['oracle_probability']:.6f} → {k['expected_decision']}")
    return cikti_verisi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Primer v2 — Python Test Kahini (Oracle) Parite Betiği")
    parser.add_argument("--check", action="store_true", help="Mevcut tests/parite.json dosyasını doğrula")
    args = parser.parse_args()
    parite_hesapla_ve_dogrula(sadece_kontrol=args.check)
