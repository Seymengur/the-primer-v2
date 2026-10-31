"""The Primer v2 — Tekrarlanabilir Model Eğitimi, Eval Kapısı ve ONNX Paketleyici (train.py).

Kullanım:
    python train.py
    python train.py --min-auc 0.999   # Eval kapısının (gate) reddetme davranışını test etmek için
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
import sklearn
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BURASI = Path(__file__).resolve().parent
VERI_YOLU = BURASI / "data" / "oturumlar.csv"
MODEL_DIR = BURASI / "model"
WEB_MODEL_DIR = BURASI / "web" / "model"

SEED = 13
HEDEF = "mudahale_gerekli"
KARAR_ESIGI = 0.55

# spec.md'de eğitim ÖNCESİ tanımlanan altın standart kabul eşikleri (Eval Gate)
VARSAYILAN_MIN_AUC = 0.8000
VARSAYILAN_MIN_ACCURACY = 0.7800
MAKS_ONNX_FARK = 1e-4

OZELLIK_TANIMLARI = [
    {
        "name": "deneyim_seviyesi",
        "label": "Deneyim Seviyesi (1:Çırak, 2:Kalfa, 3:Usta)",
        "unit": "seviye",
        "min": 1,
        "max": 3,
        "example": 1.0,
    },
    {
        "name": "sahne_zorlugu",
        "label": "Sahne Zorluğu (1-5)",
        "unit": "katsayı",
        "min": 1,
        "max": 5,
        "example": 4.0,
    },
    {
        "name": "karar_suresi_sn",
        "label": "Karar Süresi (sn)",
        "unit": "saniye",
        "min": 1.0,
        "max": 60.0,
        "example": 34.5,
    },
    {
        "name": "secenek_degistirme_sayisi",
        "label": "Seçenek Değiştirme / Tereddüt Sayısı",
        "unit": "adet",
        "min": 0,
        "max": 10,
        "example": 5.0,
    },
    {
        "name": "onceki_kritik_hata_sayisi",
        "label": "Önceki Kritik Hata Sayısı",
        "unit": "adet",
        "min": 0,
        "max": 5,
        "example": 2.0,
    },
    {
        "name": "acilan_ipucu_sayisi",
        "label": "Açılan İpucu Sayısı",
        "unit": "adet",
        "min": 0,
        "max": 5,
        "example": 2.0,
    },
    {
        "name": "mimari_risk_skoru",
        "label": "Mimari Risk / Teknik Borç Skoru (0-1)",
        "unit": "oran",
        "min": 0.0,
        "max": 1.0,
        "example": 0.72,
    },
]
OZELLIK_ADLARI = [o["name"] for o in OZELLIK_TANIMLARI]


def veri_dogrula(df: pd.DataFrame):
    gerekli_sutunlar = set(OZELLIK_ADLARI + [HEDEF, "kaynak", "oturum_id"])
    eksik = gerekli_sutunlar - set(df.columns)
    if eksik:
        sys.exit(f"✗ Veri sözleşmesi hatası! Eksik sütunlar: {sorted(eksik)}")
    if not set(df["kaynak"].unique()).issubset({"gercek", "sentetik"}):
        sys.exit("✗ 'kaynak' sütunu yalnızca 'gercek' veya 'sentetik' değerlerini alabilir.")
    gercek_oturum_sayisi = df[df["kaynak"] == "gercek"]["oturum_id"].nunique()
    if gercek_oturum_sayisi < 15:
        sys.exit(f"✗ En az 15 gerçek oturum gereklidir (mevcut: {gercek_oturum_sayisi}).")
    return gercek_oturum_sayisi


def onnx_aktar_ve_dogrula(model, X_ornek: pd.DataFrame, hedef_yol: Path) -> float:
    """sklearn Pipeline → ONNX dönüşümü ve 1e-4 eşdeğerlik (parite) kontrolü."""
    onx = to_onnx(
        model,
        initial_types=[("input", FloatTensorType([None, X_ornek.shape[1]]))],
        options={id(model): {"zipmap": False}},
        target_opset=17,
    )
    hedef_yol.write_bytes(onx.SerializeToString())

    oturum = ort.InferenceSession(str(hedef_yol))
    p_onnx = oturum.run(["probabilities"], {"input": X_ornek.to_numpy(np.float32)})[0]
    p_sklearn = model.predict_proba(X_ornek)
    maks_fark = float(np.abs(p_onnx - p_sklearn).max())
    if maks_fark > MAKS_ONNX_FARK:
        hedef_yol.unlink(missing_ok=True)
        sys.exit(f"✗ ONNX çıktısı sklearn'den sapıyor (maks fark {maks_fark:.2e} > {MAKS_ONNX_FARK:.1e})")
    return maks_fark


def model_karti_uret(config: dict) -> str:
    m = config["metrics"]
    esikler = config["eval_gate"]
    agirlik_satirlari = "\n".join(
        f"| `{ad}` | {deger:+.4f} |" for ad, deger in config["feature_coefficients"].items()
    )
    ornek_girdi = json.dumps({o["name"]: o["example"] for o in config["features"]}, ensure_ascii=False)
    return f"""---
library_name: sklearn
tags:
- onnx
- tabular-classification
- adaptive-learning
- the-primer-v2
- software-engineering-education
pipeline_tag: tabular-classification
license: mit
---

# {config["title"]}

{config["description"]}

## 1. Model Özeti ve Mimari (Mitchell vd., 2019 — Model Card)

- **Mimari:** `sklearn.pipeline.Pipeline(StandardScaler → LogisticRegression(C=1.0, random_state={config["seed"]}))`
- **Format:** Taşınabilir `model.onnx` (`opset=17`, `zipmap=False`) + `model.joblib`
- **Tohum (Seed):** `{config["seed"]}` (Deterministik ve tekrarlanabilir eğitim)
- **Eğitim Ortamı:** Python `{config["python_version"]}`, scikit-learn `{config["sklearn_version"]}`
- **Karar Eşiği (Decision Threshold):** `{config["decision_threshold"]}` (`P(mudahale_gerekli=1) >= 0.55` ise Sokratik Mentor ve Uyarlamalı Destek açılır)

## 2. Amaçlanan Kullanım (Intended Use)

- **Birincil Amaç:** **The Primer v2** etkileşimli yazılım mühendisliği hikâyesinde oyuncunun sahne içi davranışsal telemetrisini (karar süresi, seçenekler arası tereddüt/değiştirme sayısı, önceki kritik hatalar, açılan ipuçları ve birikimli teknik borç riski) gerçek zamanlı analiz ederek oyuncuya **Sokratik Mentor müdahalesi ve uyarlamalı anlatım derinliği** sunulup sunulmayacağına karar vermek.
- **Hedef Tüketiciler:**
  1. Tarayıcı İstemcisi (`onnxruntime-web` — GitHub Pages / AI Studio)
  2. Python Test Kahini / Oracle (`onnxruntime` — `tests/parite.py`)
  3. Mobil Uygulama (`onnxruntime-react-native` — Expo) ve Sunucu API (`server/app.py` / `rails-api`)

## 3. Eğitim Verisi ve KVKK / Gizlilik

- **Veri Kaynağı (`data/oturumlar.csv`):** Toplam `{config["dataset_summary"]["total_rows"]}` karar satırı (`{config["dataset_summary"]["real_sessions"]}` gerçek oyuncu oturumu = `{config["dataset_summary"]["real_rows"]}` gerçek satır; `{config["dataset_summary"]["synthetic_sessions"]}` sentetik oturum = `{config["dataset_summary"]["synthetic_rows"]}` sentetik satır, `SEED={config["seed"]}`).
- **Gizlilik (KVKK):** Hiçbir kişisel veri (ad, e-posta, IP, cihaz kimliği) toplanmamıştır; oturum kimlikleri rastgele anonim dizgelerdir.

## 4. Değerlendirme Metrikleri ve Kabul Kapısı (Eval Gate — %30 Hold-out Test Kümesi)

| Metrik | Test Değeri | Kabul Eşiği (`spec.md`) | Durum |
| :--- | :--- | :--- | :--- |
| **ROC-AUC (`auc`)** | **`{m["auc"]}`** | `>= {esikler["min_auc"]}` | ✅ Geçti |
| **Accuracy (`accuracy`)** | **`{m["accuracy"]}`** | `>= {esikler["min_accuracy"]}` | ✅ Geçti |
| **F1 Skoru (`f1`)** | `{m["f1"]}` | Bilgi amaçlı | ✅ Ölçüldü |
| **Precision (`precision`)** | `{m["precision"]}` | Bilgi amaçlı | ✅ Ölçüldü |
| **Recall (`recall`)** | `{m["recall"]}` | Bilgi amaçlı | ✅ Ölçüldü |
| **Gerçek Veri Alt Kümesi Accuracy** | `{m["real_subset_accuracy"]}` | Bilgi amaçlı | ✅ Ölçüldü |
| **ONNX ↔ sklearn Maks Fark** | **`{m["onnx_max_diff"]:.2e}`** | `<= {esikler["max_onnx_diff"]:.1e}` | ✅ Geçti |

### Öğrenilen Standartlaştırılmış Lojistik Regresyon Katsayıları
| Özellik (`feature`) | Katsayı (`weight`) |
| :--- | :--- |
{agirlik_satirlari}

## 5. Sınırlar, Yanlılıklar ve Kullanılmaması Gereken Durumlar (Out-of-Scope)

1. **Resmi Sınav / Notlandırma Yasağı:** Bu model yalnızca pedagojik hikâye uyarlaması (ipucu ve anlatım derinliği) içindir; öğrencilerin resmi ders notunu, işe alım kararını veya akademik başarısını değerlendirmek için **kesinlikle kullanılamaz**.
2. **Alan Dışı (Out-of-Distribution) Girdiler:** Karar süresinin `60` saniyeyi çok aştığı (kullanıcının ekran başından ayrıldığı / AFK kaldığı) durumlarda model bunu "yüksek bilişsel tereddüt" olarak yorumlayabilir. İstemci tarafında `karar_suresi_sn` değeri `[1.0, 60.0]` aralığına kırpılmalıdır (`clamping`).
3. **Küçük Veri Kümesi Sınırı:** Gerçek oturumlar 4 farklı yazılım mühendisliği öğrencisinin 18 oturumundan toplanmıştır; farklı yaş veya meslek gruplarında yeniden kalibrasyon gerektirir.

## 6. Örnek Çıkarım (Python & ONNX)

```python
import numpy as np, onnxruntime as ort

session = ort.InferenceSession("model/model.onnx")
x = np.array([[1.0, 4.0, 34.5, 5.0, 2.0, 2.0, 0.72]], dtype=np.float32)
proba = session.run(["probabilities"], {{"input": x}})[0][0]
print("Müdahale olasılığı P(1):", round(float(proba[1]), 4))
```
Örnek JSON girdisi: `{ornek_girdi}`
"""


def egit_ve_paketle(min_auc: float = VARSAYILAN_MIN_AUC, min_accuracy: float = VARSAYILAN_MIN_ACCURACY):
    if not VERI_YOLU.exists():
        sys.exit(f"✗ Veri dosyası bulunamadı: {VERI_YOLU}")

    df = pd.read_csv(VERI_YOLU)
    gercek_oturum_sayisi = veri_dogrula(df)
    sentetik_oturum_sayisi = df[df["kaynak"] == "sentetik"]["oturum_id"].nunique()

    X = df[OZELLIK_ADLARI].astype(np.float32)
    y = df[HEDEF].astype(int)

    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, df, test_size=0.30, stratify=y, random_state=SEED
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, random_state=SEED)),
    ])
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    gercek_maske = df_te["kaynak"] == "gercek"
    gercek_acc = round(float(accuracy_score(y_te[gercek_maske], y_pred[gercek_maske])), 4) if gercek_maske.any() else 0.0

    metrikler = {
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "auc": round(float(roc_auc_score(y_te, y_proba)), 4),
        "f1": round(float(f1_score(y_te, y_pred)), 4),
        "precision": round(float(precision_score(y_te, y_pred)), 4),
        "recall": round(float(recall_score(y_te, y_pred)), 4),
        "real_subset_accuracy": gercek_acc,
    }

    # --- EVAL KAPISI (EVAL GATE) KONTROLÜ ---
    if metrikler["auc"] < min_auc:
        sys.exit(
            f"✗ EVAL KAPISI REDDEDİLDİ: Test AUC ({metrikler['auc']}) < Kabul Eşiği ({min_auc}). "
            "Model artefaktı yayınlanmadı!"
        )
    if metrikler["accuracy"] < min_accuracy:
        sys.exit(
            f"✗ EVAL KAPISI REDDEDİLDİ: Test Accuracy ({metrikler['accuracy']}) < Kabul Eşiği ({min_accuracy}). "
            "Model artefaktı yayınlanmadı!"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx_yolu = MODEL_DIR / "model.onnx"
    maks_fark = onnx_aktar_ve_dogrula(model, X_te, onnx_yolu)
    metrikler["onnx_max_diff"] = round(maks_fark, 8)

    joblib.dump(model, MODEL_DIR / "model.joblib")

    lr_step = model.named_steps["lr"]
    katsayilar = {ad: round(float(w), 4) for ad, w in zip(OZELLIK_ADLARI, lr_step.coef_[0])}

    config = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "The Primer v2 — Sokratik Mentor Uyarlama Modeli",
        "version": "2.0.0",
        "description": (
            "Oyuncunun sahne içi karar süresi, seçenek değiştirme (tereddüt) sayısı, hata örüntüsü "
            "ve teknik borç riskine göre Sokratik Mentor müdahalesi gerekip gerekmediğini tahmin eden "
            "göreve özel (dikey) tabular ONNX modeli."
        ),
        "target": HEDEF,
        "decision_threshold": KARAR_ESIGI,
        "class_labels": {
            "0": "Otonom Akış (Müdahale Gerekmez)",
            "1": "Sokratik Mentor Müdahalesi (Uyarlamalı Destek)",
        },
        "classes": [str(c) for c in model.classes_],
        "features": OZELLIK_TANIMLARI,
        "metrics": metrikler,
        "eval_gate": {
            "min_auc": min_auc,
            "min_accuracy": min_accuracy,
            "max_onnx_diff": MAKS_ONNX_FARK,
            "passed": True,
        },
        "dataset_summary": {
            "total_rows": int(len(df)),
            "real_rows": int((df["kaynak"] == "gercek").sum()),
            "synthetic_rows": int((df["kaynak"] == "sentetik").sum()),
            "real_sessions": int(gercek_oturum_sayisi),
            "synthetic_sessions": int(sentetik_oturum_sayisi),
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
        },
        "feature_coefficients": katsayilar,
        "intercept": round(float(lr_step.intercept_[0]), 4),
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }

    (MODEL_DIR / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (MODEL_DIR / "README.md").write_text(model_karti_uret(config), encoding="utf-8")

    # Web istemcisinin yerel çalışmada da aynı artefaktı kullanması için kopyala
    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_DIR / "model.onnx", WEB_MODEL_DIR / "model.onnx")
    shutil.copy2(MODEL_DIR / "config.json", WEB_MODEL_DIR / "config.json")

    print(f"✓ Eval kapısı geçildi ve model paketlendi → {MODEL_DIR}")
    print(f"  Test metrikleri: AUC={metrikler['auc']}, Accuracy={metrikler['accuracy']}, F1={metrikler['f1']}")
    print(f"  ONNX ↔ sklearn maks fark: {maks_fark:.2e} (Eşik: <= {MAKS_ONNX_FARK:.1e})")
    return config


def main():
    parser = argparse.ArgumentParser(description="The Primer v2 — Model Eğitimi ve Eval Kapısı")
    parser.add_argument("--min-auc", type=float, default=VARSAYILAN_MIN_AUC, help="Minimum kabul AUC eşiği")
    parser.add_argument(
        "--min-accuracy", type=float, default=VARSAYILAN_MIN_ACCURACY, help="Minimum kabul Accuracy eşiği"
    )
    args = parser.parse_args()
    egit_ve_paketle(min_auc=args.min_auc, min_accuracy=args.min_accuracy)


if __name__ == "__main__":
    main()
