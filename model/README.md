---
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

# The Primer v2 — Sokratik Mentor Uyarlama Modeli

Oyuncunun sahne içi karar süresi, seçenek değiştirme (tereddüt) sayısı, hata örüntüsü ve teknik borç riskine göre Sokratik Mentor müdahalesi gerekip gerekmediğini tahmin eden göreve özel (dikey) tabular ONNX modeli.

## 1. Model Özeti ve Mimari (Mitchell vd., 2019 — Model Card)

- **Mimari:** `sklearn.pipeline.Pipeline(StandardScaler → LogisticRegression(C=1.0, random_state=13))`
- **Format:** Taşınabilir `model.onnx` (`opset=17`, `zipmap=False`) + `model.joblib`
- **Tohum (Seed):** `13` (Deterministik ve tekrarlanabilir eğitim)
- **Eğitim Ortamı:** Python `3.12`, scikit-learn `1.9.1`
- **Karar Eşiği (Decision Threshold):** `0.55` (`P(mudahale_gerekli=1) >= 0.55` ise Sokratik Mentor ve Uyarlamalı Destek açılır)

## 2. Amaçlanan Kullanım (Intended Use)

- **Birincil Amaç:** **The Primer v2** etkileşimli yazılım mühendisliği hikâyesinde oyuncunun sahne içi davranışsal telemetrisini (karar süresi, seçenekler arası tereddüt/değiştirme sayısı, önceki kritik hatalar, açılan ipuçları ve birikimli teknik borç riski) gerçek zamanlı analiz ederek oyuncuya **Sokratik Mentor müdahalesi ve uyarlamalı anlatım derinliği** sunulup sunulmayacağına karar vermek.
- **Hedef Tüketiciler:**
  1. Tarayıcı İstemcisi (`onnxruntime-web` — GitHub Pages / AI Studio)
  2. Python Test Kahini / Oracle (`onnxruntime` — `tests/parite.py`)
  3. Mobil Uygulama (`onnxruntime-react-native` — Expo) ve Sunucu API (`server/app.py` / `rails-api`)

## 3. Eğitim Verisi ve KVKK / Gizlilik

- **Veri Kaynağı (`data/oturumlar.csv`):** Toplam `250` karar satırı (`18` gerçek oyuncu oturumu = `90` gerçek satır; `32` sentetik oturum = `160` sentetik satır, `SEED=13`).
- **Gizlilik (KVKK):** Hiçbir kişisel veri (ad, e-posta, IP, cihaz kimliği) toplanmamıştır; oturum kimlikleri rastgele anonim dizgelerdir.

## 4. Değerlendirme Metrikleri ve Kabul Kapısı (Eval Gate — %30 Hold-out Test Kümesi)

| Metrik | Test Değeri | Kabul Eşiği (`spec.md`) | Durum |
| :--- | :--- | :--- | :--- |
| **ROC-AUC (`auc`)** | **`0.9686`** | `>= 0.8` | ✅ Geçti |
| **Accuracy (`accuracy`)** | **`0.88`** | `>= 0.78` | ✅ Geçti |
| **F1 Skoru (`f1`)** | `0.8861` | Bilgi amaçlı | ✅ Ölçüldü |
| **Precision (`precision`)** | `0.8974` | Bilgi amaçlı | ✅ Ölçüldü |
| **Recall (`recall`)** | `0.875` | Bilgi amaçlı | ✅ Ölçüldü |
| **Gerçek Veri Alt Kümesi Accuracy** | `0.8333` | Bilgi amaçlı | ✅ Ölçüldü |
| **ONNX ↔ sklearn Maks Fark** | **`1.20e-07`** | `<= 1.0e-04` | ✅ Geçti |

### Öğrenilen Standartlaştırılmış Lojistik Regresyon Katsayıları
| Özellik (`feature`) | Katsayı (`weight`) |
| :--- | :--- |
| `deneyim_seviyesi` | -1.2432 |
| `sahne_zorlugu` | +0.7834 |
| `karar_suresi_sn` | +2.4668 |
| `secenek_degistirme_sayisi` | +1.0147 |
| `onceki_kritik_hata_sayisi` | +1.1269 |
| `acilan_ipucu_sayisi` | +0.8799 |
| `mimari_risk_skoru` | -0.2127 |

## 5. Sınırlar, Yanlılıklar ve Kullanılmaması Gereken Durumlar (Out-of-Scope)

1. **Resmi Sınav / Notlandırma Yasağı:** Bu model yalnızca pedagojik hikâye uyarlaması (ipucu ve anlatım derinliği) içindir; öğrencilerin resmi ders notunu, işe alım kararını veya akademik başarısını değerlendirmek için **kesinlikle kullanılamaz**.
2. **Alan Dışı (Out-of-Distribution) Girdiler:** Karar süresinin `60` saniyeyi çok aştığı (kullanıcının ekran başından ayrıldığı / AFK kaldığı) durumlarda model bunu "yüksek bilişsel tereddüt" olarak yorumlayabilir. İstemci tarafında `karar_suresi_sn` değeri `[1.0, 60.0]` aralığına kırpılmalıdır (`clamping`).
3. **Küçük Veri Kümesi Sınırı:** Gerçek oturumlar 4 farklı yazılım mühendisliği öğrencisinin 18 oturumundan toplanmıştır; farklı yaş veya meslek gruplarında yeniden kalibrasyon gerektirir.

## 6. Örnek Çıkarım (Python & ONNX)

```python
import numpy as np, onnxruntime as ort

session = ort.InferenceSession("model/model.onnx")
x = np.array([[1.0, 4.0, 34.5, 5.0, 2.0, 2.0, 0.72]], dtype=np.float32)
proba = session.run(["probabilities"], {"input": x})[0][0]
print("Müdahale olasılığı P(1):", round(float(proba[1]), 4))
```
Örnek JSON girdisi: `{"deneyim_seviyesi": 1.0, "sahne_zorlugu": 4.0, "karar_suresi_sn": 34.5, "secenek_degistirme_sayisi": 5.0, "onceki_kritik_hata_sayisi": 2.0, "acilan_ipucu_sayisi": 2.0, "mimari_risk_skoru": 0.72}`
