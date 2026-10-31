# The Primer v2: Hikâyene Küçük Bir Model Ekle

> **Ders:** YZM301 Yazılım Testi — Ödev 7 · **Tohum (Seed):** `13` · **Hata Toleransı:** `< 1e-4` ($10^{-4}$)
> - **Canlı Uygulama (GitHub Pages):** <https://seymengur.github.io/the-primer-v2/>
> - **GitHub Reposu:** <https://github.com/Seymengur/the-primer-v2>
> - **Hugging Face Model Reposu:** <https://huggingface.co/Seymen55/the-primer-v2-onnx>
> - **Commit-Sabitli ONNX Model URL'si:** `https://huggingface.co/Seymen55/the-primer-v2-onnx/resolve/846390dc7ad50af1e942307346bfd0e8e93824a2/model.onnx`

**The Primer v2**, kullanıcının başlangıçtaki 3 soruluk profiliyle başlayan etkileşimli yazılım mühendisliği hikâyesini, oyuncunun sahne içindeki gerçek zamanlı davranışından (karar süresi, seçenek değiştirme/tereddüt sayısı, birikimli kritik hata örüntüsü ve teknik borç riski) öğrenen **küçük, göreve özel (dikey) bir tabular ONNX modeline** (`StandardScaler → LogisticRegression`) bağlar.

---

## 1. Hızlı Kurulum ve Çalıştırma (En Fazla 3 Komut)

```bash
pip install -r requirements.txt
python train.py && python tests/parite.py
python -m http.server 8080 --directory web
```

*(Bonus +10 Sunucu Tüketicisini başlatmak için: `python server/app.py --port 3100`)*

---

## 2. Mimari ve Uçtan Uca Boru Hattı (Development · Deployment · Inference)

```text
data/oturumlar.csv (18 gerçek oturum [90 satır] + 32 sentetik oturum [160 satır], SEED=13)
       │
       ▼  python train.py (Eval Kapısı: AUC >= 0.80, Accuracy >= 0.78, ONNX fark <= 1e-4)
model/{model.onnx, config.json, README.md, model.joblib}
       │
       ├──► 1. Tarayıcı İstemcisi (web/index.html — onnxruntime-web)
       │       • P(mudahale=1) >= 0.55 ise Sokratik Mentor + Ödünleşim Analizi + Tutor Koruması açılır
       │       • Shadow Deployment karşılaştırması (v1 sabit kural vs. v2 ONNX) canlı izlenir
       │
       ├──► 2. Python Test Kahini / Oracle (tests/parite.py → tests/parite.json)
       │       • 5 referans girdide |p_web - p_oracle| < 1e-4 diferansiyel parite kanıtı (Cypress)
       │
       └──► 3. Sunucu API Tüketicisi (+10 Bonus: server/app.py — POST /api/v1/predictions)
               • Aynı 5 referans girdide < 1e-4 toleransla birebir aynı olasılığı üretir
```

---

## 3. Veri Seti ve Sentetik Veri Üretici Kuralı (`SEED = 13`)

`data/oturumlar.csv` toplam **250 karar satırından** (50 oturum × 5 sahne) oluşur:
- **Gerçek Oturumlar (`kaynak = gercek`):** 4 farklı yazılım mühendisliği öğrencisinin Veri Toplama Modu ile oynadığı **18 gerçek oturum (90 karar satırı)**. KVKK gereği hiçbir kişisel veri (ad, e-posta, IP, cihaz kimliği) toplanmamış; rastgele anonim `oturum_id` (`ses_g01`..`ses_g18`) kullanılmıştır.
- **Sentetik Oturumlar (`kaynak = sentetik`):** `generate_data.py` içinde sabit tohum **`SEED = 13`** (`random.Random(13)`) kullanılarak üretilen **32 sentetik oturum (160 karar satırı)**.
  - **Üretici Kural:** Her sahnede oyuncunun bilişsel yük ve tereddüt skoru $z$ aşağıdaki doğrusal birleşim ve Gauss gözlem gürültüsü ile hesaplanır:
    $$z = -0.85 \cdot \text{seviye} + 0.42 \cdot \text{zorluk} + 0.09 \cdot (\text{sure} - 16.0) + 0.48 \cdot \text{degisim} + 0.65 \cdot \text{hata} + 0.35 \cdot \text{ipucu} + 2.20 \cdot \text{risk} - 1.35 + \mathcal{N}(0, 0.45)$$
    $$\sigma(z) = \frac{1}{1 + e^{-z}} \ge 0.50 \implies \text{mudahale\_gerekli} = 1, \quad \text{aksi halde } 0$$

---

## 4. Beklenen Repo Yapısı ve Dosya Haritası

| Dosya / Klasör | Açıklama |
| :--- | :--- |
| [`spec.md`](spec.md) | Model gereksinimi, eğitimden **önce** tanımlı Eval Kapısı eşikleri ve 3 kabul örneği |
| [`data/oturumlar.csv`](data/oturumlar.csv) | 18 gerçek + 32 sentetik oturum (250 karar satırı, `kaynak` sütunu zorunlu) |
| [`data/SOZLUK.md`](data/SOZLUK.md) | 11 sütunun veri tipi, birimi, geçerli aralığı ve KVKK gizlilik ilkeleri |
| [`train.py`](train.py) | Sabit tohumlu (`SEED=13`) eğitim, Eval Kapısı kontrolü ve `sklearn → ONNX` paketleyici |
| [`model/model.onnx`](model/model.onnx) | Taşınabilir ONNX modeli (`opset=17`, `zipmap=False`) |
| [`model/config.json`](model/config.json) | Özellik sırası, sınıf etiketleri, kabul eşikleri ve test metriklerini içeren sözleşme |
| [`model/README.md`](model/README.md) | Model Kartı (Mitchell vd., 2019: amaç, veri, metrikler, sınırlar, kapsam dışı durumlar) |
| [`RELEASE.md`](RELEASE.md) | `v2.0` sürüm notları, commit sabitleme, Rollback, Shadow/Canary ve Mimari Ödünleşim Matrisi |
| [`tests/parite.py`](tests/parite.py) | İkinci tüketici (Python Test Kahini / Oracle) doğrulama betiği |
| [`tests/parite.json`](tests/parite.json) | 5 referans girdi vektörü ve Python Oracle olasılık çıktıları |
| [`web/index.html`](web/index.html) | `onnxruntime-web` ile tarayıcıda çalışan etkileşimli Primer v2 + Parite Lab + Veri Toplama Modu |
| [`cypress/e2e/primer_v2.cy.js`](cypress/e2e/primer_v2.cy.js) | Diferansiyel parite testi (`< 1e-4`) ve v1→v2 regresyon E2E test süiti |
| [`SINAV_TASARIMI.md`](SINAV_TASARIMI.md) | 1 sayfalık *"Sizinle Uyumlanan Klasik Vize Sınav Sorusu"* düşünce egzersizi |
| [`server/app.py`](server/app.py) | **Bonus (+10):** Kendi sunucumuzda çalışan ve `< 1e-4` parite sağlayan ONNX API servisi |

---

## 5. Kaynaklar ve Akademik Referanslar

- **Amershi, S. vd. (2019).** *Software Engineering for Machine Learning: A Case Study.* IEEE/ACM ICSE-SEIP.
- **Shankar, S. vd. (2022).** *Operationalizing Machine Learning: An Interview Study* (3V: Velocity, Validation, Versioning).
- **Mitchell, M. vd. (2019).** *Model Cards for Model Reporting.* ACM FAT*.
- **Breck, E. vd. (2017).** *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.*
- **Sommerville, I. (2015).** *Software Engineering* (10. Baskı) — Bl. 8 Yazılım Testi (V&V, Regresyon Testi), Bl. 25 Konfigürasyon Yönetimi.
- **Stanford HAI (2025) & Gartner (2025).** *AI Index Report & Small Task-Specific Models Forecast.*
- **Başlangıç Reposu:** [`seyyah/huggingface-hello-web`](https://github.com/seyyah/huggingface-hello-web)

---

## 6. Coding Agent Mühendislik Beyanı (Şeffaflık Bildirimi)

Bu projenin geliştirilmesinde **Google Antigravity** kodlama ajanı mühendislik gözetiminde kullanılmış, aşağıdaki tasarım kararları ve doğrulamalar bizzat kontrol edilip düzenlenmiştir:

1. **Özellik Mühendisliği ve Sözleşme Tasarımı (`spec.md` & `data/SOZLUK.md`):** Başlangıç reposundaki 5 özelliklik taban örnek doğrudan kopyalanmamış; oyuncunun kararsızlığını ölçen `secenek_degistirme_sayisi` ve birikimli teknik borcu ölçen `mimari_risk_skoru` eklenerek 7 özellikli davranışsal şema tanımlanmıştır. Meta veri sütunları (`oturum_id`, `sahne_no`, `kaynak`) veri sızıntısını (*data leakage*) önlemek için eğitim matrisinden çıkarılmıştır.
2. **ONNX `zipmap=False` ve `float32` Parite Düzeltmesi (`train.py` & `web/index.html`):** `skl2onnx` varsayılan olarak çıktı olasılıklarını sözlük dizisi (`Sequence<Map>`) olarak ürettiğinde `onnxruntime-web` içinde tensör okuma hatası oluştuğu tespit edilmiş; `options={id(model): {"zipmap": False}}` ve `Float32Array` dönüşümü uygulanarak Python Oracle ile tarayıcı arasındaki farkın `< 1e-6` (tolerans `< 1e-4`) düzeyine inmesi sağlanmıştır.
3. **Eval Kapısı (Gate) Doğrulaması (`train.py`):** Modelin her durumda koşulsuz paketlenmesini önlemek için eğitim öncesinde `spec.md` içinde tanımlanan `AUC >= 0.80` ve `Accuracy >= 0.78` kontrolleri `train.py` içine zorunlu kapı olarak eklenmiş ve yüksek eşik (`--min-auc 0.999`) verilerek betiğin `sys.exit(1)` ile durduğu bizzat test edilmiştir.
4. **Oyun İçi Davranışsal Etki ve Gölge Dağıtım (`web/index.html`):** Model çıktısının yalnızca ekranda bir sayı olarak kalmaması için `p >= 0.55` eşiğinde Sokratik Mentor kartının açılması, anlatım derinliğinin (*Mimari Ödünleşim Analizi*) genişlemesi ve yanlış seçimlerdeki risk cezasının `%40` yumuşatılması mekanizmaları eklenmiştir.
