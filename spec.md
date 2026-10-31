# The Primer v2 — Yazılım ve Model Spesifikasyonu (`spec.md`)

> **Sürüm:** `v2.0` · **Ders:** YZM301 Yazılım Testi · **Tohum (Seed):** `13`
> **Altın Kural:** Kabul eşikleri (Eval Gate) model eğitilmeden **ÖNCE** bu belgede tanımlanmıştır. Bu eşikleri geçemeyen hiçbir model artefaktı (`model.onnx`) paketlenmez ve yayınlanmaz.

---

## 1. Vizyon ve Problem Tanımı

**The Primer v1**, kullanıcının girişte yanıtladığı 3 profil sorusuna göre sabit kurallarla (`if/else`) uyarlanan etkileşimli bir yazılım mühendisliği hikâyesiydi. Ancak sabit kurallar, oyuncunun sahne içindeki **gerçek zamanlı bilişsel yükünü** (tereddüt süresi, seçenekler arasında kararsız kalma, birikimli kritik hata örüntüsü ve teknik borç riski) yakalayamaz.

**The Primer v2**, *"Yazılım mühendisi olmak isteyen bir insanla maksimum uyumlanan yapay zekâ destekli bir Primer nasıl olurdu?"* sorusuna yanıt olarak sabit kuralları **göreve özel (dikey) küçük bir makine öğrenmesi modeli** (`StandardScaler → LogisticRegression`, `sklearn → ONNX`) ile değiştirir.

### Modelin Görevi
Oyuncunun her sahnedeki karar davranışını temsil eden **7 tabular özelliği** girdi olarak alır ve oyuncunun o sahnede **Sokratik Mentor Müdahalesine ve Uyarlamalı Destek Moduna (`mudahale_gerekli = 1`)** ihtiyacı olup olmadığını olasılıksal olarak (`P(mudahale_gerekli = 1)`) tahmin eder.

### Model Çıktısının Oyun Davranışına Somut Etkisi
Modelin ürettiği olasılık `p = P(mudahale_gerekli = 1)` karar eşiği olan **`0.55`** ile karşılaştırılır:

1. **Uyarlamalı Sokratik Destek Modu (`p >= 0.55`):**
   - **Sokratik Mentor Kartı Açılır:** Doğrudan doğru şıkkı söylemek yerine sahnedeki mühendislik ikileminin (ör. *Diferansiyel Parite*, *Shadow Deployment*, *KVKK Anonimleştirme*, *Commit Sabitleme*) temel ilkesini hatırlatan Sokratik bir ipucu ve akademik referans (Sommerville / Amershi / Shankar) gösterilir.
   - **Anlatım Derinliği Artırılır (Adaptive Scaffolding):** Sahneye *"Mimari Ödünleşim (Trade-off) Analizi"* kutusu eklenir ve teknik borç tuzağı içeren seçenek üzerinde görsel risk rozeti (`⚠️ Yüksek Teknik Borç Riski`) belirir.
   - **Baskı Dengelenir:** Oyuncunun o sahnede yanlış karar vermesi durumunda birikimli `mimari_risk_skoru` artışı `%40` oranında yumuşatılır (tutor koruması).
2. **Otonom Mühendis / Akış Modu (`p < 0.55`):**
   - Oyuncu akışta (flow) kabul edilir; ek ipucu veya risk rozeti gösterilmez, hikâye tam zorlukta ve standart anlatım derinliğinde ilerler.

---

## 2. Girdi / Çıktı Sözleşmesi (Contract)

Model (`model/model.onnx`) ve sözleşme dosyası (`model/config.json`) tüm tüketiciler (Web `onnxruntime-web`, Python Test Kahini `onnxruntime`, Mobil `Expo` ve Sunucu API) için tek doğruluk kaynağıdır:

- **Girdi Tensörü (`input`):** `float32 [n, 7]` — Özellik sırası `config.json` içindeki `features` dizisiyle birebir aynıdır:
  1. `deneyim_seviyesi` (`1`: Çırak/Junior, `2`: Kalfa/Mid, `3`: Usta/Senior Adayı)
  2. `sahne_zorlugu` (`1` – `5` arası sahne teknik karmaşıklık derecesi)
  3. `karar_suresi_sn` (`1.0` – `60.0` sn arası sahnede karar verme süresi)
  4. `secenek_degistirme_sayisi` (`0` – `10` arası onay öncesi seçenekler arası tereddüt/geçiş sayısı)
  5. `onceki_kritik_hata_sayisi` (`0` – `5` arası önceki sahnelerde yapılan kritik mimari/test hatası)
  6. `acilan_ipucu_sayisi` (`0` – `5` arası önceki sahnelerde açılan Sokratik ipucu sayısı)
  7. `mimari_risk_skoru` (`0.0` – `1.0` arası birikimli teknik borç / kötü sona gitme riski)
- **Çıktı Tensörü (`probabilities`):** `float32 [n, 2]` — `[P(sınıf="0"), P(sınıf="1")]` (`zipmap=False`).
  - `"0"`: `"Otonom Akış (Müdahale Gerekmez)"`
  - `"1"`: `"Sokratik Mentor Müdahalesi (Uyarlamalı Destek)"`

---

## 3. Eğitim Öncesi Tanımlı Kabul Eşikleri (Eval Gate / Altın Standart)

`train.py` betiği `%30` ayrılmış test kümesi (`stratify=y, random_state=13`) üzerinde aşağıdaki **üç zorunlu kabul kapısını** kontrol eder. Herhangi biri sağlanmazsa eğitim betiği `sys.exit(1)` ile durur ve artefakt üretmez:

| Kapı (Gate) | Metrik | Kabul Eşiği (Altın Standart) | Gerekçe |
| :--- | :--- | :--- | :--- |
| **Eval Kapısı 1** | `ROC-AUC` (`auc`) | **`>= 0.8000`** | Sokratik müdahale gereken oyuncularla otonom akıştaki oyuncuları olasılıksal sıralamada yüksek ayrıştırma gücü. |
| **Eval Kapısı 2** | `Accuracy` (`accuracy`) | **`>= 0.7800`** | Karar eşiğinde doğru sınıflandırma başarımı. |
| **Parite Kapısı 3** | `ONNX ↔ sklearn Maks Fark` | **`<= 1e-4` ($10^{-4}$)** | `skl2onnx` dönüşümü sonrası float32 sapmasının ihmal edilebilir düzeyde kalması. |
| **Diferansiyel Parite Kapısı 4** | `Python Oracle ↔ Web/Sunucu Maks Fark` | **`< 1e-4` ($10^{-4}$)** | `tests/parite.json` içindeki 5 referans girdide tüm ortamların aynı olasılığı üretmesi. |

---

## 4. Yeni Kabul Örnekleri (v2 Davranışsal Uyarlama)

Aşağıdaki 3 kabul örneği, modelin uç profillerde ve sınır durumlarında nasıl davranması gerektiğini eğitim öncesinde tanımlar:

### Kabul Örneği 1: Yüksek Tereddüt ve Birikimli Hata Yaşayan Çırak (Müdahale Zorunlu)
- **Önkoşul (Given):** Oyuncu `deneyim_seviyesi = 1` (Çırak) profilindedir, `sahne_zorlugu = 4` olan *Diferansiyel Parite* sahnesindedir, `karar_suresi_sn = 34.5` saniye boyunca beklemiş, seçenekler arasında `secenek_degistirme_sayisi = 5` kez gidip gelmiş, `onceki_kritik_hata_sayisi = 2`, `acilan_ipucu_sayisi = 2` ve `mimari_risk_skoru = 0.72` düzeyindedir.
- **Eylem (When):** Model bu 7 özelliği `[1, 4, 34.5, 5, 2, 2, 0.72]` vektörüyle değerlendirir.
- **Beklenen Sonuç (Then):**
  - `P(mudahale_gerekli = 1) >= 0.85` (Karar eşiği `0.55` üzerindedir).
  - Arayüzde `data-cy="adaptation-badge"` üzerinde **"Sokratik Mentor Müdahalesi Aktif"** görünür.
  - Sokratik Mentor kartı (`data-cy="mentor-hint-card"`) otomatik açılır ve anlatım derinliği *"Uyarlamalı Destek + Ödünleşim Analizi"* moduna geçer.

### Kabul Örneği 2: Hızlı ve Hatasız İlerleyen Usta Mühendis Adayı (Otonom Akış)
- **Önkoşul (Given):** Oyuncu `deneyim_seviyesi = 3` (Usta Adayı) profilindedir, `sahne_zorlugu = 2` olan sahnede `karar_suresi_sn = 6.2` saniyede karar vermiş, `secenek_degistirme_sayisi = 0` (hiç tereddüt etmemiş), `onceki_kritik_hata_sayisi = 0`, `acilan_ipucu_sayisi = 0` ve `mimari_risk_skoru = 0.10` düzeyindedir.
- **Eylem (When):** Model `[3, 2, 6.2, 0, 0, 0, 0.10]` vektörüyle çıkarım yapar.
- **Beklenen Sonuç (Then):**
  - `P(mudahale_gerekli = 1) <= 0.15` (Karar eşiği `0.55` altındadır).
  - Arayüzde `data-cy="adaptation-badge"` üzerinde **"Otonom Mühendis Modu (Akış)"** görünür.
  - Ek ipucu kartı açılmaz; oyuncu kesintisiz şekilde tam zorlukta ilerler.

### Kabul Örneği 3: Sınırda Kararsızlık Yaşayan Orta Seviye Oyuncu ve Diferansiyel Parite Kanıtı
- **Önkoşul (Given):** Oyuncu `deneyim_seviyesi = 2` (Kalfa), `sahne_zorlugu = 3`, `karar_suresi_sn = 21.0`, `secenek_degistirme_sayisi = 3`, `onceki_kritik_hata_sayisi = 1`, `acilan_ipucu_sayisi = 1`, `mimari_risk_skoru = 0.48` girdilerine sahiptir.
- **Eylem (When):** Aynı girdi vektörü hem Python Test Kahini (`tests/parite.py`) hem de tarayıcıdaki `onnxruntime-web` (`data-cy="p"`) tarafından çalıştırılır.
- **Beklenen Sonuç (Then):**
  - İki ortamın ürettiği olasılık değerleri arasındaki mutlak fark `|p_web - p_python| < 1e-4` ($10^{-4}$) olmalıdır.
  - Üretilen olasılık `0.55` eşiğine göre oyun modunu deterministik olarak belirler.

---

## 5. Regresyon Kabul Kriterleri (v1 E2E Uyumluluğu)

Yeni ONNX modeli devreye alındığında geçen haftaki **The Primer v1** temel işlevleri kırılmadan çalışmaya devam etmelidir:
1. **3 Soruluk Profil Başlangıcı:** Kullanıcı deneyim seviyesi, mimari yaklaşım ve kriz refleksi sorularını seçip hikâyeyi başlatabilmelidir.
2. **5 Sahnelik Karar Akışı:** Her sahnede seçenek seçildiğinde telemetri kaydedilmeli, sonraki sahneye geçilmeli ve 5. sahne sonunda **Mühendislik Karnesi (Final Özeti)** gösterilmelidir.
3. **Veri Toplama ve Gizlilik (KVKK):** Oturum kimlikleri rastgele (`ses_xxxx`) üretilmeli, kişisel veri toplanmamalı ve oturum kararları CSV formatında dışa aktarılabilmelidir.
