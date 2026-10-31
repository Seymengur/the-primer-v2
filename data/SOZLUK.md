# Veri Sözlüğü (`data/SOZLUK.md`)

Bu belge, `data/oturumlar.csv` veri setindeki her bir sütunun anlamını, veri tipini, birimini, geçerli değer aralığını ve gizlilik (KVKK) ilkelerini tanımlar.

---

## 1. Veri Toplama ve KVKK / Gizlilik İlkeleri

- **Birim Satır Tanımı:** `data/oturumlar.csv` içindeki her satır, bir oyuncunun **tek bir hikâye sahnesinde verdiği bir kararı** temsil eder (5 sahnelik bir oturum = 5 satır).
- **Kişisel Veri Yasağı (KVKK Uyumluluğu):** Oyuncuların adı, soyadı, öğrenci numarası, e-posta adresi, IP adresi, tarayıcı parmak izi (fingerprint) veya konum bilgisi **kesinlikle toplanmaz ve kaydedilmez**.
- **Anonim Oturum Kimliği:** Her oyun oturumuna yalnızca o oturumdaki sahneleri gruplamak amacıyla rastgele üretilen anonim bir kimlik (`oturum_id`, örn. `ses_g01_9f2a`) atanır.
- **Veri Kaynağı Ayrımı:** Gerçek insan oyuncuların Veri Toplama Modu ile ürettiği oturumlar `kaynak = gercek`, sabit tohumlu (`SEED = 13`) simülasyon kuralıyla üretilen satırlar `kaynak = sentetik` olarak işaretlenmiştir.
  - **Gerçek Oturum Sayısı:** **18 gerçek oturum** (`ses_g01` – `ses_g18`, 4 farklı yazılım mühendisliği öğrencisi tarafından oynanmış toplam **90 gerçek karar satırı**).
  - **Sentetik Oturum Sayısı:** **32 sentetik oturum** (`ses_s01` – `ses_s32`, `SEED = 13` ile üretilmiş **160 sentetik karar satırı**).
  - **Toplam:** **50 oturum / 250 karar satırı**.

---

## 2. Sütun Tanımları

| # | Sütun Adı | Rolü | Veri Tipi | Birim | Geçerli Aralık | Açıklama |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `oturum_id` | Meta veri | `string` | Kimlik (ID) | `ses_g01` .. `ses_s32` | Anonim, rastgele üretilmiş oturum kimliği. Eğitimde özellik olarak **kullanılmaz** (veri sızıntısını önlemek için çıkarılır). |
| 2 | `sahne_no` | Meta veri | `int` | Sahne sırası | `1` – `5` | Oyuncunun bulunduğu hikâye sahnesinin sıra numarası (`1`: Regresyon, `2`: Shadow/Canary, `3`: KVKK, `4`: Oracle/Parite, `5`: Rollback). |
| 3 | `deneyim_seviyesi` | **Model Girdisi (Özellik 1)** | `float32` (tam sayı) | Seviye (Ordinal) | `1`, `2`, `3` | Girişteki 3 sorudan hesaplanan mühendislik hazırlık seviyesi (`1`: Çırak/Junior, `2`: Kalfa/Mid, `3`: Usta/Senior Adayı). |
| 4 | `sahne_zorlugu` | **Model Girdisi (Özellik 2)** | `float32` (tam sayı) | Zorluk katsayısı | `1` – `5` | Mevcut sahnedeki yazılım mühendisliği ikileminin bilişsel ve teknik zorluk derecesi. |
| 5 | `karar_suresi_sn` | **Model Girdisi (Özellik 3)** | `float32` | Saniye (`sn`) | `1.5` – `60.0` | Sahne ekrana geldiği andan oyuncunun kararını onayladığı ana kadar geçen süre (`performance.now()` farkı, saniye cinsinden). |
| 6 | `secenek_degistirme_sayisi` | **Model Girdisi (Özellik 4)** | `float32` (tam sayı) | Adet (Tereddüt) | `0` – `10` | Oyuncunun kararı onaylamadan önce farklı şıklar arasında kaç kez fikir/odak değiştirdiği (kararsızlık ve tereddüt göstergesi). |
| 7 | `onceki_kritik_hata_sayisi` | **Model Girdisi (Özellik 5)** | `float32` (tam sayı) | Adet | `0` – `4` | Oyuncunun o sahneye gelene kadar önceki sahnelerde seçtiği yüksek riskli / hatalı mimari kararların toplam sayısı. |
| 8 | `acilan_ipucu_sayisi` | **Model Girdisi (Özellik 6)** | `float32` (tam sayı) | Adet | `0` – `4` | Oyuncunun önceki sahnelerde Sokratik mentordan talep ettiği veya açtığı toplam ipucu sayısı. |
| 9 | `mimari_risk_skoru` | **Model Girdisi (Özellik 7)** | `float32` | Oran (Boyutsuz) | `0.00` – `1.00` | Oyuncunun önceki seçimleri sonucunda biriken teknik borç ve kötü sona (sistem çöküşüne) gitme riski (`0.00`: tamamen güvenli, `1.00`: kritik çöküş riski). |
| 10 | `mudahale_gerekli` | **Hedef Değişken (`target`)** | `int` (`0` \| `1`) | İkili Sınıf | `0` veya `1` | `1`: Oyuncuya Sokratik Mentor müdahalesi ve uyarlamalı anlatım desteği verilmeli; `0`: Oyuncu otonom akışta bırakılmalı. |
| 11 | `kaynak` | Meta veri (Zorunlu) | `string` | Kategori | `gercek` \| `sentetik` | Satırın gerçek kullanıcı oturumundan mı (`gercek`) yoksa deterministik simülasyondan mı (`sentetik`) geldiğini belirtir. |
