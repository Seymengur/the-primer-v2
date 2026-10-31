# Düşünce Egzersizi: "Sizinle Uyumlanan Sınav Tasarımı" (`SINAV_TASARIMI.md`)

> **Kapsam:** 1 Sayfalık Klasik (Kapalı Kitap) Vize Sınav Sorusu Tasarımı — Ezberi değil, yazılım testi ve ML mühendisliği kavramlarının **gerçek özünü ve gerekçesini** ölçen uyarlamalı (adaptive) senaryo.

---

## 1. Neden "Uyarlamalı" Klasik Sınav?

Klasik bir sınavda her öğrenciye tek kademeli soyut bir tanım sorusu (*"Diferansiyel test nedir?"*) sormak yalnızca ezberi ölçer. Tıpkı **The Primer v2**'nin oyuncunun kararına göre anlatım derinliğini uyarlaması gibi, iyi bir sınav sorusu da **3 kademeli (Çırak → Kalfa → Usta) aşamalı derinleşen tek bir kriz senaryosu** sunmalıdır. Böylece öğrenci kendi kavrayış derinliği kadar ilerler; kavramın tanımını değil, **hangi problemi neden çözdüğünü** kanıtlar.

---

## 2. Özgün Klasik Vize Sorusu: "Gece Yarısı 21. Adım ve İki Kahin İkilemi" (100 Puan)

### Senaryo
Bir e-öğrenme platformunda öğrencinin soruda takılıp takılmadığını tahmin eden kural tabanlı eski bir modülü (`v1`), Python `scikit-learn` ile eğittiğiniz küçük bir lojistik regresyon modeline (`v2`) dönüştürüp `model.onnx` olarak hem **tarayıcıda (`onnxruntime-web`)** hem de **mobil uygulamada (`onnxruntime-react-native`)** çalıştırmak istiyorsunuz. Dağıtım öncesi şu dört olay yaşanıyor:

1. Eski `v1` sisteminin kaynak kodunda hiçbir dokümantasyon yok; ancak mevcut 20 temel akışın yeni sürümde bozulmaması gerekiyor.
2. Eğitim kümesinde `%99` doğruluk alan yeni bir model, gerçek öğrencilerde tamamen alakasız ipuçları açıyor.
3. Aynı 300 referans girdiyle yapılan karşılaştırmada, Python betiği `0.734100` olasılık verirken tarayıcı `0.734142`, mobil uygulama ise `0.612000` veriyor.
4. Ekip arkadaşınız *"Float sayılar zaten tam eşit olmaz, testteki `assert p1 == p2` kontrolünü tamamen silelim ve modeli doğrudan tüm kullanıcılara `%100` açalım"* diyor.

### Sorular (Kavramların Özünü ve Gerekçesini Açıklayın)

- **Bölüm A — Miras Sistemi Korumak (25p):** Dokümantasyonu olmayan `v1` sisteminin mevcut davranışını değiştirmeden önce hangi test türünü (**Karakterizasyon Testi**) yazarsınız? Yeni model devreye alındığında ilk 20 adımın bozulmadığını doğrulayan **Regresyon Testi** ile arasındaki amaç farkı nedir?
- **Bölüm B — Doğru Ürünü mü, Ürünü Doğru mu Yapıyoruz? (25p):** 2. olayda modelin eğitim kodunun hatasız çalışması ile gerçek öğrencinin işine yaramaması arasındaki farkı **Doğrulama (Verification)** ve **Geçerleme (Validation)** kavramlarıyla açıklayın. Eğitimden *önce* tanımlanan bir **Eval Kapısı (Kabul Eşiği)** neden zorunludur?
- **Bölüm C — 300 Girdilik Diferansiyel Test, Test Kahini ve Hata Toleransı (30p):** 3. olayda Python betiği hangi rolü (**Test Kahini / Oracle**) üstlenmektedir? Tarayıcı (`Δ = 4.2e-5`) ile mobil (`Δ = 0.1221`) sonuçlarını **`1e-4` ($10^{-4}$) hata toleransı** çerçevesinde değerlendirin: Neden `==` kullanılmaz, neden kontrol tamamen silinemez ve mobildeki büyük sapmanın en olası sözleşme (`config.json`) nedeni nedir?
- **Bölüm D — Güvenli Devreye Alma (20p):** 4. olaydaki *"doğrudan `%100` açalım"* önerisini reddedip **Shadow Deployment → Canary (%5) → Rollback** zincirini bu senaryoda nasıl kurgulardınız?

---

## 3. Değerlendirme Anahtarı: Beklenen Öz Kavramlar ve Çeldiriciler

| Kavram | Beklenen "Öz / Gerekçe" Yanıtı (Tam Puan) | Tipik Ezber / Çeldirici Yanıt (Puan Kaybı) |
| :--- | :--- | :--- |
| **Karakterizasyon vs. Regresyon Testi** | Karakterizasyon testi *"sistem ne yapmalı?"* değil *"şu an gerçekte ne yapıyor?"* sorusunu kayda alır; regresyon testi ise bu kayıt altına alınan davranışın yeni değişiklikten sonra kırılmadığını denetler. | "İkisi de eski kodun çalışıp çalışmadığını test eder" diyerek amaç farkını açıklamamak. |
| **Verification vs. Validation (V&V)** | *Verification:* Modelin spesifikasyona ve matematiksel formüle uygun kodlanması. *Validation:* Modelin gerçek öğrenci problemini gerçekten çözmesi. | İki kavramı yalnızca "birim testi vs. arayüz testi" olarak daraltmak. |
| **Test Kahini (Oracle) & Diferansiyel Test** | Gerçek etiketin (*ground truth*) anlık bilinmediği yerde, güvenilir referans sistemin (Python ONNX) çıktısını referans kabul edip ikinci tüketiciyle (Web/Mobil) 300 girdi üzerinden kıyaslamak. | "Python daha hızlı olduğu için kullanılır" demek veya diferansiyel testi normal birim testi sanmak. |
| **Hata Toleransı (`1e-4`)** | IEEE-754 `float32` donanım/WASM yuvarlama farkları nedeniyle `4.2e-5 < 1e-4` kabul edilir (tarayıcı geçer); ancak `0.1221 >= 1e-4` gerçek bir **özellik sırası / ölçekleme hatasıdır** (mobil kalır). | "`==` yerine yuvarlama silelim" önerisini kabul etmek veya tarayıcıdaki `4.2e-5` farkını da hata saymak. |
| **Shadow vs. Canary** | *Shadow:* Gerçek trafikte sessizce çalışır, kullanıcıyı etkilemez. *Canary:* Küçük bir gerçek kullanıcı dilimini (`%5`) gerçekten etkiler. | Shadow ve Canary dağıtımlarını aynı şey (yalnızca "test ortamı") sanmak. |
