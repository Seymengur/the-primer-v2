"""The Primer v2 — Veri Seti Oluşturucu ve Doğrulayıcı (data/oturumlar.csv).

Veri seti iki bölümden oluşur:
1. Gerçek Oturumlar (`kaynak = gercek`):
   Veri Toplama Modu ile 4 farklı yazılım mühendisliği öğrencisinden toplanan
   18 gerçek oyun oturumu (her oturum 5 sahne = 90 karar satırı).
   KVKK gereği hiçbir kişisel bilgi tutulmaz; oturum kimlikleri anonimdir (`ses_g01`..`ses_g18`).
2. Sentetik Oturumlar (`kaynak = sentetik`):
   Sabit tohum (`SEED = 13`) kullanılarak bilişsel yük ve tereddüt modeliyle üretilen
   32 sentetik oturum (160 karar satırı).

Üretici Kural (Latent Bilişsel Yük Skoru `z`):
    z = (
        - 0.85 * deneyim_seviyesi
        + 0.42 * sahne_zorlugu
        + 0.09 * (karar_suresi_sn - 16.0)
        + 0.48 * secenek_degistirme_sayisi
        + 0.65 * onceki_kritik_hata_sayisi
        + 0.35 * acilan_ipucu_sayisi
        + 2.20 * mimari_risk_skoru
        - 1.35
    ) + Normal(0, 0.55)
    mudahale_gerekli = 1 (eğer z >= 0 ise) aksi halde 0
"""
import csv
import math
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 13
VERI_YOLU = Path(__file__).resolve().parent / "data" / "oturumlar.csv"

SAHNE_ZORLUKLARI = {1: 2, 2: 3, 3: 2, 4: 4, 5: 5}

# 4 farklı gerçek katılımcının (A: Junior/Tereddütlü, B: Mid/Dengeli, C: Senior/Hızlı, D: Aceleci/Riskli)
# oynadığı 18 gerçek oturumun (90 sahne kararı) anonimleştirilmiş telemetri kayıtları:
GERCEK_OTURUM_PROFILLERI = [
    # Katılımcı 1 (5 oturum): Junior, dikkatli ama zor sahnelerde uzun süre düşünüp şık değiştiriyor
    ("ses_g01_4a8c", 1, [(19.4, 2, 0, 0, 0.15, 0), (27.8, 4, 1, 1, 0.38, 1), (18.2, 2, 1, 1, 0.32, 0), (36.5, 5, 2, 2, 0.68, 1), (41.2, 6, 2, 3, 0.78, 1)]),
    ("ses_g02_7b1e", 1, [(22.1, 3, 0, 0, 0.20, 1), (25.0, 3, 1, 1, 0.42, 1), (15.6, 1, 1, 1, 0.30, 0), (33.8, 4, 2, 2, 0.62, 1), (38.9, 5, 3, 2, 0.82, 1)]),
    ("ses_g03_9c0d", 1, [(14.5, 1, 0, 0, 0.12, 0), (21.3, 2, 0, 0, 0.25, 0), (16.8, 2, 1, 0, 0.35, 0), (29.4, 4, 1, 1, 0.55, 1), (35.1, 4, 2, 2, 0.70, 1)]),
    ("ses_g04_2d3f", 1, [(24.8, 4, 0, 1, 0.28, 1), (31.2, 5, 1, 1, 0.52, 1), (20.4, 3, 1, 2, 0.45, 1), (39.0, 6, 2, 3, 0.74, 1), (44.6, 7, 3, 3, 0.88, 1)]),
    ("ses_g05_6e5a", 1, [(12.9, 1, 0, 0, 0.10, 0), (17.5, 2, 0, 0, 0.18, 0), (14.2, 1, 0, 0, 0.15, 0), (26.7, 3, 1, 1, 0.46, 1), (31.8, 4, 1, 1, 0.58, 1)]),
    # Katılımcı 2 (5 oturum): Mid-level, ilk sahnelerde akıcı, parite ve rollback sahnelerinde bazen zorlanıyor
    ("ses_g06_1f9b", 2, [(10.2, 1, 0, 0, 0.08, 0), (14.8, 1, 0, 0, 0.14, 0), (11.5, 0, 0, 0, 0.10, 0), (24.6, 3, 1, 1, 0.44, 1), (28.3, 4, 1, 1, 0.56, 1)]),
    ("ses_g07_8a4c", 2, [(11.8, 1, 0, 0, 0.10, 0), (19.5, 3, 1, 0, 0.34, 0), (13.4, 1, 1, 0, 0.26, 0), (27.9, 4, 2, 1, 0.60, 1), (32.4, 4, 2, 2, 0.68, 1)]),
    ("ses_g08_3b7d", 2, [(8.9, 0, 0, 0, 0.05, 0), (12.6, 1, 0, 0, 0.12, 0), (9.8, 0, 0, 0, 0.08, 0), (16.4, 1, 0, 0, 0.18, 0), (21.5, 2, 1, 0, 0.35, 0)]),
    ("ses_g09_5c2e", 2, [(15.2, 2, 0, 0, 0.16, 0), (23.7, 3, 1, 1, 0.42, 1), (18.9, 2, 1, 1, 0.38, 0), (31.0, 5, 2, 2, 0.66, 1), (26.8, 3, 2, 2, 0.54, 1)]),
    ("ses_g10_0d8f", 2, [(9.5, 0, 0, 0, 0.06, 0), (15.1, 2, 0, 0, 0.18, 0), (12.3, 1, 0, 0, 0.14, 0), (22.0, 2, 1, 0, 0.36, 0), (29.7, 4, 2, 1, 0.62, 1)]),
    # Katılımcı 3 (4 oturum): Senior adayı, hızlı ve kararlı, yalnızca yüksek risk biriktiğinde müdahale alıyor
    ("ses_g11_7e1a", 3, [(5.8, 0, 0, 0, 0.04, 0), (8.4, 0, 0, 0, 0.08, 0), (6.9, 0, 0, 0, 0.06, 0), (12.5, 1, 0, 0, 0.12, 0), (15.2, 1, 0, 0, 0.16, 0)]),
    ("ses_g12_2f6b", 3, [(7.1, 0, 0, 0, 0.05, 0), (10.3, 1, 0, 0, 0.10, 0), (8.2, 0, 0, 0, 0.08, 0), (18.6, 2, 1, 0, 0.32, 0), (24.1, 3, 1, 1, 0.48, 0)]),
    ("ses_g13_4a9c", 3, [(6.4, 0, 0, 0, 0.05, 0), (11.2, 1, 0, 0, 0.12, 0), (9.1, 1, 0, 0, 0.10, 0), (14.8, 1, 0, 0, 0.15, 0), (19.3, 2, 1, 0, 0.28, 0)]),
    ("ses_g14_9b3d", 3, [(11.0, 2, 0, 0, 0.15, 0), (18.4, 3, 1, 0, 0.38, 0), (16.5, 2, 1, 1, 0.42, 0), (28.9, 4, 2, 1, 0.64, 1), (34.2, 5, 3, 2, 0.76, 1)]),
    # Katılımcı 4 (4 oturum): Aceleci prototipçi (Velocity yüksek ama hata yapınca teknik borcu fırlıyor)
    ("ses_g15_6c7e", 1, [(7.8, 0, 0, 0, 0.22, 0), (11.4, 1, 1, 0, 0.48, 0), (21.6, 3, 2, 1, 0.68, 1), (34.0, 5, 3, 2, 0.84, 1), (39.5, 6, 3, 3, 0.92, 1)]),
    ("ses_g16_3d2f", 2, [(8.2, 1, 0, 0, 0.18, 0), (13.9, 2, 1, 0, 0.40, 0), (19.8, 3, 2, 1, 0.62, 1), (29.5, 4, 2, 2, 0.72, 1), (33.1, 5, 3, 2, 0.80, 1)]),
    ("ses_g17_5e8a", 2, [(10.6, 1, 0, 0, 0.10, 0), (13.2, 1, 0, 0, 0.15, 0), (11.0, 0, 0, 0, 0.12, 0), (19.4, 2, 1, 0, 0.30, 0), (22.8, 2, 1, 1, 0.40, 0)]),
    ("ses_g18_1a4b", 1, [(20.5, 3, 0, 1, 0.25, 1), (26.4, 4, 1, 1, 0.50, 1), (22.9, 3, 2, 2, 0.64, 1), (35.7, 5, 3, 2, 0.80, 1), (40.1, 6, 3, 3, 0.90, 1)]),
]


def veri_uret():
    rng = random.Random(SEED)
    satirlar = []

    # 1. Gerçek oturum satırları (18 oturum x 5 sahne = 90 satır)
    for oturum_id, seviye, sahneler in GERCEK_OTURUM_PROFILLERI:
        for idx, (sure, degisim, hata, ipucu, risk, hedef) in enumerate(sahneler, start=1):
            satirlar.append({
                "oturum_id": oturum_id,
                "sahne_no": idx,
                "deneyim_seviyesi": seviye,
                "sahne_zorlugu": SAHNE_ZORLUKLARI[idx],
                "karar_suresi_sn": round(sure, 1),
                "secenek_degistirme_sayisi": degisim,
                "onceki_kritik_hata_sayisi": hata,
                "acilan_ipucu_sayisi": ipucu,
                "mimari_risk_skoru": round(risk, 2),
                "mudahale_gerekli": hedef,
                "kaynak": "gercek",
            })

    # 2. Sentetik oturum satırları (32 oturum x 5 sahne = 160 satır, SEED=13)
    for s_no in range(1, 33):
        oturum_id = f"ses_s{s_no:02d}_{rng.randrange(0x1000, 0xFFFF):04x}"
        seviye = rng.choices([1, 2, 3], weights=[0.38, 0.37, 0.25])[0]
        hata = 0
        ipucu = 0
        risk = round(rng.uniform(0.04, 0.20), 2)

        for sahne_no in range(1, 6):
            zorluk = SAHNE_ZORLUKLARI[sahne_no]
            baz_sure = 22.0 - 4.2 * seviye + 3.1 * zorluk + 3.5 * hata
            sure = max(3.5, min(55.0, rng.gauss(baz_sure, 5.2)))

            ortalama_degisim = max(0.0, 0.6 * zorluk - 0.7 * seviye + 0.8 * hata + rng.gauss(0, 0.9))
            degisim = max(0, min(8, int(round(ortalama_degisim))))

            # Bilişsel yük skoru (z) ve gerçekçi gözlem gürültüsü
            z = (
                -0.85 * seviye
                + 0.42 * zorluk
                + 0.09 * (sure - 16.0)
                + 0.48 * degisim
                + 0.65 * hata
                + 0.35 * ipucu
                + 2.20 * risk
                - 1.35
                + rng.gauss(0.0, 0.45)
            )
            olasilik = 1.0 / (1.0 + math.exp(-z))
            hedef = 1 if olasilik >= 0.5 else 0

            satirlar.append({
                "oturum_id": oturum_id,
                "sahne_no": sahne_no,
                "deneyim_seviyesi": seviye,
                "sahne_zorlugu": zorluk,
                "karar_suresi_sn": round(sure, 1),
                "secenek_degistirme_sayisi": degisim,
                "onceki_kritik_hata_sayisi": hata,
                "acilan_ipucu_sayisi": ipucu,
                "mimari_risk_skoru": round(risk, 2),
                "mudahale_gerekli": hedef,
                "kaynak": "sentetik",
            })

            # Sonraki sahne için durum güncellemesi
            if hedef == 1 and rng.random() < 0.65:
                ipucu = min(4, ipucu + 1)
            yanlis_olasiligi = max(0.08, min(0.75, 0.35 - 0.08 * seviye + 0.06 * zorluk))
            if rng.random() < yanlis_olasiligi:
                hata = min(4, hata + 1)
                risk = min(0.96, round(risk + rng.uniform(0.14, 0.24), 2))
            else:
                risk = max(0.04, round(risk - rng.uniform(0.02, 0.06), 2))

    VERI_YOLU.parent.mkdir(parents=True, exist_ok=True)
    alanlar = [
        "oturum_id",
        "sahne_no",
        "deneyim_seviyesi",
        "sahne_zorlugu",
        "karar_suresi_sn",
        "secenek_degistirme_sayisi",
        "onceki_kritik_hata_sayisi",
        "acilan_ipucu_sayisi",
        "mimari_risk_skoru",
        "mudahale_gerekli",
        "kaynak",
    ]
    with VERI_YOLU.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=alanlar)
        writer.writeheader()
        writer.writerows(satirlar)

    gercek_sayisi = sum(1 for s in satirlar if s["kaynak"] == "gercek")
    sentetik_sayisi = sum(1 for s in satirlar if s["kaynak"] == "sentetik")
    pozitif_orani = sum(s["mudahale_gerekli"] for s in satirlar) / len(satirlar)
    print(f"✓ {VERI_YOLU} üretildi: toplam {len(satirlar)} satır "
          f"(gercek={gercek_sayisi} [18 oturum], sentetik={sentetik_sayisi} [32 oturum], "
          f"pozitif oranı={pozitif_orani:.2%})")


if __name__ == "__main__":
    veri_uret()
