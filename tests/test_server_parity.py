"""The Primer v2 — Sunucu Tüketicisi (+10 Bonus) Parite Doğrulama Testi (tests/test_server_parity.py)."""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))

from server.app import ModelYukleyici

def main():
    yukleyici = ModelYukleyici()
    parite = json.loads((KOK / "tests" / "parite.json").read_text(encoding="utf-8"))
    maks_fark = 0.0
    for vaka in parite["cases"]:
        sonuc = yukleyici.tahmin_et(vaka["input"])
        fark = abs(sonuc["probability"] - vaka["oracle_probability"])
        maks_fark = max(maks_fark, fark)
        assert fark < 1e-4, f"Sunucu parite hatası: {vaka['id']} fark={fark}"
        assert sonuc["decision"] == vaka["expected_decision"]
    print(f"✓ Sunucu tüketicisi (server/app.py) 5/5 referans girdide doğrulandı (maks fark={maks_fark:.2e} < 1e-4)")

if __name__ == "__main__":
    main()
