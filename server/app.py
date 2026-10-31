"""The Primer v2 — Sunucu Tarafı ONNX Çıkarım Servisi (+10 Bonus Tüketici, server/app.py).

Hem yerel geliştirme ortamında (`python server/app.py`) hem de Render.com üzerinde çalışır.
Aynı `model/model.onnx` ve `model/config.json` sözleşmesini kullanarak:
- `GET  /health`              → servis ve model durumu + SHA256 özeti
- `POST /api/v1/predictions`  → `{"inputs": {...}}` alır, `{"probability": p, "decision": ...}` döner
- `GET  /api/v1/parity`       → `tests/parite.json` içindeki 5 referans girdiyi sunucuda koşturup
                                Python Oracle ile `< 1e-4` paritesini canlı doğrular.

Gizlilik (KVKK): İstek gövdesindeki `inputs` değerleri sunucu loglarına yazılmaz (filtrelenir).
"""
import argparse
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnxruntime as ort

KOK_DIZIN = Path(__file__).resolve().parent.parent
MODEL_DIR = KOK_DIZIN / "model"
PARITE_YOLU = KOK_DIZIN / "tests" / "parite.json"


class ModelYukleyici:
    def __init__(self):
        self.config_yolu = MODEL_DIR / "config.json"
        self.onnx_yolu = MODEL_DIR / "model.onnx"
        self.config = json.loads(self.config_yolu.read_text(encoding="utf-8"))
        self.oturum = ort.InferenceSession(str(self.onnx_yolu))
        self.ozellik_sirasi = [f["name"] for f in self.config["features"]]
        self.esik = float(self.config.get("decision_threshold", 0.55))
        self.sha256 = hashlib.sha256(self.onnx_yolu.read_bytes()).hexdigest()

    def tahmin_et(self, girdi: dict) -> dict:
        eksik = [ad for ad in self.ozellik_sirasi if ad not in girdi]
        if eksik:
            raise ValueError(f"Eksik özellikler: {eksik}")
        vektor = np.array([[float(girdi[ad]) for ad in self.ozellik_sirasi]], dtype=np.float32)
        cikti = self.oturum.run(["probabilities"], {"input": vektor})[0][0]
        p_otonom = float(cikti[0])
        p_mudahale = float(cikti[1])
        return {
            "probability": round(p_mudahale, 6),
            "probability_4dec": round(p_mudahale, 4),
            "decision": "mudahale" if p_mudahale >= self.esik else "otonom",
            "threshold": self.esik,
            "proba": {
                self.config["class_labels"]["0"]: round(p_otonom, 4),
                self.config["class_labels"]["1"]: round(p_mudahale, 4),
            },
            "model_sha256": self.sha256[:12],
        }


SERVIS = None


class TahminHandler(BaseHTTPRequestHandler):
    def log_message(self, format_str, *args):
        # KVKK & AGENTS.md kuralı: kullanıcı girdileri (`inputs`) loglara yazılmaz
        print(f"[Primer-Server] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    def _json_yanit(self, kod: int, veri: dict):
        govde = json.dumps(veri, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(kod)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", str(len(govde)))
        self.end_headers()
        self.wfile.write(govde)

    def do_OPTIONS(self):
        self._json_yanit(200, {"ok": True})

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self._json_yanit(
                200,
                {
                    "status": "ok",
                    "service": "the-primer-v2-onnx-server",
                    "version": SERVIS.config.get("version", "2.0.0"),
                    "model_sha256": SERVIS.sha256,
                    "features": SERVIS.ozellik_sirasi,
                },
            )
            return

        if self.path == "/api/v1/parity":
            parite = json.loads(PARITE_YOLU.read_text(encoding="utf-8"))
            sonuclar = []
            maks_fark = 0.0
            for vaka in parite["cases"]:
                res = SERVIS.tahmin_et(vaka["input"])
                fark = abs(res["probability"] - vaka["oracle_probability"])
                maks_fark = max(maks_fark, fark)
                sonuclar.append({
                    "id": vaka["id"],
                    "server_probability": res["probability"],
                    "oracle_probability": vaka["oracle_probability"],
                    "abs_diff": round(fark, 8),
                    "within_1e_4": fark < 1e-4,
                })
            self._json_yanit(
                200,
                {
                    "all_passed": maks_fark < 1e-4,
                    "max_abs_diff": round(maks_fark, 8),
                    "tolerance": 1e-4,
                    "cases": sonuclar,
                },
            )
            return

        self._json_yanit(404, {"error": "Uç nokta bulunamadı"})

    def do_POST(self):
        if self.path != "/api/v1/predictions":
            self._json_yanit(404, {"error": "Uç nokta bulunamadı"})
            return
        try:
            uzunluk = int(self.headers.get("Content-Length", "0"))
            ham = self.rfile.read(uzunluk).decode("utf-8")
            istek = json.loads(ham)
            girdi = istek.get("inputs", istek)
            sonuc = SERVIS.tahmin_et(girdi)
            self._json_yanit(200, sonuc)
        except Exception as e:
            self._json_yanit(400, {"error": str(e)})


def main():
    global SERVIS
    parser = argparse.ArgumentParser(description="The Primer v2 — ONNX Tahmin Sunucusu")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "3100")))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    SERVIS = ModelYukleyici()
    httpd = HTTPServer((args.host, args.port), TahminHandler)
    print(f"✓ The Primer v2 ONNX Sunucusu dinlemede: http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
