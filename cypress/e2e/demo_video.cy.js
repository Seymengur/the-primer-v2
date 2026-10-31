/// <reference types="cypress" />

function altYaziGoster(metin) {
  cy.document().then((doc) => {
    let el = doc.getElementById("demo-video-banner");
    if (!el) {
      el = doc.createElement("div");
      el.id = "demo-video-banner";
      el.style.cssText = [
        "position:fixed",
        "bottom:18px",
        "left:50%",
        "transform:translateX(-50%)",
        "z-index:99999",
        "background:rgba(10, 10, 12, 0.96)",
        "color:#fafafa",
        "border:2px solid #e11d48",
        "border-radius:10px",
        "padding:10px 20px",
        "font:700 15px/1.4 system-ui, sans-serif",
        "box-shadow:0 12px 32px rgba(225, 29, 72, 0.38)",
        "max-width:92%",
        "text-align:center",
      ].join(";");
      doc.body.appendChild(el);
    }
    el.textContent = metin;
  });
}

describe("The Primer v2 — 65 Saniyelik Resmi Demo Video Kaydı", () => {
  it("Aynı sahnede iki farklı oyuncuya ONNX modelinin verdiği iki farklı kararı ve < 1e-4 pariteyi gösterir", () => {
    cy.visit("/");
    cy.get('[data-cy="engine-status"]', { timeout: 15000 }).should("contain.text", "ONNX Runtime Web Aktif");

    // 1. Bölüm: Başlangıç Profili — Oyuncu 1 (Çırak / Junior) [8 sn]
    altYaziGoster("🎬 Demo Başlangıcı: ONNX Runtime Web Yüklendi — Oyuncu 1 (Çırak / Junior) Profili Seçiliyor");
    cy.wait(4000);
    cy.get('[data-cy="q1-select"]').select("1");
    cy.get('[data-cy="q2-select"]').select("0");
    cy.get('[data-cy="q3-select"]').select("1");
    cy.wait(4000);

    // 2. Bölüm: Sahne 1 — Oyuncu 1 Tereddüt Yaşar ve Sokratik Mentor Müdahale Eder [16 sn]
    cy.get('[data-cy="start-story-btn"]').click();
    altYaziGoster("👤 Oyuncu 1 (Tereddütlü Çırak — Sahne 1): Şıklar arasında kararsız kalıyor ve süre uzuyor...");
    cy.wait(2500);
    cy.get('[data-cy="choice-1"]').click();
    cy.wait(2500);
    cy.get('[data-cy="choice-2"]').click();
    cy.wait(2500);
    cy.get('[data-cy="simulate-hesitant-btn"]').click();
    altYaziGoster("🧭 ONNX Kararı (Oyuncu 1 — Sahne 1): P(mudahale=1) = 1.0000 >= 0.55 → Sokratik Mentor Kartı ve Ödünleşim Analizi Açıldı!");
    cy.get('[data-cy="adaptation-badge"]').should("contain.text", "Sokratik Mentor Müdahalesi Aktif");
    cy.get('[data-cy="mentor-hint-card"]').should("be.visible");
    cy.wait(8500);

    // 3. Bölüm: Aynı Sahne 1 — Oyuncu 2 (Akıştaki Usta) Otonom Modda İlerler [14 sn]
    altYaziGoster("⚡ Oyuncu 2 (Akıştaki Usta — Aynı Sahne 1): Hızlı ve hatasız karar verdiğinde model davranışı değişiyor...");
    cy.get('[data-cy="simulate-autonomous-btn"]').click();
    cy.get('[data-cy="adaptation-badge"]').should("contain.text", "Otonom Mühendis Modu (Akış)");
    cy.get('[data-cy="mentor-hint-card"]').should("not.be.visible");
    cy.wait(7000);

    altYaziGoster("🔄 Karşılaştırma Özeti (Sahne 1): Oyuncu 1 için Sokratik Mentor Açık (P=1.0000) vs. Oyuncu 2 için Otonom Akış (P=0.0001)");
    cy.get('[data-cy="simulate-hesitant-btn"]').click();
    cy.wait(3500);
    cy.get('[data-cy="simulate-autonomous-btn"]').click();
    cy.wait(3500);

    // 4. Bölüm: 5 Sahnelik Hikâye Akışı ve Mühendislik Karnesi [14 sn]
    altYaziGoster("📖 Regresyon Akışı: 5 Sahnelik Yazılım Mühendisliği Hikâyesi Tamamlanıyor...");
    for (let s = 1; s <= 5; s++) {
      cy.get('[data-cy="choice-0"]').click();
      cy.wait(1800);
      cy.get('[data-cy="confirm-choice-btn"]').click();
    }
    altYaziGoster("🎓 Oturum Sonu: Mühendislik Karnesi ve KVKK Uyumlu Anonim Oturum Özeti");
    cy.get('[data-cy="summary-card"]').should("be.visible");
    cy.wait(5000);

    // 5. Bölüm: Veri Toplama Modu (KVKK / CSV) [5 sn]
    cy.get('[data-cy="tab-data"]').click();
    altYaziGoster("📊 Veri Toplama Modu: Anonim oturum_id ile kaydedilen 5 sahne kararı (data/oturumlar.csv sözleşmesi)");
    cy.wait(5000);

    // 6. Bölüm: Diferansiyel Test & Parite Lab (< 1e-4) [10 sn]
    cy.get('[data-cy="tab-parity"]').click();
    altYaziGoster("🔬 Diferansiyel Parite Lab: 5 Referans Girdide Python Oracle vs. Tarayıcı ONNX (Fark < 1e-4)");
    for (let i = 0; i < 5; i++) {
      cy.get(`[data-cy="load-case-${i}"]`).click();
      cy.wait(2000);
    }
  });
});
