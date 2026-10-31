/// <reference types="cypress" />

describe("The Primer v2 — Diferansiyel Parite ve Regresyon E2E Test Süiti", () => {
  beforeEach(() => {
    cy.visit("/");
    cy.get('[data-cy="engine-status"]', { timeout: 15000 }).should("contain.text", "ONNX Runtime Web Aktif");
  });

  it("(a) Diferansiyel Test (Parite Testi): tests/parite.json içindeki 5 referans girdide |p_web - p_oracle| < 1e-4 olmalıdır", () => {
    cy.get('[data-cy="tab-parity"]').click();

    cy.readFile("tests/parite.json").then((parite) => {
      expect(parite.cases).to.have.length(5);

      parite.cases.forEach((vaka, idx) => {
        cy.get(`[data-cy="load-case-${idx}"]`).click();

        cy.get('[data-cy="p"]')
          .invoke("text")
          .then((metin) => {
            const pWeb = parseFloat(metin.trim());
            const pOracle = Number(vaka.oracle_probability);
            const fark = Math.abs(pWeb - pOracle);

            cy.log(`${vaka.id}: Web=${pWeb} vs Oracle=${pOracle} (|Δ|=${fark.toExponential(2)})`);
            expect(fark, `${vaka.id} parite farkı 1e-4 altında olmalı`).to.be.lessThan(1e-4);
          });
      });
    });
  });

  it("(b) Davranışsal Uyarlama Testi: Aynı sahnede iki farklı oyuncu profiline model farklı karar verir ve oyunun davranışı değişir", () => {
    cy.get('[data-cy="start-story-btn"]').click();
    cy.get('[data-cy="scene-card"]').should("be.visible");

    // Oyuncu 1: Tereddütlü Çırak -> Sokratik Mentor Müdahalesi açılmalı (p >= 0.55)
    cy.get('[data-cy="simulate-hesitant-btn"]').click();
    cy.get('[data-cy="story-p"]')
      .invoke("text")
      .then((t) => {
        expect(parseFloat(t)).to.be.at.least(0.55);
      });
    cy.get('[data-cy="adaptation-badge"]').should("contain.text", "Sokratik Mentor Müdahalesi Aktif");
    cy.get('[data-cy="mentor-hint-card"]').should("be.visible");
    cy.get('[data-cy="adaptive-scaffolding"]').should("be.visible");

    // Oyuncu 2: Akıştaki Usta -> Otonom Mühendis Modu olmalı ve mentor kartı gizlenmeli (p < 0.55)
    cy.get('[data-cy="simulate-autonomous-btn"]').click();
    cy.get('[data-cy="story-p"]')
      .invoke("text")
      .then((t) => {
        expect(parseFloat(t)).to.be.lessThan(0.55);
      });
    cy.get('[data-cy="adaptation-badge"]').should("contain.text", "Otonom Mühendis Modu (Akış)");
    cy.get('[data-cy="mentor-hint-card"]').should("not.be.visible");
  });

  it("(c) Regresyon E2E Testi (v1 -> v2): 3 başlangıç sorusu ve 5 sahnelik hikâye kırılmadan tamamlanır ve telemetri kaydedilir", () => {
    cy.get('[data-cy="q1-select"]').select("2");
    cy.get('[data-cy="q2-select"]').select("1");
    cy.get('[data-cy="q3-select"]').select("2");
    cy.get('[data-cy="start-story-btn"]').click();

    for (let sahne = 1; sahne <= 5; sahne++) {
      cy.get('[data-cy="scene-number"]').should("contain.text", `Sahne ${sahne} / 5`);
      cy.get('[data-cy="choice-0"]').click();
      cy.get('[data-cy="confirm-choice-btn"]').click();
    }

    // 5. sahne sonunda Mühendislik Karnesi görünmeli
    cy.get('[data-cy="summary-card"]').should("be.visible");
    cy.get('[data-cy="summary-outcome"]').should("not.be.empty");

    // Veri Toplama Modu sekmesinde 5 gerçek karar satırı yer almalı
    cy.get('[data-cy="tab-data"]').click();
    cy.get('[data-cy="telemetry-log-body"] tr').should("have.length", 5);
  });
});
