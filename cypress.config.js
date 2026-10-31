const { defineConfig } = require("cypress");

module.exports = defineConfig({
  e2e: {
    baseUrl: "http://127.0.0.1:8080",
    supportFile: false,
    specPattern: "cypress/e2e/**/*.cy.{js,jsx,ts,tsx}",
    video: true,
    videoCompression: false,
    viewportWidth: 1280,
    viewportHeight: 800,
  },
});
