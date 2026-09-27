
const { chromium } = require("playwright");
const fs = require("fs");

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
    deviceScaleFactor: 1,
  });

  const failures = [];
  const consoleErrors = [];
  const pageErrors = [];

  page.on("console", msg => {
    if (msg.type() === "error") {
      consoleErrors.push(msg.text());
    }
  });

  page.on("pageerror", err => {
    pageErrors.push(String(err));
  });

  // Mock backend so we can test the COMPLETE successful analysis workflow
  // without pretending that the real backend is connected.
  await page.route("**/api/analyze", async route => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        status: "analysis_complete",
        stage: "attest",
        evidence: {
          signal: "mock-test-signal",
          score: 0.87,
          source: "E2E TEST ONLY"
        }
      })
    });
  });

  const fail = (name, detail) => {
    failures.push({ name, detail });
    console.log(`❌ ${name}: ${detail}`);
  };

  const pass = name => console.log(`✅ ${name}`);

  // ---------------- PAGE LOAD ----------------
  const response = await page.goto("http://127.0.0.1:3000", {
    waitUntil: "networkidle",
    timeout: 30000
  });

  if (!response || response.status() !== 200) {
    fail("Page load", `HTTP ${response?.status()}`);
  } else {
    pass("Page loads");
  }

  // ---------------- REQUIRED TEXT ----------------
  for (const text of [
    "MechGuard",
    "See what's",
    "Attest",
    "Watch",
    "Review",
    "Try MechGuard",
    "Current evidence"
  ]) {
    const count = await page.getByText(text, { exact: false }).count();
    if (!count) fail(`Content: ${text}`, "not found");
    else pass(`Content: ${text}`);
  }

  // ---------------- THEME ----------------
  const themeButton = page.getByRole("button", {
    name: /Switch to (light|dark) mode/i
  });

  if (!(await themeButton.count())) {
    fail("Theme toggle", "button not found");
  } else {
    const before = await page.evaluate(
      () => document.documentElement.classList.contains("dark")
    );

    await themeButton.click();

    const after = await page.evaluate(
      () => document.documentElement.classList.contains("dark")
    );

    if (before === after) {
      fail("Theme toggle", "HTML dark class did not change");
    } else {
      pass("Theme toggle changes theme");
    }

    await themeButton.click();
  }

  // ---------------- HERO NODE INTERACTION ----------------
  const modelButton = page.getByRole("button", { name: /Model/i }).first();

  if (await modelButton.count()) {
    await modelButton.click();
    pass("Hero Model node clickable");
  } else {
    fail("Hero Model node", "not found");
  }

  const attestButtons = page.getByRole("button", { name: /Attest/i });
  if (await attestButtons.count()) {
    await attestButtons.first().click();
    pass("Attest interaction");
  } else {
    fail("Attest interaction", "button not found");
  }

  const watchButtons = page.getByRole("button", { name: /Agent A/i });
  if (await watchButtons.count()) {
    await watchButtons.first().click();
    pass("Watch/agent interaction");
  } else {
    fail("Watch/agent interaction", "button not found");
  }

  // ---------------- PRODUCT LAYERS ----------------
  for (const layer of ["Attest", "Watch", "Review"]) {
    const buttons = page.getByRole("button", { name: new RegExp(layer) });
    if (await buttons.count()) {
      await buttons.last().click();
      pass(`Product layer ${layer} clickable`);
    } else {
      fail(`Product layer ${layer}`, "button not found");
    }
  }

  // ---------------- PAYLOAD ----------------
  const textarea = page.locator("textarea");

  if (!(await textarea.count())) {
    fail("Payload input", "textarea missing");
  } else {
    await textarea.fill(
      '{"model":"test-model","checkpoint":"checkpoint-1","analysis":"test"}'
    );

    const value = await textarea.inputValue();

    if (!value.includes("test-model")) {
      fail("Payload input", "value was not retained");
    } else {
      pass("Payload input works");
    }
  }

  // ---------------- FILE INPUT ----------------
  const fileInput = page.locator('input[type="file"]');

  if (!(await fileInput.count())) {
    fail("File upload", "file input missing");
  } else {
    const temp = "/tmp/mechguard-e2e.json";

    fs.writeFileSync(
      temp,
      JSON.stringify({
        model: "e2e-model",
        checkpoint: "checkpoint-1",
        test: true
      })
    );

    await fileInput.setInputFiles(temp);

    const fileText = await page.getByText("mechguard-e2e.json", {
      exact: false
    }).count();

    if (!fileText) {
      fail("File upload", "selected file was not displayed");
    } else {
      pass("File upload works");
    }
  }

  // ---------------- ANALYSIS ----------------
  const runButton = page.getByRole("button", {
    name: /Run MechGuard/i
  });

  if (!(await runButton.count())) {
    fail("Run analysis", "button not found");
  } else {
    await runButton.click();

    await page.waitForTimeout(1800);

    const resultText = await page.getByText("Analysis returned", {
      exact: false
    }).count();

    if (!resultText) {
      fail("Analysis workflow", "successful backend result was not rendered");
    } else {
      pass("Analysis workflow renders backend result");
    }
  }

  // ---------------- NAVIGATION ----------------
  for (const href of ["#system", "#product", "#try", "#evidence"]) {
    const link = page.locator(`a[href="${href}"]`).first();

    if (await link.count()) {
      await link.click();
      pass(`Navigation ${href}`);
    } else {
      fail(`Navigation ${href}`, "link missing");
    }
  }

  // ---------------- DESKTOP OVERFLOW ----------------
  const desktopOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth + 2
  );

  if (desktopOverflow) {
    fail("Desktop horizontal overflow", "page wider than viewport");
  } else {
    pass("Desktop horizontal overflow");
  }

  // ---------------- MOBILE ----------------
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(500);

  const mobileOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth + 2
  );

  if (mobileOverflow) {
    fail("Mobile horizontal overflow", "page wider than 390px viewport");
  } else {
    pass("Mobile horizontal overflow");
  }

  // ---------------- MOBILE KEY CONTENT ----------------
  for (const text of ["MechGuard", "Attest", "Watch", "Review"]) {
    const count = await page.getByText(text, { exact: false }).count();
    if (!count) fail(`Mobile content: ${text}`, "missing");
    else pass(`Mobile content: ${text}`);
  }

  // ---------------- VISUAL SCREENSHOTS ----------------
  await page.screenshot({
    path: "/content/mechguard-site/e2e-desktop.png",
    fullPage: true
  });

  await page.setViewportSize({ width: 390, height: 844 });

  await page.screenshot({
    path: "/content/mechguard-site/e2e-mobile.png",
    fullPage: true
  });

  // ---------------- CONSOLE / JS ERRORS ----------------
  if (consoleErrors.length) {
    console.log("\n⚠️ CONSOLE ERRORS");
    consoleErrors.forEach(x => console.log(x));
  } else {
    pass("Browser console");
  }

  if (pageErrors.length) {
    console.log("\n⚠️ PAGE ERRORS");
    pageErrors.forEach(x => console.log(x));
  } else {
    pass("Runtime JavaScript");
  }

  console.log("\n===== E2E SUMMARY =====");
  console.log(`Failures: ${failures.length}`);

  if (failures.length) {
    console.log(JSON.stringify(failures, null, 2));
    process.exitCode = 1;
  } else {
    console.log("🎉 ALL INTERACTION TESTS PASSED");
  }

  await browser.close();
})();
