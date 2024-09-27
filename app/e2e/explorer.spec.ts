import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import AxeBuilder from "@axe-core/playwright";

async function openExplorer(page: Page) {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Two synthetic stacks" }),
  ).toBeVisible();
}

async function assertNoOverflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  ).toBe(true);
}

test("candidate comparison responds to channel, keyboard angle, and comparison action", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await openExplorer(page);
  await assertNoOverflow(page);
  const firstPath = await page
    .locator(".spectra-card .plot")
    .first()
    .locator("path")
    .first()
    .getAttribute("d");
  await page
    .getByRole("button", { name: "Absorber region", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Absorber region", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("img", {
      name: "Absorber reflectivity for two synthetic stacks at 6 degrees",
      exact: true,
    }),
  ).toBeVisible();
  expect(
    await page
      .locator(".spectra-card .plot")
      .first()
      .locator("path")
      .first()
      .getAttribute("d"),
  ).not.toEqual(firstPath);
  await page.getByRole("slider", { name: "Incidence angle" }).focus();
  await page.keyboard.press("End");
  await expect(page.locator("#angle-value")).toHaveText("30°");
  const distanceAtThirty = Number(
    (await page.locator(".distance strong").innerText()).replaceAll(",", ""),
  );
  await page.getByRole("button", { name: "Return to the 6° view" }).click();
  await expect(page.locator("#angle-value")).toHaveText("6°");
  expect(distanceAtThirty).toBeGreaterThan(
    Number(await page.locator(".distance strong").innerText()),
  );
  await page.getByRole("button", { name: "Compare at 30°" }).click();
  await expect(page.locator("#angle-value")).toHaveText("30°");
  await expect(
    page.getByRole("button", { name: "Return to the 6° view" }),
  ).toBeFocused();
  await assertNoOverflow(page);
  expect(errors).toEqual([]);
});

test("measured and learned exhibits display data with appropriate limits", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await openExplorer(page);
  await page.getByRole("button", { name: "Measured spectra" }).click();
  await expect(
    page.getByRole("heading", { name: "Measured reflectivity and refit" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", {
      name: "Signed reflectivity residuals",
      exact: true,
    }),
  ).toBeVisible();
  await page.getByRole("slider", { name: "Incidence angle" }).focus();
  await page.keyboard.press("End");
  await expect(page.locator("#angle-value")).toHaveText("8°");
  await expect(
    page.getByRole("heading", { name: "An exploratory holdout." }),
  ).toBeVisible();
  await assertNoOverflow(page);
  await page.getByRole("button", { name: "Learned inference" }).click();
  await expect(page.locator(".metrics-table tbody tr")).not.toHaveCount(0);
  await expect(page.locator(".scatter circle")).toHaveCount(500);
  if (page.viewportSize()!.width < 700) {
    const plotBox = (await page.locator(".scatter").boundingBox())!;
    expect(plotBox.width).toBeGreaterThan(page.viewportSize()!.width * 0.8);
  }
  const nominalPoint = await page
    .locator(".scatter circle")
    .first()
    .getAttribute("cy");
  await page.getByRole("button", { name: "Shifted", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Shifted", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("img", {
      name: "Predicted versus true phase offsets for Shifted",
      exact: true,
    }),
  ).toBeVisible();
  expect(
    await page.locator(".scatter circle").first().getAttribute("cy"),
  ).not.toEqual(nominalPoint);
  await page.getByRole("button", { name: "Methods and assumptions" }).click();
  await expect(page.locator("#provenance-content")).toBeVisible();
  await expect(page.locator("#provenance-content")).toContainText(
    "Experimental phase accuracy has not been established",
  );
  await expect(
    page.getByRole("button", { name: "Spectra CSV" }),
  ).toBeDisabled();
  await expect(page.getByRole("button", { name: "Plot SVG" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Result JSON" })).toBeEnabled();
  await assertNoOverflow(page);
  expect(errors).toEqual([]);
});

test("exports contain reproducible data and a standalone figure", async ({
  page,
}) => {
  await openExplorer(page);
  await expect(
    page.getByRole("button", { name: "Methods and assumptions" }),
  ).toHaveAttribute("aria-expanded", "false");
  for (const [name, filename] of [
    ["Result JSON", "euv-diagnose-results.json"],
    ["Spectra CSV", "euv-diagnose-spectra.csv"],
    ["Plot SVG", "euv-diagnose-spectrum.svg"],
  ]) {
    const pending = page.waitForEvent("download");
    await page.getByRole("button", { name }).click();
    const download = await pending;
    expect(download.suggestedFilename()).toBe(filename);
    const content = await readFile((await download.path())!, "utf8");
    if (filename.endsWith(".json")) {
      const bundle = JSON.parse(content);
      expect(bundle.schemaVersion).toBe(1);
      expect(bundle.synthetic.nominal.spectra.length).toBe(
        bundle.synthetic.angles.length,
      );
      expect(bundle.measured.heldoutAngles).toContain(8);
    } else if (filename.endsWith(".csv")) {
      expect(content).toMatch(
        /^wavelength_nm,angle_from_normal_deg,channel,stack_a,stack_b\n/,
      );
      expect(content.split("\n").length).toBeGreaterThan(100);
      expect(content).not.toMatch(/NaN|undefined|Infinity/);
    } else {
      expect(content).toContain('xmlns="http://www.w3.org/2000/svg"');
      expect(content).toContain("<path");
      expect(content).toContain("Wavelength (nm)");
    }
    await expect(page.getByRole("status")).toHaveText(`Downloaded ${filename}`);
  }
});

test("keyboard navigation works with reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openExplorer(page);
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("link", { name: "Skip to results" }),
  ).toBeFocused();
  await page.keyboard.press("Enter");
  await page.getByRole("button", { name: "Measured spectra" }).focus();
  await page.keyboard.press("Enter");
  await expect(
    page.getByRole("button", { name: "Measured spectra" }),
  ).toBeFocused();
  await page.getByRole("slider", { name: "Incidence angle" }).focus();
  await page.keyboard.press("Home");
  await expect(page.locator("#angle-value")).toHaveText("2°");
  await expect(
    page.getByRole("slider", { name: "Incidence angle" }),
  ).toBeFocused();
  await assertNoOverflow(page);
});

test("a missing result bundle presents a useful recovery state", async ({
  page,
}) => {
  await page.route("**/data/demo.json", (route) =>
    route.fulfill({ status: 404, body: "Missing fixture" }),
  );
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Could not load results." }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
  await assertNoOverflow(page);
});

test("dragging the incidence slider reaches the intended angle", async ({
  page,
}) => {
  await openExplorer(page);
  const slider = page.getByRole("slider", { name: "Incidence angle" });
  await slider.scrollIntoViewIfNeeded();
  const box = (await slider.boundingBox())!;
  await page.mouse.move(
    box.x + 8 + ((box.width - 16) * 4) / 28,
    box.y + box.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 2, box.y + box.height / 2, {
    steps: 20,
  });
  await page.mouse.up();
  await expect(page.locator("#angle-value")).toHaveText("30°");
  await expect(
    page.getByRole("img", {
      name: "Multilayer reflectivity for two synthetic stacks at 30 degrees",
      exact: true,
    }),
  ).toBeVisible();
});

test("measured model comparison updates both phase and exported predictions", async ({
  page,
}) => {
  await openExplorer(page);
  await page.getByRole("button", { name: "Measured spectra" }).click();
  const phase = await page.locator(".model-phase strong").innerText();
  await page.getByLabel("Model assumptions").selectOption("1");
  expect(await page.locator(".model-phase strong").innerText()).not.toEqual(
    phase,
  );
  await expect(page.getByLabel("Model assumptions")).toBeFocused();
  await page.getByRole("button", { name: "Methods and assumptions" }).click();
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "Spectra CSV" }).click();
  const content = await readFile((await (await pending).path())!, "utf8");
  const bundle = await (await page.request.get("/data/demo.json")).json();
  const firstRow = content.trim().split("\n")[1].split(",");
  expect(Number(firstRow[4])).toBe(
    bundle.measured.comparisons[1].predicted[0][0][0],
  );
});

test("all exhibits pass automated WCAG A/AA checks", async ({ page }) => {
  await openExplorer(page);
  for (const tab of [
    "Phase ambiguity",
    "Measured spectra",
    "Learned inference",
  ]) {
    await page.getByRole("button", { name: tab }).click();
    const result = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    const violations = result.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => ({
        target: n.target,
        reason: n.failureSummary,
      })),
    }));
    expect(violations, tab).toEqual([]);
  }
});
