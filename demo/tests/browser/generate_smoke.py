"""Browser end-to-end: load a preset, press Generate against a real model server, expect audio."""
import os
import sys

from playwright.sync_api import sync_playwright

URL = os.environ.get("VR_URL", "http://127.0.0.1:7871/")
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1100})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL)
    page.wait_for_selector(".vr-roll", timeout=60000)
    page.get_by_role("button", name="随机预设").click()
    page.wait_for_selector(".vr-note", timeout=10000)
    counts = page.text_content(".vr-counts")
    page.get_by_role("button", name="生成歌声").click()
    page.wait_for_function(
        "() => document.body.innerText.includes('Generated') || document.body.innerText.includes('❌')",
        timeout=300000,
    )
    status = page.locator("#vr-generate").locator("xpath=..").text_content().strip()
    shots = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
    os.makedirs(shots, exist_ok=True)
    page.screenshot(path=os.path.join(shots, "gpu_generate.png"), full_page=True)
    print("counts:", counts)
    print("status:", status)
    print("pageerrors:", [e for e in errors if "aborted" not in e])
    browser.close()
    sys.exit(0 if "Generated" in status else 1)
