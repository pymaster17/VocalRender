"""Headless-browser smoke test for the piano roll.

Run the demo in UI-only mode first:
    VOCALRENDER_UI_ONLY=1 GRADIO_SERVER_PORT=7871 python app.py
then:
    python tests/browser/drive_piano_roll.py
Requires the dev group (playwright) and `playwright install chromium`.
"""

import os

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7871/"
os.makedirs("tests/browser/shots", exist_ok=True)
out = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    page.on("console", lambda m: errors.append(f"[{m.type}] {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
    page.goto(URL)
    page.wait_for_selector(".vr-roll", timeout=30000)
    page.wait_for_timeout(500)
    page.screenshot(path="tests/browser/shots/01_empty.png")
    out.append(("empty counts", page.text_content(".vr-counts")))

    page.get_by_role("button", name="随机预设").click()
    page.wait_for_selector(".vr-note", timeout=10000)
    page.wait_for_timeout(500)
    out.append(("preset notes/rests", (page.locator(".vr-note").count(), page.locator(".vr-rest").count())))
    out.append(("preset counts", page.text_content(".vr-counts")))
    out.append(("lyrics", page.locator("#vr-lyrics-row textarea, #vr-lyrics-row input").first.input_value()))
    page.screenshot(path="tests/browser/shots/02_preset.png")

    first = page.locator(".vr-note").first
    first.click()
    page.wait_for_timeout(100)
    before = page.locator(".vr-pitch").input_value()
    page.locator(".vr-roll").press("ArrowUp")
    page.wait_for_timeout(100)
    out.append(("arrow up pitch", (before, page.locator(".vr-pitch").input_value())))

    box = page.locator(".vr-note.selected").bounding_box()
    page.mouse.move(box["x"] + 5, box["y"] + 6)
    page.mouse.down()
    page.mouse.move(box["x"] + 5, box["y"] + 6 + 16 * 3, steps=5)
    page.mouse.up()
    page.wait_for_timeout(100)
    out.append(("after drag -3 pitch", page.locator(".vr-pitch").input_value()))

    handle = page.locator(".vr-note.selected .vr-handle")
    hb = handle.bounding_box()
    dur_before = page.locator(".vr-duration").input_value()
    page.mouse.move(hb["x"] + 3, hb["y"] + 6)
    page.mouse.down()
    page.mouse.move(hb["x"] + 3 + 60, hb["y"] + 6, steps=5)
    page.mouse.up()
    page.wait_for_timeout(100)
    out.append(("resize duration index", (dur_before, page.locator(".vr-duration").input_value())))

    second = page.locator(".vr-note").nth(1)
    second.dblclick()
    page.wait_for_selector(".vr-lyric-editor")
    page.locator(".vr-lyric-editor").fill("-")
    page.locator(".vr-lyric-editor").press("Enter")
    page.wait_for_timeout(100)
    out.append(("second note lyric", page.locator(".vr-note").nth(1).text_content()))
    out.append(("message", page.text_content(".vr-message")))

    page.locator(".vr-note").nth(2).click()
    n0 = page.locator(".vr-note").count()
    page.get_by_role("button", name="延音 Melisma").click()
    page.wait_for_timeout(100)
    n1 = page.locator(".vr-note").count()
    page.get_by_role("button", name="休止 Rest").click()
    page.wait_for_timeout(100)
    r1 = page.locator(".vr-rest").count()
    page.get_by_role("button", name="删除 Delete").click()
    page.wait_for_timeout(100)
    r2 = page.locator(".vr-rest").count()
    out.append(("melisma/rest/delete", (n0, n1, r1, r2)))

    page.locator(".vr-roll").press("Control+z")
    page.locator(".vr-roll").press("Control+z")
    page.wait_for_timeout(100)
    out.append(("after undo x2 notes/rests", (page.locator(".vr-note").count(), page.locator(".vr-rest").count())))

    page.get_by_role("button", name="铅笔 Draw").click()
    end = page.locator(".vr-end").bounding_box()
    wrap = page.locator(".vr-grid-wrap").bounding_box()
    page.mouse.click(end["x"] + 40, wrap["y"] + wrap["height"] / 2)
    page.wait_for_timeout(100)
    out.append(("after pencil notes", page.locator(".vr-note").count()))
    out.append(("pencil lyric", page.locator(".vr-note.selected").text_content()))
    page.screenshot(path="tests/browser/shots/03_edited.png")

    page.locator(".vr-bpm").fill("90")
    page.locator(".vr-bpm").press("Enter")
    page.locator(".vr-bpm").blur()
    page.wait_for_timeout(200)
    page.get_by_role("button", name="生成歌声").click()
    page.wait_for_timeout(3000)
    out.append(("status", page.locator("#vr-generate").locator("xpath=..").text_content().strip()[:200]))
    out.append(("debug prompt", page.get_by_label("Generated SVS prompt (debug)").input_value()[:220]))
    page.screenshot(path="tests/browser/shots/04_generated.png", full_page=True)

    page.locator("#vr-lyrics-row textarea, #vr-lyrics-row input").first.fill("我爱唱歌你好世界")
    page.get_by_role("button", name="应用歌词").click()
    page.wait_for_timeout(1500)
    out.append(("after apply lyrics", [page.locator(".vr-note").nth(i).text_content() for i in range(6)]))
    out.append(("apply message", page.locator("#piano-roll").locator("xpath=following-sibling::*[1]").text_content().strip()[:200]))
    browser.close()

for k, v in out:
    print(f"{k:28s} {v}")
print("\nconsole errors/warnings:")
print("\n".join(errors) or "(none)")
