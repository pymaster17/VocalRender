"""Headless-browser smoke test for the piano roll.

Run the demo in UI-only mode first:
    VOCALRENDER_UI_ONLY=1 GRADIO_SERVER_PORT=7871 python demo/app.py
then:
    python demo/tests/browser/drive_piano_roll.py
Requires the dev extra (playwright) and `playwright install chromium`.
"""

import os

from playwright.sync_api import sync_playwright, expect

URL = "http://127.0.0.1:7871/"
SHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
os.makedirs(SHOTS, exist_ok=True)
out = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    page.on("console", lambda m: errors.append(f"[{m.type}] {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
    page.goto(URL)
    page.wait_for_selector(".vr-roll", timeout=30000)
    page.wait_for_selector(".vr-note", timeout=15000)
    assert page.get_by_label("Generated SVS prompt (debug)").is_visible(), "Only run this test in UI-only mode"
    assert page.locator("#vr-generate").bounding_box()["y"] < 700
    page.screenshot(path=SHOTS + "/01_initial_example.png")
    out.append(("initial random example", page.text_content(".vr-counts")))

    page.get_by_role("button", name="换个随机示例").click()
    page.wait_for_selector(".vr-note", timeout=10000)
    page.wait_for_timeout(500)
    out.append(("preset notes/rests", (page.locator(".vr-note").count(), page.locator(".vr-rest").count())))
    out.append(("preset counts", page.text_content(".vr-counts")))
    out.append(("lyrics", page.locator("#vr-lyrics-row textarea, #vr-lyrics-row input").first.input_value()))
    page.screenshot(path=SHOTS + "/02_preset.png")

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
    page.get_by_role("button", name="一字多音").click()
    page.wait_for_timeout(100)
    n1 = page.locator(".vr-note").count()
    page.locator(".vr-more summary").click()
    page.get_by_role("button", name="+ 休止").click()
    page.wait_for_timeout(100)
    r1 = page.locator(".vr-rest").count()
    page.get_by_role("button", name="删除").click()
    page.wait_for_timeout(100)
    r2 = page.locator(".vr-rest").count()
    out.append(("melisma/rest/delete", (n0, n1, r1, r2)))

    page.locator(".vr-roll").press("Control+z")
    page.locator(".vr-roll").press("Control+z")
    page.wait_for_timeout(100)
    out.append(("after undo x2 notes/rests", (page.locator(".vr-note").count(), page.locator(".vr-rest").count())))

    page.get_by_role("button", name="画音符").click()
    page.locator(".vr-more summary").click()
    end = page.locator(".vr-end").bounding_box()
    wrap = page.locator(".vr-grid-wrap").bounding_box()
    page.mouse.click(end["x"] + 40, wrap["y"] + wrap["height"] / 2)
    page.wait_for_timeout(100)
    out.append(("after pencil notes", page.locator(".vr-note").count()))
    out.append(("pencil lyric", page.locator(".vr-note.selected").text_content()))
    page.screenshot(path=SHOTS + "/03_edited.png")

    page.locator(".vr-bpm").fill("90")
    page.locator(".vr-bpm").press("Enter")
    page.locator(".vr-bpm").blur()
    page.wait_for_timeout(200)
    page.get_by_role("button", name="生成歌声").click()
    page.wait_for_timeout(3000)
    out.append(("status", page.locator("#vr-generate").locator("xpath=..").text_content().strip()[:200]))
    out.append(("debug prompt", page.get_by_label("Generated SVS prompt (debug)").input_value()[:220]))
    page.screenshot(path=SHOTS + "/04_generated.png", full_page=True)

    # Changing language must preserve the score and undo state.
    notes_before = page.locator(".vr-note").all_text_contents()
    page.get_by_text("English", exact=True).click()
    expect(page.locator(".vr-play")).to_have_text("▶ Preview melody")
    assert page.locator(".vr-note").all_text_contents() == notes_before
    assert page.locator('[data-action="undo"]').is_enabled()
    page.locator('[data-action="play"]').click()
    expect(page.locator(".vr-play")).to_have_text("⏸ Pause")
    page.locator('[data-action="stop"]').click()

    # Short lyrics highlight missing slots, and applying them leaves notes intact.
    page.locator("#vr-lyrics-row textarea").fill("我")
    expect(page.locator("#vr-alignment")).to_contain_text("lyric slots")
    expect(page.locator(".lyric-missing").first).to_be_attached()
    page.get_by_role("button", name="Fill notes with lyrics").click()
    expect(page.get_by_text("Match the lyrics to the notes first", exact=False)).to_be_visible()
    assert page.locator(".vr-note").all_text_contents() == notes_before
    page.get_by_role("button", name="Random example", exact=True).click()
    expect(page.locator("#vr-alignment")).to_have_text("")
    expect(page.locator(".lyric-missing")).to_have_count(0)

    # Retain the score-import path in the compact sidebar.
    page.get_by_text("Import ABC / MusicXML", exact=True).click()
    page.get_by_label("ABC notation", exact=True).fill("X:1\nM:4/4\nL:1/4\nQ:1/4=90\nK:C\nC D E F |\nw: 我 爱 唱 歌")
    page.get_by_role("button", name="Parse score", exact=True).click()
    expect(page.get_by_text("Score parsed.", exact=False)).to_be_visible()
    page.get_by_role("button", name="Load into piano roll", exact=True).click()
    expect(page.locator(".vr-note")).to_have_count(4)
    assert page.locator(".vr-note").all_text_contents() == ["我", "爱", "唱", "歌"]
    page.screenshot(path=SHOTS + "/05_english_import.png", full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.screenshot(path=SHOTS + "/06_mobile.png", full_page=True)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), page.evaluate("({width:innerWidth, scroll:document.documentElement.scrollWidth})")
    browser.close()

for k, v in out:
    print(f"{k:28s} {v}")
print("\nconsole errors/warnings:")
print("\n".join(errors) or "(none)")
