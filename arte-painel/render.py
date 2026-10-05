import pathlib
from playwright.sync_api import sync_playwright

here = pathlib.Path(__file__).parent.resolve()
temas = {"a": "crescer-sem-se-perder-A-azul.png", "b": "crescer-sem-se-perder-B-papel.png"}

with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={"width": 1080, "height": 1920}, device_scale_factor=2)
    for tema, nome in temas.items():
        pg.goto((here / "arte.html").as_uri() + f"?tema={tema}")
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(500)
        pg.locator("#art").screenshot(path=str(here / nome))
    b.close()
