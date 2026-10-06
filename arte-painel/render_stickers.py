"""Exporta cada sticker como PNG com fundo transparente + uma folha de prévia."""
import functools
import http.server
import pathlib
import threading
from playwright.sync_api import sync_playwright

here = pathlib.Path(__file__).parent.resolve()
out = here / "final" / "stickers"
out.mkdir(parents=True, exist_ok=True)

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(here))
handler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}/stickers.html"

with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=3)
    pg.goto(base)
    pg.evaluate("document.fonts.ready")
    pg.wait_for_timeout(600)
    ids = pg.eval_on_selector_all(".st", "els => els.map(e => e.id)")
    for i, sid in enumerate(ids, 1):
        pg.locator(f"#{sid}").screenshot(path=str(out / f"sticker-{i:02d}-{sid}.png"), omit_background=True)
    pg.close()

    pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    pg.goto(base + "?preview=1")
    pg.evaluate("document.fonts.ready")
    pg.wait_for_timeout(600)
    pg.screenshot(path=str(here / "final" / "stickers-previa.png"), full_page=True)
    pg.close()
    b.close()
srv.shutdown()
