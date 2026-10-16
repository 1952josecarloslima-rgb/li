"""Gera a régua gigante: cada trecho vira PNG (~200 dpi) e depois o PDF de 7 folhas A4 em tamanho real."""
import functools
import http.server
import pathlib
import threading
from playwright.sync_api import sync_playwright

here = pathlib.Path(__file__).parent.resolve()
out = here / "final"
(out / "regua").mkdir(parents=True, exist_ok=True)
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(here))
handler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}/regua.html"
PAGES = 7

with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={"width": 720, "height": 1021}, device_scale_factor=2.1)  # ~200 dpi
    for n in range(PAGES):
        pg.goto(f"{base}?seg={n}")
        pg.wait_for_selector("body[data-ready='1']")
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(300)
        pg.locator("#segsvg").screenshot(path=str(out / "regua" / f"trecho-{n + 1}.png"))
        print("trecho", n + 1, flush=True)
    pg.close()
    pg = b.new_page()
    pg.goto(base)
    pg.wait_for_selector("body[data-ready='1']")
    pg.wait_for_timeout(800)
    pg.pdf(path=str(out / "regua-gigante-7-folhas-A4.pdf"), width="210mm", height="297mm",
           print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    b.close()
srv.shutdown()
