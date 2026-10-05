import functools
import http.server
import pathlib
import threading
from playwright.sync_api import sync_playwright

here = pathlib.Path(__file__).parent.resolve()
temas = {"a": "crescer-sem-se-perder-A-azul.png", "c": "crescer-sem-se-perder-C-vinho.png"}

# servidor local: a máscara de textura (mask-image) não carrega via file://
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(here))
handler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}/arte.html"

with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={"width": 1080, "height": 1920}, device_scale_factor=2)
    for tema, nome in temas.items():
        pg.goto(f"{base}?tema={tema}")
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(500)
        pg.locator("#art").screenshot(path=str(here / nome))
    b.close()
srv.shutdown()
