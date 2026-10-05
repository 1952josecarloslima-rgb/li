import functools
import http.server
import pathlib
import threading
from playwright.sync_api import sync_playwright

here = pathlib.Path(__file__).parent.resolve()
out = here / "final"
out.mkdir(exist_ok=True)

temas = {"a": "azul"}
formatos = {  # nome: (parametro, largura, altura)
    "story": ("v", 1080, 1920),      # exporta 2160×3840
    "projecao": ("h", 1920, 1080),   # exporta 3840×2160 (4K 16:9)
}

# servidor local: a máscara de textura (mask-image) não carrega via file://
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(here))
handler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}/arte.html"

with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    for fmt, (param, w, h) in formatos.items():
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        for tema, nome in temas.items():
            pg.goto(f"{base}?tema={tema}&formato={param}")
            pg.evaluate("document.fonts.ready")
            pg.wait_for_timeout(500)
            pg.locator("#art").screenshot(path=str(out / f"crescer-sem-se-perder-{nome}-{fmt}.png"))
        pg.close()
    # série de suspense (stories)
    pg = b.new_page(viewport={"width": 1080, "height": 1920}, device_scale_factor=2)
    for n in (1, 2, 3):
        pg.goto(base.replace("arte.html", "teaser.html") + f"?n={n}")
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(500)
        pg.locator("#art").screenshot(path=str(out / f"suspense-{n}-story.png"))
    pg.close()
    b.close()
srv.shutdown()
