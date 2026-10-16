"""Gera os 3 cartões da apresentadora (A6) e a folha A4 com os três para imprimir e recortar."""
import functools
import http.server
import pathlib
import threading
from playwright.sync_api import sync_playwright

here = pathlib.Path(__file__).parent.resolve()
out = here / "final" / "cartoes-manu"
out.mkdir(parents=True, exist_ok=True)

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(here))
handler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}/cartoes.html"

nomes = {1: "abertura-bloco-1", 2: "bloco-2", 3: "bloco-3"}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page()
    for n, nome in nomes.items():
        pg.goto(f"{base}?c={n}")
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(400)
        pg.pdf(path=str(out / f"cartao-{n}-{nome}-A6.pdf"), width="105mm", height="148mm",
               print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    pg.goto(base)
    pg.evaluate("document.fonts.ready")
    pg.wait_for_timeout(400)
    pg.pdf(path=str(out / "cartoes-manu-folha-A4-para-imprimir.pdf"), format="A4", print_background=True,
           margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    pg.close()
    b.close()
srv.shutdown()
