"""Gera o PDF do roteiro e os cards de perguntas dos convidados."""
import functools
import http.server
import pathlib
import threading
from playwright.sync_api import sync_playwright

here = pathlib.Path(__file__).parent.resolve()
out = here / "final"
out.mkdir(exist_ok=True)

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(here))
handler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}"

with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')

    pg = b.new_page()
    pg.goto(f"{base}/roteiro.html")
    pg.evaluate("document.fonts.ready")
    pg.wait_for_timeout(500)
    pg.pdf(path=str(out / "roteiro-painel-crescer-sem-se-perder.pdf"), format="A4", print_background=True,
           display_header_footer=True, header_template="<span></span>",
           footer_template="<div style='width:100%;text-align:center;font-size:8px;color:#888'>"
                           "<span class='pageNumber'></span> / <span class='totalPages'></span></div>",
           margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
    pg.close()

    pg = b.new_page(viewport={"width": 1080, "height": 1920}, device_scale_factor=2)
    for n, nome in ((1, "nathallie"), (2, "adriano"), (3, "gean")):
        pg.goto(f"{base}/perguntas.html?p={n}")
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(500)
        pg.locator("#art").screenshot(path=str(out / f"perguntas-{n}-{nome}.png"))
    pg.close()
    b.close()
srv.shutdown()
