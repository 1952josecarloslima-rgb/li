"""Gera as 4 telas do roteiro da Manu para celular (1080×1920 @2x)."""
import functools, http.server, pathlib, threading
from playwright.sync_api import sync_playwright
here = pathlib.Path(__file__).parent.resolve()
out = here / "final" / "roteiro-celular"; out.mkdir(parents=True, exist_ok=True)
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(here)); h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
nomes = {1: "abertura", 2: "bloco-1", 3: "bloco-2", 4: "bloco-3"}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={"width": 1080, "height": 2160}, device_scale_factor=2)
    for n, nome in nomes.items():
        pg.goto(f"http://127.0.0.1:{srv.server_port}/roteiro-celular.html?s={n}")
        pg.evaluate("document.fonts.ready"); pg.wait_for_timeout(400)
        pg.locator("#art").screenshot(path=str(out / f"roteiro-{n}-{nome}.png"))
        # checa se o conteúdo cabe na tela
        print(n, pg.evaluate("(() => { const s=document.querySelector('.scr:not([style*=none])'); const v=[...document.querySelectorAll('.scr')].find(e=>getComputedStyle(e).display!=='none'); return v.scrollHeight })()"))
    b.close()
srv.shutdown()
