"""
Servidor de desenvolvimento.

Existe por um motivo só: `python -m http.server` não manda Cache-Control, e o
navegador então decide sozinho por heurística — o que na prática significa que
o HTML recarrega e os módulos ES importados por ele NÃO. Um `?v=2` na URL da
página invalida a página e nada mais, porque as URLs dos imports não mudaram.

O sintoma é cruel: você corrige um shader, recarrega, os testes rodam com o
código antigo e você passa a depurar uma correção que o navegador nunca
carregou. Aqui perdi uma rodada inteira de validação exatamente assim,
concluindo que o solver estava lento depois de já ter deixado de estar.

`no-store` em tudo. Um servidor de desenvolvimento não tem nada a ganhar com
cache.
"""

import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORTA = int(sys.argv[1]) if len(sys.argv) > 1 else 8600


class SemCache(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.js': 'text/javascript',
        '.mjs': 'text/javascript',
        '.wgsl': 'text/plain',
        '.glb': 'model/gltf-binary',
        '.gltf': 'model/gltf+json',
        '.stl': 'application/octet-stream',
    }

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, fmt, *args):
        """Silencia o log por requisição; um teste que carrega seis modelos de
        60 MB enche o terminal e esconde o que interessa."""
        if '"GET' in (fmt % args) and ' 200 ' in (fmt % args):
            return
        super().log_message(fmt, *args)


if __name__ == '__main__':
    with ThreadingHTTPServer(('127.0.0.1', PORTA), SemCache) as s:
        print(f'CFD2026 em http://127.0.0.1:{PORTA}/ (sem cache)')
        s.serve_forever()
