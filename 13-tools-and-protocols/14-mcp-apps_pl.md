# MCP Apps — Interaktywne Zasoby UI przez `ui://`

> Tekstowe wyjście narzędzi ogranicza to, co agenci mogą pokazać. MCP Apps (SEP-1724, oficjalny 26 stycznia 2026) pozwala narzędziu zwrócić izolowane interaktywne HTML renderowane wbudowanie w Claude Desktop, ChatGPT, Cursor, Goose i VS Code. Dashboardy, formularze, mapy, sceny 3D — wszystko przez jedno rozszerzenie. Ta lekcja omawia schemat zasobu `ui://`, typ MIME `text/html;profile=mcp-app`, protokół postMessage w iframe-sandbox oraz powierzchnię bezpieczeństwa, która pojawia się, gdy serwer może renderować HTML.

**Type:** Build
**Languages:** Python (stdlib, UI resource emitter), HTML (sample app)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## Learning Objectives

- Zwrócić zasób `ui://` z wywołania narzędzia i ustawić poprawny typ MIME i metadane.
- Zadeklarować powiązane UI narzędzia przez `_meta.ui.resourceUri`, `_meta.ui.csp` i `_meta.ui.permissions`.
- Zaimplementować postMessage JSON-RPC w iframe sandbox dla komunikacji UI-host.
- Zastosować domyślne CSP i permissions-policy, które bronią przed atakami pochodzącymi z UI.

## Problem

Narzędzie `visualize_timeline` z ery 2025 może zwrócić "Oto 14 notatek ułożonych chronologicznie: ...". To jest akapit. Użytkownicy faktycznie chcą interaktywnej osi czasu. Przed MCP Apps opcjami były: API widgetów specyficzne dla klienta (Claude artifacts, OpenAI Custom GPT HTML) albo brak UI w ogóle.

MCP Apps (SEP-1724, dostarczony 26 stycznia 2026) standaryzuje kontrakt. Wynik narzędzia zawiera `resource`, którego URI to `ui://...`, a MIME to `text/html;profile=mcp-app`. Host renderuje go w izolowanym iframe z ograniczonym CSP i bez dostępu do sieci, chyba że wyraźnie dozwolone. UI wewnątrz iframe wysyła wiadomości do hosta przez mały dialekt postMessage JSON-RPC.

Każdy kompatybilny klient (Claude Desktop, ChatGPT, Goose, VS Code) renderuje ten sam zasób `ui://` w ten sam sposób. Jeden serwer, jeden pakiet HTML, uniwersalne UI.

## Koncepcja

### Schemat zasobu `ui://`

Narzędzie zwraca:

```json
{
  "content": [
    {"type": "text", "text": "Oto twoja oś czasu notatek:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

Host następnie wywołuje `resources/read` na URI `ui://notes/timeline` i otrzymuje:

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Piaskownica iframe

Host renderuje HTML wewnątrz izolowanego `<iframe>` z:

- `sandbox="allow-scripts allow-same-origin"` (lub bardziej restrykcyjne zgodnie z deklaracją serwera)
- CSP zadeklarowany przez serwer zastosowany przez nagłówki odpowiedzi.
- Brak ciasteczek, brak localStorage z pochodzenia hosta.
- Dostęp do sieci ograniczony do `connectSrc` w CSP.

### Protokół postMessage

Iframe komunikuje się z hostem przez `window.postMessage`. Mały dialekt JSON-RPC 2.0:

Zawsze przypinaj `targetOrigin` do dokładnego pochodzenia peera, a po stronie odbiorczej waliduj `event.origin` względem listy dozwolonych przed przetworzeniem jakiegokolwiek ładunku. Nigdy nie używaj `"*"` po żadnej stronie tego kanału — treść niesie wywołania narzędzi i odczyty zasobów.

```js
// iframe do hosta (przypnij do pochodzenia hosta)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host do iframe (przypnij do pochodzenia iframe)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// odbiorca po obu stronach
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // bezpiecznie przetwarzać event.data
});
```

Dostępne metody po stronie hosta, które UI może wywołać:

- `host.callTool(name, arguments)` — wywołuje narzędzie serwera.
- `host.readResource(uri)` — odczytuje zasób MCP.
- `host.getPrompt(name, arguments)` — pobiera szablon promptu.
- `host.close()` — zamyka UI.

Każde wywołanie wciąż przechodzi przez protokół MCP i dziedziczy uprawnienia serwera.

### Uprawnienia

Lista `_meta.ui.permissions` żąda dodatkowych możliwości:

- `camera` — dostęp do kamery użytkownika (używane w UI do skanowania dokumentów).
- `microphone` — wejście głosowe.
- `geolocation` — lokalizacja.
- `network:*` — szerszy dostęp do sieci niż pozwala sam `connectSrc`.

Każde uprawnienie to prompt, który użytkownik widzi przed renderowaniem UI.

### Zagrożenia bezpieczeństwa

HTML w iframe to wciąż HTML. Nowa powierzchnia ataku:

- **Wstrzyknięcie promptu przez UI.** Złośliwe UI serwera może pokazać tekst wyglądający jak wiadomość systemowa i oszukać użytkownika. Renderowanie hosta powinno wizualnie odróżniać UI serwera od UI hosta.
- **Eksfiltracja przez `connectSrc`.** Jeśli CSP zezwala na `connect-src: *`, UI może wysyłać dane gdziekolwiek. Domyślnie powinno być restrykcyjne.
- **Clickjacking.** UI nakłada się na chrom hosta. Hosty muszą zapobiegać manipulacji z-index i egzekwować reguły przezroczystości.
- **Kradzież fokusu.** UI przejmuje fokus klawiatury i przechwytuje następną wiadomość. Hosty muszą przechwytywać.

Phase 13 · 15 omawia je szczegółowo jako część bezpieczeństwa MCP; ta lekcja wprowadza je.

### Uzgadnianie `ui/initialize`

Po załadowaniu iframe wysyła `ui/initialize` przez postMessage:

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

Host odpowiada z możliwościami i tokenem sesji. UI używa tokena sesji przy każdym kolejnym wywołaniu hosta.

### Prymitywy SDK AppRenderer / AppFrame

SDK ext-apps udostępnia dwa wygodne prymitywy:

- `AppRenderer` (strona serwera) — opakowuje komponent React / Vue / Solid i emituje zasób `ui://` z odpowiednim MIME i metadanymi.
- `AppFrame` (strona klienta) — odbiera zasób, montuje iframe i pośredniczy w postMessage.

Możesz ich użyć lub samodzielnie napisać HTML i JSON-RPC.

### Stan ekosystemu

MCP Apps dostarczono 26 stycznia 2026. Obsługa klientów na kwiecień 2026:

- **Claude Desktop.** Pełna obsługa od stycznia 2026.
- **ChatGPT.** Pełna obsługa przez Apps SDK (ten sam bazowy protokół MCP Apps).
- **Cursor.** Beta; włącz przez ustawienia.
- **VS Code.** Tylko wersje Insider.
- **Goose.** Pełna obsługa.
- **Zed, Windsurf.** W planach.

Serwery w produkcji: dashboardy, wizualizacje map, tabele danych, kreatory wykresów, podglądy IDE w piaskownicy.

## Użyj

`code/main.py` rozszerza serwer notatek o narzędzie `visualize_timeline`, które zwraca zasób `ui://notes/timeline`, plus handler dla `resources/read` na tym URI, który zwraca mały, ale kompletny pakiet HTML z osią czasu SVG. HTML jest templatowany przez stdlib — bez systemu budowania. postMessage jest naszkicowane w komentarzach JS, ponieważ stdlib nie może sterować przeglądarką.

Na co zwrócić uwagę:

- `_meta.ui` w odpowiedzi narzędzia niesie resourceUri, CSP, uprawnienia.
- HTML renderuje się bez dostępu do sieci; wszystkie dane są wbudowane.
- JS wywołuje `host.callTool` przez `window.parent.postMessage` (udokumentowane, ale nieaktywne w tym demo stdlib).

## Ship It

Ta lekcja produkuje `outputs/skill-mcp-apps-spec.md`. Dla narzędzia, które skorzystałoby z interaktywnego UI, umiejętność produkuje pełny kontrakt MCP Apps: URI `ui://`, CSP, uprawnienia, punkty wejścia postMessage i listę kontrolną bezpieczeństwa.

## Ćwiczenia

1. Uruchom `code/main.py` i sprawdź wyemitowany HTML. Otwórz HTML bezpośrednio w przeglądarce; sprawdź, czy SVG renderuje się. Następnie naszkicuj kontrakt postMessage, którego UI użyłoby do wywołania `host.callTool("notes_update", ...)`.

2. Zaostrz CSP: usuń `'unsafe-inline'` i użyj polityki skryptów opartej na nonce. Co zmienia się w kodzie generowania HTML?

3. Dodaj drugi zasób UI `ui://notes/editor` z formularzem do edycji notatki na miejscu. Gdy użytkownik prześle, iframe wywołuje `host.callTool("notes_update", ...)`.

4. Przeprowadź audyt powierzchni ataku UI. Gdzie złośliwy serwer mógłby wstrzyknąć treść? Przed czym broni piaskownica iframe, a przed czym nie?

5. Przeczytaj specyfikację SEP-1724 i zidentyfikuj jedną możliwość w SDK MCP Apps, której ta zabawkowa implementacja nie używa. (Podpowiedź: synchronizacja stanu na poziomie komponentu.)

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| MCP Apps | "Interaktywne zasoby UI" | Rozszerzenie SEP-1724 dostarczone 2026-01-26 |
| `ui://` | "Schemat URI aplikacji" | Schemat zasobu dla pakietów UI |
| `text/html;profile=mcp-app` | "Typ MIME" | Content-type dla HTML aplikacji MCP |
| Piaskownica iframe | "Kontener renderowania" | Izolacja przeglądarkowa UI z CSP i uprawnieniami |
| postMessage JSON-RPC | "Połączenie UI-host" | Mały dialekt JSON-RPC przez postMessage dla wywołań hosta |
| `_meta.ui` | "Powiązanie narzędzie-UI" | Metadane łączące wynik narzędzia z zasobem UI |
| CSP | "Content-Security-Policy" | Deklaruje dozwolone źródła dla skryptów, sieci, stylów |
| AppRenderer | "Prymityw SDK serwera" | Konwertuje komponent frameworka na zasób `ui://` |
| AppFrame | "Prymityw SDK klienta" | Pomocnik montowania iframe, który pośredniczy w postMessage |
| `ui/initialize` | "Uzgadnianie" | Pierwsza wiadomość postMessage z UI do hosta |

## Dalsze czytanie

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — implementacja referencyjna i SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — formalny dokument specyfikacji
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — dokumentacja wysokiego poziomu
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — wpis o uruchomieniu ze stycznia 2026
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — referencja SDK w stylu JSDoc