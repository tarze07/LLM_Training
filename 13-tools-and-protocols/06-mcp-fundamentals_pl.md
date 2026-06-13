# Podstawy MCP — Prymitywy, Cykl Życia, Baza JSON-RPC

> Każda integracja przed MCP była jednorazowym rozwiązaniem. Model Context Protocol, wydany po raz pierwszy przez Anthropic w listopadzie 2024 roku, a obecnie zarządzany przez Agentic AI Foundation przy Linux Foundation, standaryzuje odkrywanie i wywoływanie, aby każdy klient mógł komunikować się z każdym serwerem. Specyfikacja z 2025-11-25 wymienia sześć prymitywów (trzy serwerowe, trzy klienckie), trzyfazowy cykl życia oraz format przesyłania JSON-RPC 2.0. Naucz się ich, a reszta rozdziału MCP w tej fazie stanie się jedynie lekturą.

**Type:** Learn
**Languages:** Python (stdlib, JSON-RPC parser)
**Prerequisites:** Phase 13 · 01 through 05 (the tool interface and function calling)
**Time:** ~45 minutes

## Cele Lekcji

- Wymienić wszystkie sześć prymitywów MCP (tools, resources, prompts po stronie serwera; roots, sampling, elicitation po stronie klienta) i podać po jednym przypadku użycia każdego z nich.
- Przejść przez trzyfazowy cykl życia (initialize, operation, shutdown) i określić, kto wysyła którą wiadomość w każdej fazie.
- Parsować i emitować koperty JSON-RPC 2.0: żądanie, odpowiedź i notyfikację.
- Wyjaśnić, czym jest negocjacja możliwości (capability negotiation) podczas `initialize` i co się psuje bez niej.

## Problem

Przed MCP każdy agent używający narzędzi miał swój własny protokół. Cursor miał system narzędzi w kształcie MCP, ale niekompatybilny. Claude Desktop dostarczono z innym. Rozszerzenie Copilot w VS Code miało trzecie. Zespół, który stworzył narzędzie "Postgres query", pisał to samo narzędzie trzy razy, każde do innego API hosta. Ponowne użycie wymagało kopiowania kodu.

Rezultatem był kambryjski wybuch jednorazowych integracji i ograniczenie szybkości rozwoju ekosystemu.

MCP rozwiązuje to poprzez standaryzację formatu przesyłania. Pojedynczy serwer MCP działa w każdym kliencie MCP: Claude Desktop, ChatGPT, Cursor, VS Code, Gemini, Goose, Zed, Windsurf — 300+ klientów do kwietnia 2026. 110 milionów miesięcznych pobrań SDK. Ponad 10 000 publicznych serwerów. Linux Foundation przejęła zarządzanie w grudniu 2025 roku w ramach nowej Agentic AI Foundation.

Rewizja specyfikacji używana w tej fazie to **2025-11-25**. Dodaje ona zadania asynchroniczne (SEP-1686), elicytację w trybie URL (SEP-1036), sampling z narzędziami (SEP-1577), inkrementalną zgodę zakresu (SEP-835) oraz semantykę wskaźnika zasobu OAuth 2.1 (SEP-1724). Faza 13 · 09 do 16 omawiają te rozszerzenia. Ta lekcja kończy się na podstawach.

## Koncepcja

### Trzy prymitywy serwerowe

1. **Tools (narzędzia).** Akcje wywoływalne. Ta sama czteroetapowa pętla co w Fazie 13 · 01.
2. **Resources (zasoby).** Udostępnione dane. Zawartość tylko do odczytu, adresowalna przez URI: `file:///path`, `db://query/...`, własne schematy.
3. **Prompts (podpowiedzi).** Szablony wielokrotnego użytku. Komendy z ukośnikiem w interfejsie hosta; serwer dostarcza szablon, klient wypełnia argumenty.

### Trzy prymitywy klienckie

4. **Roots (korzenie).** Zbiór URI, których serwer może dotykać. Klient deklaruje je; serwer respektuje je.
5. **Sampling (próbkowanie).** Serwer prosi model klienta o wykonanie dopełnienia (completion). Umożliwia pętle agentowe hostowane po stronie serwera bez kluczy API po stronie serwera.
6. **Elicitation (elicytacja).** Serwer prosi użytkownika klienta o strukturalne dane wejściowe w trakcie działania. Formularze lub URL-e (SEP-1036).

Każda możliwość w MCP należy dokładnie do jednego z tych sześciu. Faza 13 · 10 do 14 omawia każdy z nich szczegółowo.

### Format przesyłania: JSON-RPC 2.0

Każda wiadomość to obiekt JSON z następującymi polami:

- Żądania: `{jsonrpc: "2.0", id, method, params}`.
- Odpowiedzi: `{jsonrpc: "2.0", id, result | error}`.
- Notyfikacje: `{jsonrpc: "2.0", method, params}` — bez `id`, brak oczekiwanej odpowiedzi.

Podstawowa specyfikacja ma ~15 metod, pogrupowanych według prymitywu. Najważniejsze:

- `initialize` / `initialized` (uzgadnianie)
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage` (serwer do klienta)
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### Trzyfazowy cykl życia

**Faza 1: initialize.**

Klient wysyła `initialize` ze swoimi `capabilities` i `clientInfo`. Serwer odpowiada własnymi `capabilities`, `serverInfo` oraz wersją specyfikacji, którą obsługuje. Klient wysyła `notifications/initialized`, gdy przetrawi odpowiedź. Od tego momentu każda ze stron może wysyłać żądania zgodnie z wynegocjowanymi możliwościami.

**Faza 2: operation.**

Dwukierunkowa. Klient wywołuje `tools/list` w celu odkrycia, a następnie `tools/call` w celu wywołania. Serwer może wysłać `sampling/createMessage`, jeśli zadeklarował tę możliwość. Serwer może wysłać `notifications/tools/list_changed`, gdy jego zestaw narzędzi ulegnie zmianie. Klient może wysłać `notifications/roots/list_changed`, gdy użytkownik zmieni zakres korzeni.

**Faza 3: shutdown.**

Każda ze stron zamyka transport. W MCP nie ma metody strukturalnego zamknięcia; transport (stdio lub Streamable HTTP, Faza 13 · 09) przenosi sygnał zakończenia połączenia.

### Negocjacja możliwości

`capabilities` w uzgadnianiu `initialize` to kontrakt. Przykład z serwera:

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

Serwer deklaruje, że może emitować notyfikacje `tools/list_changed` i obsługuje `resources/subscribe`. Klient zgadza się, deklarując własne:

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

Jeśli klient nie zadeklaruje `sampling`, serwer nie może wywołać `sampling/createMessage`. Symetrycznie: jeśli serwer nie zadeklaruje `resources.subscribe`, klient nie może próbować subskrypcji.

To właśnie zapobiega dryfowi ekosystemu. Klient, który nie obsługuje próbkowania, nadal jest poprawnym klientem MCP; serwer, który nie wywołuje `sampling`, nadal jest poprawnym serwerem MCP. Po prostu nie używają tej funkcji razem.

### Strukturalna zawartość i kształty błędów

`tools/call` zwraca tablicę `content` z typowanymi blokami: `text`, `image`, `resource`. Faza 13 · 14 dodaje do tej listy MCP Apps (`ui://` interaktywny interfejs).

Błędy używają kodów błędów JSON-RPC. Dodatki zdefiniowane w specyfikacji: `-32002` "Resource not found", `-32603` "Internal error", plus dane błędu specyficzne dla MCP jako `error.data`.

### Możliwości klienta a szczegóły wywołania narzędzia

Częste nieporozumienie: `capabilities.tools` mówi, czy klient obsługuje notyfikacje o zmianie listy narzędzi. To, czy klient WYWOŁA konkretne narzędzia, jest decyzją czasu wykonania podejmowaną przez jego model, a nie flagą możliwości. Flaga możliwości to kontrakt na poziomie specyfikacji. Wybór modelu jest ortogonalny.

### Dlaczego JSON-RPC, a nie REST?

JSON-RPC 2.0 (2010) to lekki, dwukierunkowy protokół. REST jest inicjowany przez klienta. MCP potrzebowało wiadomości inicjowanych przez serwer (próbkowanie, notyfikacje), więc JSON-RPC z jego symetrycznym kształtem żądanie/odpowiedź był naturalnym dopasowaniem. JSON-RPC również komponuje się czysto na stdio i WebSocket/Streamable HTTP bez wymyślania na nowo kształtu żądania HTTP.

```figure
mcp-tool-call
```

## Użyj Tego

`code/main.py` dostarcza minimalistyczny parser i emiter JSON-RPC 2.0, a następnie ręcznie przechodzi sekwencję `initialize` → `tools/list` → `tools/call` → `shutdown`, wyświetlając każdą wiadomość. Żaden prawdziwy transport; tylko kształty wiadomości. Porównaj ze specyfikacją linkowaną w Further Reading, aby zweryfikować każdą kopertę.

Na co zwrócić uwagę:

- `initialize` deklaruje możliwości w obie strony; odpowiedź zawiera `serverInfo` i `protocolVersion: "2025-11-25"`.
- `tools/list` zwraca tablicę `tools`; każdy wpis ma `name`, `description`, `inputSchema`.
- `tools/call` używa `params.name` i `params.arguments`.
- Zawartość odpowiedzi `content` to tablica bloków `{type, text}`.

## Dostarcz

Ta lekcja produkuje `outputs/skill-mcp-handshake-tracer.md`. Mając transkrypt w stylu pcap interakcji klient-serwer MCP, umiejętność adnotuje każdą wiadomość: który prymityw, która faza cyklu życia i od której możliwości zależy.

## Ćwiczenia

1. Uruchom `code/main.py`. Zidentyfikuj linię, w której odbywa się negocjacja możliwości i opisz, co by się zmieniło, gdyby serwer nie zadeklarował `tools.listChanged`.

2. Rozszerz parser o obsługę `notifications/progress`. Kształt wiadomości: `{method: "notifications/progress", params: {progressToken, progress, total}}`. Wyemituj ją podczas długotrwałego `tools/call` i potwierdź, że handler klienta wyświetliłby pasek postępu.

3. Przeczytaj specyfikację MCP 2025-11-25 od deski do deski — cały dokument ma około 80 stron. Zidentyfikuj jedną flagę możliwości, której większość serwerów NIE potrzebuje. Podpowiedź: dotyczy subskrypcji zasobów.

4. Naszkicuj na papierze, do którego prymitywu należałaby hipotetyczna funkcja "cron job". (Podpowiedź: serwer chce, aby klient wywołał go o zaplanowanym czasie. Żaden z sześciu prymitywów obecnie nie pasuje.) Mapa drogowa MCP na 2026 rok ma wstępny SEP dla tego.

5. Sparsuj jeden dziennik sesji z otwartego serwera MCP na GitHubie. Policz wiadomości: żądania vs odpowiedzi vs notyfikacje. Oblicz, jaka część ruchu to cykl życia, a jaka to operacje.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| MCP | "Model Context Protocol" | Otwarty protokół do odkrywania i wywoływania model-narzędzie |
| Prymityw serwerowy | "Co serwer udostępnia" | tools (akcje), resources (dane), prompts (szablony) |
| Prymityw kliencki | "Co klient pozwala używać serwerom" | roots (zakres), sampling (wywołania zwrotne LLM), elicitation (dane od użytkownika) |
| JSON-RPC 2.0 | "Format przesyłania" | Symetryczne koperty żądanie/odpowiedź/notyfikacja |
| Uzgadnianie `initialize` | "Negocjacja możliwości" | Pierwsza para wiadomości; serwery i klienci deklarują obsługiwane funkcje |
| `tools/list` | "Odkrywanie" | Klient pyta serwer o jego bieżący zestaw narzędzi |
| `tools/call` | "Wywoływanie" | Klient prosi serwer o wykonanie narzędzia z argumentami |
| `notifications/*_changed` | "Zdarzenia mutacji" | Serwer informuje klienta, że jego lista prymitywów się zmieniła |
| Blok zawartości | "Typowany wynik" | `{type: "text" \| "image" \| "resource" \| "ui_resource"}` w wyniku narzędzia |
| SEP | "Spec Evolution Proposal" | Nazwana propozycja projektu (np. SEP-1686 dla zadań asynchronicznych) |

## Dalsza Lektura

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — kanoniczny dokument specyfikacji
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — model mentalny sześciu prymitywów
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — post premierowy z listopada 2024
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — roczne podsumowanie i zmiany w specyfikacji 2025-11-25
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — podsumowanie SEP-1686, 1036, 1577, 835 i 1724
