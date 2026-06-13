# Budowa Serwera MCP — SDK Python + TypeScript

> Większość tutoriali MCP pokazuje tylko stdio hello-world. Prawdziwy serwer udostępnia narzędzia plus zasoby plus podpowiedzi, obsługuje negocjację możliwości, emituje strukturalne błędy i działa tak samo w różnych SDK. Ta lekcja buduje serwer notatek od początku do końca: stdlib stdio transport, dyspozytor JSON-RPC, trzy prymitywy serwerowe i styl czystych funkcji, który można przełożyć na Python SDK (FastMCP) lub TypeScript SDK, gdy przejdziesz na wyższy poziom.

**Type:** Build
**Languages:** Python (stdlib, stdio MCP server)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals)
**Time:** ~75 minutes

## Cele Lekcji

- Zaimplementować metody `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list` i `prompts/get`.
- Napisać pętlę dyspozytora, która czyta wiadomości JSON-RPC ze stdin i zapisuje odpowiedzi na stdout.
- Emitować strukturalne odpowiedzi błędów zgodnie ze specyfikacją JSON-RPC 2.0 i dodatkowymi kodami MCP.
- Przenieść implementację stdlib do FastMCP (Python SDK) lub TypeScript SDK bez przepisywania logiki narzędzi.

## Problem

Zanim będziesz mógł użyć zdalnego transportu (Faza 13 · 09) lub warstwy uwierzytelniania (Faza 13 · 16), potrzebujesz czystego lokalnego serwera. Lokalny oznacza stdio: serwer jest uruchamiany przez klienta jako proces potomny, wiadomości płyną przez stdin/stdout oddzielone znakiem nowej linii.

Specyfikacja 2025-11-25 nakazuje, aby wiadomości stdio były kodowane jako obiekty JSON z jawnym separatorem `\n`. Bez SSE; SSE było starym trybem zdalnym i jest usuwane w połowie 2026 roku (serwer MCP Atlassian Rovo wycofał go 30 czerwca 2026; Keboola 1 kwietnia 2026). Dla stdio, jeden obiekt JSON na linię to cały format przesyłania.

Serwer notatek jest dobrym kształtem, ponieważ ćwiczy wszystkie trzy prymitywy serwerowe. Narzędzia wykonują mutacje (`notes_create`). Zasoby udostępniają dane (`notes://{id}`). Podpowiedzi dostarczają szablony (`review_note`). Kształt tej lekcji uogólnia się na dowolną dziedzinę.

## Koncepcja

### Pętla dyspozytora

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

Trzy zasady:

- Nie wypisuj na stdout niczego, co nie jest kopertą JSON-RPC. Logi debugowania idą na stderr.
- Każde żądanie MUSI być dopasowane do odpowiedzi z tym samym `id`.
- Notyfikacje NIE MOGĄ być odpowiadane.

### Implementacja `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

Deklaruj tylko to, co obsługujesz. Klient polega na zestawie możliwości, aby blokować funkcje.

### Implementacja `tools/list` i `tools/call`

`tools/list` zwraca `{tools: [...]}`, gdzie każdy wpis ma `name`, `description`, `inputSchema`. `tools/call` przyjmuje `{name, arguments}` i zwraca `{content: [blocks], isError: bool}`.

Bloki zawartości są typowane. Najczęstsze:

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

Błędy narzędzi występują w dwóch kształtach. Błędy na poziomie protokołu (nieznana metoda, złe parametry) to błędy JSON-RPC. Błędy na poziomie narzędzia (poprawne wywołanie, ale narzędzie zawiodło) są zwracane jako `{content: [...], isError: true}`. To pozwala modelowi zobaczyć błąd w swoim kontekście.

### Implementacja zasobów

Zasoby są z założenia tylko do odczytu. `resources/list` zwraca manifest; `resources/read` zwraca zawartość. URI mogą mieć postać `file://...`, `http://...` lub własny schemat, np. `notes://`.

Gdy udostępniasz dane jako zasób zamiast narzędzia:

- Model ich nie "wywołuje"; klient może wstrzyknąć je do kontekstu na żądanie użytkownika.
- Subskrypcje pozwalają serwerowi wypychać aktualizacje, gdy zasób się zmieni (Faza 13 · 10).
- Faza 13 · 14 rozszerza to o `ui://` dla interaktywnych zasobów.

### Implementacja podpowiedzi

Podpowiedzi to szablony z nazwanymi argumentami. Host udostępnia je jako komendy z ukośnikiem. Podpowiedź `review_note` może przyjąć argument `note_id` i wyprodukować szablon wielo-message, który klient przekazuje swojemu modelowi.

### Subtelności transportu stdio

- JSON oddzielany znakiem nowej linii. Brak ramkowania z prefiksem długości.
- Nie buforuj. `sys.stdout.flush()` po każdym zapisie.
- Klient kontroluje czas życia. Gdy stdin się zamknie (EOF), zakończ czysto.
- Nie obsługuj SIGPIPE po cichu; zaloguj i zakończ.

### Adnotacje

Każde narzędzie może mieć `annotations` opisujące właściwości bezpieczeństwa:

- `readOnlyHint: true` — tylko do odczytu, bezpieczne do ponowienia.
- `destructiveHint: true` — nieodwracalne skutki uboczne; klient powinien potwierdzić.
- `idempotentHint: true` — te same dane wejściowe dają te same wyniki.
- `openWorldHint: true` — wchodzi w interakcje z systemami zewnętrznymi.

Klient używa ich do decyzji UX (okna dialogowe potwierdzenia, wskaźniki stanu) i routingu (Faza 13 · 17).

### Ścieżka awansu

Serwer stdlib w `code/main.py` ma około 180 linii. FastMCP (Python) sprowadza tę samą logikę do stylu dekoratorowego:

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK ma równoważny kształt. Ścieżka awansu jest typu "wrzuć i działa", gdy będziesz gotowy; koncepcje (możliwości, dyspozytor, bloki zawartości) są takie same.

## Użyj Tego

`code/main.py` to kompletny serwer notatek MCP przez stdio, tylko stdlib. Obsługuje `initialize`, `tools/list`, `tools/call` dla trzech narzędzi (`notes_list`, `notes_search`, `notes_create`), `resources/list` i `resources/read` dla każdej notatki oraz podpowiedź `review_note`. Możesz nim sterować, przesyłając wiadomości JSON-RPC:

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

Na co zwrócić uwagę:

- Dyspozytor to `dict[str, Callable]` indeksowany po nazwie metody.
- Każdy executor narzędzia zwraca listę bloków zawartości, a nie goły string.
- `isError: true` jest ustawiane, gdy executor rzuca wyjątkiem.

## Dostarcz

Ta lekcja produkuje `outputs/skill-mcp-server-scaffolder.md`. Mając dziedzinę (notatki, zgłoszenia, pliki, baza danych), umiejętność tworzy szkielet serwera MCP z odpowiednim podziałem na narzędzia / zasoby / podpowiedzi i ścieżką awansu SDK.

## Ćwiczenia

1. Uruchom `code/main.py` i steruj nim za pomocą ręcznie skonstruowanych wiadomości JSON-RPC. Wypróbuj `notes_create`, a następnie `resources/read`, aby pobrać nową notatkę.

2. Dodaj narzędzie `notes_delete` z `annotations: {destructiveHint: true}`. Sprawdź, że klient wyświetliłby okno dialogowe potwierdzenia (wymaga to prawdziwego hosta; Claude Desktop działa).

3. Zaimplementuj `resources/subscribe`, aby serwer wypychał `notifications/resources/updated` za każdym razem, gdy notatka zostanie zmodyfikowana. Dodaj zadanie keepalive.

4. Przenieś serwer do FastMCP. Plik Python powinien skurczyć się do poniżej 80 linii. Zachowanie na poziomie przesyłania musi być identyczne; zweryfikuj za pomocą tego samego harnessa testowego JSON-RPC.

5. Przeczytaj sekcję specyfikacji `server/tools` i zidentyfikuj jedno pole definicji narzędzia, które nie jest zaimplementowane w serwerze tej lekcji. (Podpowiedź: jest ich kilka; wybierz jedno i dodaj je.)

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Serwer MCP | "Rzecz, która udostępnia narzędzia" | Proces mówiący MCP JSON-RPC przez stdio lub HTTP |
| Transport stdio | "Model procesu potomnego" | Serwer jest uruchamiany przez klienta; komunikuje się przez stdin/stdout |
| Dyspozytor | "Router metod" | Mapa nazwy metody JSON-RPC do funkcji obsługującej |
| Blok zawartości | "Kawałek wyniku narzędzia" | Typowany element w tablicy `content` odpowiedzi narzędzia |
| `isError` | "Błąd na poziomie narzędzia" | Sygnalizuje, że narzędzie zawiodło; odróżnia od błędu JSON-RPC |
| Adnotacje | "Wskazówki bezpieczeństwa" | Flagi readOnly / destructive / idempotent / openWorld |
| FastMCP | "Python SDK" | Framework wyższego poziomu oparty na dekoratorach, na bazie protokołu MCP |
| URI zasobu | "Adresowalne dane" | `file://`, `db://` lub własny schemat identyfikujący zasób |
| Szablon podpowiedzi | "Skrót komendy z ukośnikiem" | Szablon dostarczony przez serwer z miejscami na argumenty dla interfejsów hosta |
| Deklaracja możliwości | "Przełącznik funkcji" | Flagi dla poszczególnych prymitywów deklarowane w `initialize` |

## Dalsza Lektura

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — referencyjna implementacja Python
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — równoległa implementacja TS
- [FastMCP — server framework](https://gofastmcp.com/) — API Python w stylu dekoratorowym dla serwerów MCP
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — tutorial od początku do końca używający dowolnego SDK
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — kompletna referencja dla wiadomości tools/*