# Asynchroniczne Zadania (SEP-1686) — Wywołaj Teraz, Pobierz Później dla Długotrwałej Pracy

> Rzeczywista praca agentów trwa od minut do godzin: uruchomienia CI, synteza głębokich badań, eksporty wsadowe. Synchroniczne wywołania narzędzi zrywają połączenia, przekraczają limity czasu lub blokują interfejs użytkownika. SEP-1686, scalony 2025-11-25, dodaje prymityw Zadań: każde żądanie może być rozszerzone, aby stać się zadaniem, a wynik może być pobrany później lub strumieniowany przez powiadomienia o stanie. Uwaga o ryzyku dryfu: Zadania są eksperymentalne do H1 2026; powierzchnia SDK wciąż jest projektowana wokół specyfikacji.

**Type:** Build
**Languages:** Python (stdlib, async task state machine)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 09 (transports)
**Time:** ~75 minutes

## Learning Objectives

- Określić, kiedy promować narzędzie z synchronicznego na rozszerzone o zadania (>30 sekund pracy po stronie serwera).
- Przejść cykl życia zadania: `working` → `input_required` → `completed` / `failed` / `cancelled`.
- Utrwalić stan zadania, aby awarie nie powodowały utraty pracy w toku.
- Poprawnie odpytywać `tasks/status` i pobierać `tasks/result`.

## Problem

Narzędzie `generate_report` uruchamia wielominutowy potok ekstrakcji. Opcje w modelu synchronicznym:

1. Utrzymywać połączenie otwarte przez trzy minuty. Zdalne transporty je zrywają; klienci przekraczają limity czasu; interfejsy się zawieszają.
2. Zwrócić natychmiast z symbolem zastępczym; wymagać od klienta odpytywania niestandardowego endpointu. Łamie jednolitość MCP.
3. Odpalić i zapomnieć; brak wyniku.

Żadne nie jest dobre. SEP-1686 dodaje czwartą: rozszerzenie o zadania. Każde żądanie (zazwyczaj `tools/call`) może być oznaczone jako zadanie. Serwer natychmiast zwraca identyfikator zadania. Klient odczytuje `tasks/status` i pobiera `tasks/result`, gdy gotowe. Stan po stronie serwera przetrwa restarty.

## Koncepcja

### Rozszerzenie o zadanie

Żądanie staje się zadaniem przez ustawienie `params._meta.task.required: true` (lub `optional: true`, serwer decyduje). Serwer odpowiada natychmiast:

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` to obietnica serwera zachowania stanu; po ttl wynik zadania jest odrzucany.

### Opcja na narzędzie

Adnotacje narzędzi mogą deklarować obsługę zadań:

- `taskSupport: "forbidden"` — to narzędzie zawsze działa synchronicznie. Bezpieczne dla szybkich narzędzi.
- `taskSupport: "optional"` — klient może zażądać rozszerzenia o zadanie.
- `taskSupport: "required"` — klient MUSI użyć rozszerzenia o zadanie.

Narzędzie `generate_report` byłoby `required`. Narzędzie `notes_search` byłoby `forbidden`.

### Stany

```
working  -> input_required -> working  (pętla przez elicytację)
working  -> completed
working  -> failed
working  -> cancelled
```

Maszyna stanów jest tylko-do-dopisywania: po `completed`, `failed` lub `cancelled` zadanie jest terminalne.

### Metody

- `tasks/status {taskId}` — zwraca bieżący stan i wskazówkę postępu.
- `tasks/result {taskId}` — blokuje lub zwraca 404, jeśli jeszcze nie gotowe.
- `tasks/cancel {taskId}` — idempotentne; stany terminalne ignorują.
- `tasks/list` — opcjonalne; wylicza aktywne i niedawno zakończone zadania.

### Strumieniowanie zmian stanu

Gdy serwer to obsługuje, klient może subskrybować powiadomienia o stanie:

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

Klienci, którzy strumieniują zamiast odpytywać, mają lepsze UX. Odpytywanie jest zawsze obsługiwane jako minimalna powierzchnia.

### Trwały stan

Specyfikacja wymaga, aby serwery deklarujące obsługę zadań utrwalały stan. Awaria nie powinna powodować utraty ukończonych wyników w ramach ttl. Magazyny obejmują od SQLite przez Redis po system plików. Harness Lekcji 13 używa systemu plików.

### Semantyka anulowania

`tasks/cancel` jest idempotentne. Jeśli zadanie jest w trakcie wykonywania, serwer próbuje zatrzymać (sprawdza kooperacyjne anulowanie wykonawcy). Jeśli już terminalne, żądanie jest bez operacji.

### Odzyskiwanie po awarii

Gdy proces serwera uruchamia się ponownie:

1. Załaduj wszystkie utrwalone stany zadań.
2. Oznacz wszystkie zadania `working`, których proces umarł, jako `failed` z błędem `CRASH_RECOVERY`.
3. Zachowaj `completed` / `failed` / `cancelled` na czas ich ttl.

### Asynchroniczne zadania plus sampling

Zadanie może samo wywołać `sampling/createMessage`. Tak działają długotrwałe zadania badawcze: wątek zadania serwera sampluje model klienta w razie potrzeby, podczas gdy UI klienta pokazuje zadanie jako `working` z okresowymi aktualizacjami postępu.

### Dlaczego to jest eksperymentalne

SEP-1686 został dostarczony w 2025-11-25, ale szersza mapa drogowa wymienia trzy otwarte kwestie: trwałe prymitywy subskrypcji, podzadania (relacje rodzic-dziecko między zadaniami) i standaryzację TTL wyniku. Oczekuj, że specyfikacja będzie ewoluować przez 2026. Kod produkcyjny powinien traktować Zadania jako stabilne tylko dla typowego przypadku i zabezpieczać się przed przyszłymi zmianami SDK dla podzadań.

## Użyj

`code/main.py` implementuje trwały magazyn zadań (na systemie plików) i narzędzie `generate_report`, które działa w wątku tła. Klienci wywołują narzędzie, natychmiast otrzymują identyfikator zadania, odczytują `tasks/status`, podczas gdy pracownik aktualizuje postęp, i pobierają `tasks/result`, gdy gotowe. Anulowanie działa; odzyskiwanie po awarii jest symulowane przez zabicie wątku pracownika i ponowne załadowanie stanu.

Na co zwrócić uwagę:

- Stan zadania JSON utrwalony w `/tmp/lesson-13-tasks/<id>.json`.
- Wątek pracownika aktualizuje pole `progress`; odczyt pokazuje jego postęp.
- Anulowanie po stronie klienta ustawia zdarzenie; pracownik sprawdza i kończy wcześniej.
- Ponowne załadowanie stanu po "awarii" oznacza zadanie w toku jako `failed` z `CRASH_RECOVERY`.

## Ship It

Ta lekcja produkuje `outputs/skill-task-store-designer.md`. Dla długotrwałego narzędzia (badania, budowanie, eksport), umiejętność projektuje magazyn zadań (kształt stanu, ttl, trwałość), wybiera odpowiednią flagę taskSupport i szkicuje powiadomienia o postępie.

## Ćwiczenia

1. Uruchom `code/main.py`. Rozpocznij zadanie `generate_report`, odczytaj stan, a następnie pobierz wynik.

2. Dodaj wywołanie `tasks/cancel` w trakcie działania. Sprawdź, czy pracownik honoruje je i stan zmienia się na `cancelled`.

3. Symuluj odzyskiwanie po awarii: zabij wątek pracownika, uruchom ponownie ładowarkę i zaobserwuj tryb awarii `CRASH_RECOVERY`.

4. Rozszerz magazyn na SQLite. Korzyści z trwałości są takie same; otwierają się opcje zapytań (lista wszystkich zadań z sesji X).

5. Przeczytaj wpis o mapie drogowej MCP na 2026. Zidentyfikuj jeden otwarty problem związany z Zadaniami, który najprawdopodobniej wpłynie na projekt API SDK w ciągu najbliższego roku.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Zadanie (Task) | "Długotrwałe wywołanie narzędzia" | Żądanie rozszerzone o `_meta.task` dla asynchronicznego wykonania |
| SEP-1686 | "Specyfikacja zadań" | Propozycja Ewolucji Specyfikacji, która dodała Zadania w 2025-11-25 |
| `_meta.task` | "Koperta zadania" | Metadane na żądanie zawierające id, stan, ttl |
| taskSupport | "Flaga narzędzia" | `forbidden` / `optional` / `required` na narzędzie |
| `tasks/status` | "Metoda odpytywania" | Pobierz bieżący stan i opcjonalną wskazówkę postępu |
| `tasks/result` | "Pobierz wynik" | Zwraca ukończony ładunek lub 404, jeśli jeszcze nie gotowe |
| `tasks/cancel` | "Zatrzymaj" | Idempotentne żądanie anulowania |
| ttl | "Budżet retencji" | Milisekundy, przez które serwer obiecuje przechowywać stan zadania |
| `notifications/tasks/updated` | "Push stanu" | Zdarzenie zmiany stanu inicjowane przez serwer |
| Trwały magazyn | "Stan odporny na awarie" | Warstwa trwałości: system plików / SQLite / Redis |

## Dalsze czytanie

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — oryginalna propozycja i pełna dyskusja
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — przewodnik projektowy z uzasadnieniem
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — mechanika i maszyna stanów
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — wzorce implementacji zadań na poziomie SDK
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — otwarte kwestie i priorytety 2026, w tym podzadania