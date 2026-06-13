# Budowa Klienta MCP — Odkrywanie, Wywoływanie, Zarządzanie Sesją

> Większość treści o MCP dostarcza tutoriale serwerów i machnięciem ręki traktuje klienta. Kod klienta to miejsce, gdzie żyje trudna orkiestracja: uruchamianie procesów, negocjacja możliwości, scalanie list narzędzi z wielu serwerów, wywołania zwrotne próbkowania, ponowne łączenie i rozwiązywanie kolizji przestrzeni nazw. Ta lekcja buduje klienta wieloserwerowego, który podnosi trzy różne serwery MCP do jednej płaskiej przestrzeni nazw narzędzi dla modelu.

**Type:** Build
**Languages:** Python (stdlib, multi-server MCP client)
**Prerequisites:** Phase 13 · 07 (building an MCP server)
**Time:** ~75 minutes

## Cele Lekcji

- Uruchomić serwer MCP jako proces potomny, przeprowadzić `initialize` i wysłać `notifications/initialized`.
- Utrzymywać stan sesji dla każdego serwera (możliwości, lista narzędzi, ostatnio widziane identyfikatory notyfikacji).
- Scalić listy narzędzi z wielu serwerów w jedną przestrzeń nazw z obsługą kolizji.
- Przekierować wywołanie narzędzia do serwera, który jest jego właścicielem, i złożyć odpowiedź.

## Problem

Prawdziwy host agenta (Claude Desktop, Cursor, Goose, Gemini CLI) ładuje wiele serwerów MCP jednocześnie. Użytkownik może mieć uruchomiony jednocześnie serwer plików, serwer Postgres i serwer GitHub. Zadanie klienta:

1. Uruchomić każdy serwer.
2. Przeprowadzić uzgadnianie z każdym niezależnie.
3. Wywołać `tools/list` na każdym i spłaszczyć wynik.
4. Gdy model wyemituje `notes_search`, znaleźć je w scalonej przestrzeni nazw i przekierować do właściwego serwera.
5. Obsługiwać notyfikacje z dowolnego serwera (`tools/list_changed`) bez blokowania.
6. Ponownie połączyć się w przypadku awarii transportu.

Ręczne wykonanie tego wszystkiego oddziela "zabawkę" od "nadającego się do użytku". Oficjalne SDK opakowują to, ale model mentalny musi być twój.

## Koncepcja

### Uruchamianie procesu potomnego

`subprocess.Popen` z `stdin=PIPE, stdout=PIPE, stderr=PIPE`. Ustaw `bufsize=1` i użyj trybu tekstowego do odczytu linia po linii. Każdy serwer to jeden proces; klient trzyma jeden uchwyt `Popen` na serwer.

### Stan sesji dla każdego serwera

Obiekt `Session` na serwer przechowuje:

- `process` — uchwyt Popen.
- `capabilities` — co serwer zadeklarował przy `initialize`.
- `tools` — ostatni wynik `tools/list`.
- `pending` — mapa identyfikatora żądania do obietnicy/future czekającej na odpowiedź.

Żądania są z natury asynchroniczne; `tools/call` wysłane do serwera A, podczas gdy serwer B jest w trakcie wywołania, nie może blokować. Użyj wątków z kolejkami lub asyncio.

### Scalona przestrzeń nazw

Gdy klient widzi zagregowaną listę narzędzi, nazwy mogą kolidować. Dwa serwery mogą oba udostępniać `search`. Klient ma trzy opcje:

1. **Prefiks od nazwy serwera.** `notes/search`, `files/search`. Jasne, ale brzydkie.
2. **Ciche pierwszeństwo.** Późniejszy serwer `search` nadpisuje wcześniejszy. Ryzykowne; ukrywa kolizje.
3. **Odrzucenie kolizji.** Odmowa załadowania drugiego serwera; powiadomienie użytkownika. Najbezpieczniejsze dla hostów wrażliwych na bezpieczeństwo.

Claude Desktop używa prefiksu od serwera. Cursor używa odrzucenia kolizji z jasnym błędem. VS Code MCP również przyjmuje prefiks od serwera.

### Routing

Po scaleniu, tabela dyspozytorska mapuje `nazwa_narzędzia -> sesja`. Model emituje wywołanie po nazwie; klient znajduje sesję i zapisuje wiadomość `tools/call` na stdin tego serwera, a następnie czeka na odpowiedź.

### Wywołanie zwrotne próbkowania

Jeśli serwer zadeklarował możliwość `sampling` przy `initialize`, może wysłać `sampling/createMessage`, prosząc klienta o uruchomienie jego LLM. Klient musi:

1. Zablokować dalsze żądania do tego serwera, aż próbka zostanie rozstrzygnięta, lub potokować, jeśli jego implementacja obsługuje współbieżność.
2. Wywołać swojego dostawcę LLM.
3. Wysłać odpowiedź z powrotem do serwera.

Lekcja 11 omawia próbkowanie od początku do końca. Ta lekcja tworzy jego atrapę dla kompletności.

### Obsługa notyfikacji

`notifications/tools/list_changed` oznacza ponowne wywołanie `tools/list`. `notifications/resources/updated` oznacza ponowne odczytanie zasobu, jeśli jest używany. Notyfikacje nie mogą generować odpowiedzi — nie próbuj ich potwierdzać.

Częsty błąd klienta: blokowanie pętli odczytu na `tools/call`, podczas gdy notyfikacja czeka w strumieniu. Użyj wątku czytnika tła, który wypycha każdą wiadomość do kolejki; główny wątek usuwa z kolejki i dysponuje.

### Ponowne łączenie

Transport może ulec awarii: serwer się zawiesił, system operacyjny zabił proces, potok stdio się przerwał. Klient wykrywa EOF na stdout i traktuje sesję jako martwą. Opcje:

- Po cichu zrestartować serwer i ponownie przeprowadzić uzgadnianie. OK dla czysto odczytowych serwerów.
- Przekazać błąd użytkownikowi. OK dla serwerów stanowych z widocznymi dla użytkownika sesjami.

Faza 13 · 09 omawia semantykę ponownego łączenia Streamable HTTP; stdio jest prostsze.

### Keepalive i identyfikator sesji

Streamable HTTP używa nagłówka `Mcp-Session-Id`. Stdio nie ma identyfikatora sesji — tożsamość procesu JEST sesją. Pingi keepalive są opcjonalne; potoki stdio nie zrywają się przy braku aktywności.

## Użyj Tego

`code/main.py` uruchamia trzy symulowane serwery MCP jako procesy potomne, przeprowadza uzgadnianie z każdym, scala ich listy narzędzi i kieruje wywołania narzędzi do właściwego serwera. "Serwery" to w rzeczywistości inne procesy Python uruchamiające zabawkowe responderki (bez prawdziwego LLM). Uruchom, aby zobaczyć:

- Trzy inicjalizacje, każda z własnym zestawem możliwości.
- Trzy wyniki `tools/list` scalone w przestrzeń nazw z 7 narzędziami.
- Decyzję routingu opartą na nazwie narzędzia.
- Kolizję zapobieżoną przez prefiksowanie przestrzeni nazw.

Na co zwrócić uwagę:

- Klasa danych `Session` czysto przechowuje stan dla każdego serwera.
- Wątek czytnika tła usuwa z kolejki każdą linię z stdout bez blokowania głównego wątku.
- Tabela dyspozytorska to prosty `dict[str, Session]`.
- Obsługa kolizji jest jawna: gdy dwa serwery deklarują tę samą nazwę, późniejszy jest zmieniany z prefiksem.

## Dostarcz

Ta lekcja produkuje `outputs/skill-mcp-client-harness.md`. Mając deklaratywną listę serwerów MCP (nazwa, polecenie, argumenty), umiejętność produkuje harness, który je uruchamia, scala listy narzędzi i dostarcza funkcję routingu z rozwiązywaniem kolizji.

## Ćwiczenia

1. Uruchom `code/main.py` i obserwuj log uruchamiania serwera. Zabij jeden z symulowanych procesów serwera sygnałem SIGTERM i zaobserwuj, jak klient wykrywa EOF i oznacza tę sesję jako martwą.

2. Zaimplementuj prefiksowanie przestrzeni nazw. Gdy dwa serwery udostępniają `search`, zmień nazwę drugiego na `<serwer>/search`. Zaktualizuj tabelę dyspozytorską i sprawdź, czy wywołania narzędzi są poprawnie kierowane.

3. Dodaj backoff w stylu puli połączeń dla restartu serwera: wykładniczy backoff przy kolejnych awariach, limit 30 sekund, wyemituj powiadomienie dla użytkownika po trzech awariach.

4. Naszkicuj klienta obsługującego 100 współbieżnych serwerów MCP. Jaka struktura danych zastępuje prosty słownik dyspozytorski? (Podpowiedź: trie dla prefiksowania przestrzeni nazw plus metryka liczby narzędzi na serwer.)

5. Przenieś klienta do oficjalnego MCP Python SDK. SDK opakowuje `stdio_client` i `ClientSession`. Kod powinien skurczyć się z ~200 linii do ~40 linii, zachowując routing wieloserwerowy.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Klient MCP | "Host agenta" | Proces, który uruchamia serwery i orkiestruje wywołania narzędzi |
| Sesja | "Stan na serwer" | Możliwości, lista narzędzi i księgowość oczekujących żądań |
| Scalona przestrzeń nazw | "Jedna lista narzędzi" | Płaski zestaw nazw narzędzi ze wszystkich aktywnych serwerów |
| Kolizja przestrzeni nazw | "Dwa serwery to samo narzędzie" | Klient musi prefiksować, odrzucić lub dać pierwszeństwo duplikatowi |
| Routing | "Kto dostaje to wywołanie?" | Dyspozycja z nazwy narzędzia do właściciela serwera |
| Czytnik tła | "Nieblokujące stdout" | Wątek lub zadanie opróżniające stdout serwera do kolejki |
| Wywołanie zwrotne próbkowania | "LLM jako usługa" | Handler klienta dla `sampling/createMessage` z serwera |
| `notifications/*_changed` | "Prymityw zmutowany" | Sygnał, że klient musi ponownie odkryć lub odczytać |
| Polityka ponownego łączenia | "Gdy serwer umiera" | Semantyka restartu, gdy transport ulegnie awarii |
| Sesja stdio | "Proces = sesja" | Brak identyfikatora sesji; czas życia procesu potomnego to sesja |

## Dalsza Lektura

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — kanoniczne zachowanie klienta
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — tutorial klienta hello-world z Python SDK
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — referencyjny `ClientSession` i `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — odpowiednik TS
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — jak VS Code multipleksuje wiele serwerów MCP w jednym hoście edytora