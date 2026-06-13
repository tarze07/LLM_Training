# Zasoby i Podpowiedzi MCP — Udostępnianie Kontekstu Poza Narzędziami

> Narzędzia otrzymują 90 procent uwagi w MCP. Pozostałe dwa prymitywy serwerowe rozwiązują inne problemy. Zasoby udostępniają dane do odczytu; podpowiedzi udostępniają szablony wielokrotnego użytku jako komendy z ukośnikiem. Wiele serwerów powinno używać zasobów zamiast opakowywania odczytów w narzędzia, a podpowiedzi zamiast kodowania na sztywno przepływów pracy w podpowiedziach klienta. Ta lekcja podaje regułę decyzyjną i omawia wiadomości `resources/*` i `prompts/*`.

**Type:** Build
**Languages:** Python (stdlib, resource + prompt handler)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## Cele Lekcji

- Decydować między udostępnieniem możliwości jako narzędzie, zasób lub podpowiedź dla danej dziedziny.
- Zaimplementować `resources/list`, `resources/read`, `resources/subscribe` i obsłużyć `notifications/resources/updated`.
- Zaimplementować `prompts/list` i `prompts/get` z szablonami argumentów.
- Rozpoznać, kiedy host udostępnia podpowiedzi jako komendy z ukośnikiem, a kiedy jako kontekst automatycznie wstrzykiwany.

## Problem

Naiwny serwer MCP dla aplikacji notatek udostępnia wszystko jako narzędzia: `notes_read`, `notes_list`, `notes_search`. To opakowuje każdy dostęp do danych w wywołanie narzędzia sterowane modelem. Konsekwencje:

- Model musi decydować, czy wywołać `notes_read` dla każdego zapytania, które mogłoby skorzystać z kontekstu.
- Zawartość tylko do odczytu nie może być subskrybowana ani przesyłana strumieniowo do panelu bocznego hosta.
- Interfejsy klienta (panel dołączania zasobów Claude Desktop, selektor "Include file" w Cursor) nie mogą wyświetlić danych.

Właściwy podział: udostępniaj dane jako zasób, udostępniaj akcje mutujące lub obliczeniowe jako narzędzia, udostępniaj wieloetapowe przepływy pracy wielokrotnego użytku jako podpowiedzi. Każdy prymityw ma swoje udogodnienie UX i wzorzec dostępu.

## Koncepcja

### Narzędzia vs zasoby vs podpowiedzi — reguła decyzyjna

| Możliwość | Prymityw |
|------------|-----------|
| Użytkownik chce wyszukiwać, filtrować lub transformować dane | tool |
| Użytkownik chce, aby host dołączył te dane jako kontekst | resource |
| Użytkownik chce szablonowego przepływu pracy, który może uruchomić ponownie | prompt |

Wskazówka: jeśli model odnosiłby korzyść z wywoływania tego przy każdym powiązanym zapytaniu, to jest to narzędzie. Jeśli użytkownik odnosiłby korzyść z dołączenia tego do rozmowy, to jest to zasób. Jeśli cały wieloetapowy przepływ pracy jest jednostką, którą użytkownik chce ponownie wykorzystać, to jest to podpowiedź.

### Zasoby

`resources/list` zwraca `{resources: [{uri, name, mimeType, description?}]}`. `resources/read` przyjmuje `{uri}` i zwraca `{contents: [{uri, mimeType, text | blob}]}`.

URI mogą być czymkolwiek adresowalnym:

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14` (własny schemat)
- `memory://session-2026-04-22/recent` (specyficzny dla serwera)

`contents[]` obsługuje zarówno tekst, jak i dane binarne. Dane binarne używają `blob` jako ciągu zakodowanego w base64 plus `mimeType`.

### Subskrypcje zasobów

Zadeklaruj `{resources: {subscribe: true}}` w możliwościach. Klient wywołuje `resources/subscribe {uri}`. Serwer wysyła `notifications/resources/updated {uri}`, gdy zasób się zmieni. Klient ponownie odczytuje.

Przypadek użycia: serwer notatek, którego zasoby to pliki na dysku; obserwator plików wywołuje notyfikacje aktualizacji; Claude Desktop ponownie pobiera plik do kontekstu, gdy został edytowany poza hostem.

### Szablony zasobów (dodatek 2025-11-25)

`resourceTemplates` pozwalają udostępnić parametryzowany wzór URI: `notes://{id}` z `id` jako celem autouzupełniania. Klient może autouzupełniać identyfikatory w selektorze zasobów.

### Podpowiedzi

`prompts/list` zwraca `{prompts: [{name, description, arguments?}]}`. `prompts/get` przyjmuje `{name, arguments}` i zwraca `{description, messages: [{role, content}]}`.

Podpowiedź to szablon, który wypełnia się do listy wiadomości, które host przekazuje swojemu modelowi. Na przykład podpowiedź `code_review` przyjmuje argument `file_path` i zwraca sekwencję trzech wiadomości: wiadomość systemowa, wiadomość użytkownika z treścią pliku i wstęp asystenta z szablonem rozumowania.

### Hosty i podpowiedzi

Claude Desktop, VS Code i Cursor udostępniają podpowiedzi jako komendy z ukośnikiem w interfejsie czatu. Użytkownik wpisuje `/code_review` i wybiera argumenty z formularza. Podpowiedź serwera to kontrakt między "skrótem użytkownika" a "pełną podpowiedzią wysłaną do modelu".

Nie każdy klient obsługuje jeszcze podpowiedzi — sprawdź negocjację możliwości. Serwer z zadeklarowaną możliwością podpowiedzi, ale klient bez obsługi podpowiedzi, po prostu nie zobaczy komend z ukośnikiem.

### Notyfikacja "lista uległa zmianie"

Zarówno zasoby, jak i podpowiedzi emitują `notifications/list_changed`, gdy zestaw ulegnie mutacji. Serwer notatek, który właśnie zaimportował 20 nowych notatek, emituje `notifications/resources/list_changed`; klient ponownie wywołuje `resources/list`, aby pobrać nowe.

### Konwencje typów zawartości

Dla tekstu: `mimeType: "text/plain"`, `text/markdown`, `application/json`.
Dla danych binarnych: `image/png`, `application/pdf`, plus pole `blob`.
Dla MCP Apps (Lekcja 14): `text/html;profile=mcp-app` w URI `ui://`.

### Zasoby dynamiczne

URI zasobu nie musi odpowiadać statycznemu plikowi. `notes://recent` może zwracać pięć ostatnich notatek przy każdym odczycie. `db://query/users/active` może wykonać parametryzowane zapytanie. Serwer może swobodnie obliczać zawartość dynamicznie.

Zasada: jeśli klient może buforować według URI, URI musi być stabilne. Jeśli obliczenie jest jednorazowe, URI powinno zawierać znacznik czasu lub nonce, aby pamięć podręczna klienta nie przestarzała.

### Subskrypcje a sondowanie

Klienci z możliwością subskrypcji otrzymują push serwera przez `notifications/resources/updated`. Klienci przedsubskrypcyjni lub hosty, które tego nie obsługują, sondują przez ponowny odczyt. Oba są zgodne ze specyfikacją. Deklaracja możliwości serwera mówi klientowi, którą obsługuje.

Koszt subskrypcji: stan na sesję po stronie serwera (kto jest zapisany do czego). Utrzymuj zestaw subskrybowanych w ograniczonych rozmiarach; rozłączeni klienci powinni być wygaszani.

### Podpowiedzi a podpowiedzi systemowe

Podpowiedzi w MCP nie są podpowiedziami systemowymi. Podpowiedź systemowa hosta (jego własne instrukcje operacyjne) i podpowiedzi MCP (szablony dostarczone przez serwer, wywoływane przez użytkownika) współistnieją obok siebie. Dobrze zaprojektowany klient nigdy nie pozwala, aby podpowiedź serwera nadpisywała jego własną podpowiedź systemową; nakłada je warstwowo.

## Użyj Tego

`code/main.py` rozszerza serwer notatek z Lekcji 07 o:

- Zasoby dla każdej notatki (`notes://note-1`, itp.) z obsługą `resources/subscribe`.
- Podpowiedź `review_note`, która renderuje się do szablonu trzech wiadomości.
- Symulację obserwatora plików, która emituje `notifications/resources/updated`, gdy notatka zostanie zmodyfikowana.
- Dynamiczny zasób `notes://recent`, który zawsze zwraca pięć ostatnich notatek.

Uruchom demo, aby zobaczyć pełny przepływ.

## Dostarcz

Ta lekcja produkuje `outputs/skill-primitive-splitter.md`. Mając proponowany serwer MCP, umiejętność kategoryzuje każdą możliwość jako narzędzie / zasób / podpowiedź z uzasadnieniem.

## Ćwiczenia

1. Uruchom `code/main.py`. Zaobserwuj początkową listę zasobów, następnie wywołaj edycję notatki i sprawdź, czy zdarzenie `notifications/resources/updated` zostanie wywołane.

2. Dodaj emiter `resources/list_changed`: gdy nowa notatka zostanie utworzona, wyślij notyfikację, aby klienci ponownie odkrywali.

3. Zaprojektuj trzy podpowiedzi dla serwera MCP GitHub: `summarize_pr`, `triage_issue`, `release_notes`. Każda ze schematami argumentów. Treść podpowiedzi powinna być wykonywalna bez dalszych edycji.

4. Weź istniejące narzędzie z serwera Lekcji 07 i sklasyfikuj, czy powinno pozostać narzędziem, czy zostać podzielone na parę zasób plus narzędzie. Uzasadnij w jednym zdaniu.

5. Przeczytaj sekcje specyfikacji `server/resources` i `server/prompts`. Zidentyfikuj jedno pole w `resources/read`, które rzadko jest wypełniane, ale obsługiwane przez specyfikację. Podpowiedź: spójrz na `_meta` w zawartości zasobu.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Zasób | "Udostępnione dane" | Zawartość adresowalna przez URI, którą host może odczytać |
| URI zasobu | "Wskaźnik do danych" | Identyfikator z prefiksem schematu (`file://`, `notes://`, itp.) |
| `resources/subscribe` | "Obserwuj zmiany" | Aktualizacje push serwera z opcją włączenia przez klienta dla konkretnego URI |
| `notifications/resources/updated` | "Zasób zmieniony" | Sygnał dla klienta, że subskrybowany zasób ma nową zawartość |
| Szablon zasobu | "Parametryzowany URI" | Wzór URI z podpowiedziami autouzupełniania dla selektora hosta |
| Podpowiedź | "Szablon komendy z ukośnikiem" | Nazwany szablon wielo-message z miejscami na argumenty |
| Argumenty podpowiedzi | "Dane wejściowe szablonu" | Typowane parametry, które host zbiera przed renderowaniem |
| `prompts/get` | "Renderuj szablon" | Serwer zwraca wypełnioną listę wiadomości |
| Blok zawartości | "Typowany fragment" | `{type: text \| image \| resource \| ui_resource}` |
| UX komendy z ukośnikiem | "Skrót użytkownika" | Host udostępnia podpowiedzi jako komendy zaczynające się od `/` |

## Dalsza Lektura

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — URI zasobów, subskrypcje i szablony
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — szablony podpowiedzi i integracja komend z ukośnikiem
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — pełna referencja wiadomości `resources/*`
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — pełna referencja wiadomości `prompts/*`
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — przewodnik społeczności rozszerzający oficjalne dokumenty