# Transporty MCP — stdio vs Streamable HTTP vs Migracja SSE

> stdio działa lokalnie i nigdzie indziej. Streamable HTTP (2025-03-26) to standard zdalny. Stary transport HTTP+SSE jest wycofywany i usuwany w połowie 2026 roku. Wybór złego transportu kosztuje migrację; wybór właściwego daje zdalnie hostowany serwer MCP z ciągłością sesji i ochroną przed DNS-rebinding.

**Type:** Learn
**Languages:** Python (stdlib, Streamable HTTP endpoint skeleton)
**Prerequisites:** Phase 13 · 07, 08 (MCP server and client)
**Time:** ~45 minutes

## Cele Lekcji

- Wybrać między stdio a Streamable HTTP w oparciu o kształt wdrożenia (lokalne vs zdalne, jeden proces vs flota).
- Zaimplementować wzór pojedynczego punktu końcowego Streamable HTTP: POST dla żądań, GET dla strumienia sesji.
- Wymusić walidację `Origin` i semantykę identyfikatora sesji, aby pokonać DNS-rebinding.
- Zmigrować stary serwer HTTP+SSE do Streamable HTTP przed terminami wycofania w połowie 2026 roku.

## Problem

Pierwszy zdalny transport MCP (2024-11) to HTTP+SSE: dwa punkty końcowe, jeden dla POST-ów klienta i jeden kanał Server-Sent-Events dla strumienia serwer-klient. Działał. Był też niezgrabny: dwa punkty końcowe na sesję, zepsute cache przed niektórymi CDN-ami i twarda zależność od długożyciowych połączeń SSE, które niektóre WAF-y agresywnie zrywają.

Specyfikacja 2025-03-26 zastąpiła go Streamable HTTP: jeden punkt końcowy, POST dla żądań klienta, GET dla ustanowienia strumienia sesji, oba współdzielące nagłówek `Mcp-Session-Id`. Każdy serwer zbudowany lub zmigrowany od tego czasu używa Streamable HTTP. Stary tryb SSE jest wycofywany — Atlassian Rovo usunął go 30 czerwca 2026; Keboola 1 kwietnia 2026; większość pozostałych serwerów korporacyjnych do końca 2026.

A stdio wciąż ma znaczenie dla serwerów lokalnych. Claude Desktop, VS Code i każdy klient w kształcie IDE uruchamiają serwery przez stdio. Właściwy model mentalny: stdio dla "tej maszyny", Streamable HTTP dla "przez sieć". Żadnego przenikania.

## Koncepcja

### stdio

- Transport procesu potomnego. Klient uruchamia serwer, komunikacja przez stdin/stdout.
- Jeden obiekt JSON na linię. Oddzielany znakiem nowej linii.
- Brak identyfikatora sesji; tożsamość procesu to sesja.
- Brak potrzeby uwierzytelniania (proces potomny dziedziczy granicę zaufania rodzica).
- Nigdy nie używaj dla serwerów zdalnych — potrzebowałbyś SSH lub socat do tunelowania, w którym to momencie użyj Streamable HTTP.

### Streamable HTTP

Pojedynczy punkt końcowy `/mcp` (lub dowolna ścieżka). Obsługuje trzy metody HTTP:

- **POST /mcp.** Klient wysyła wiadomość JSON-RPC. Serwer odpowiada albo pojedynczą odpowiedzią JSON, albo strumieniem SSE z jedną lub więcej odpowiedziami (przydatne dla odpowiedzi wsadowych i notyfikacji związanych z tym żądaniem).
- **GET /mcp.** Klient otwiera długożyciowy kanał SSE. Serwer używa go dla żądań serwer-klient (próbkowanie, notyfikacje, elicytacja).
- **DELETE /mcp.** Klient jawnie kończy sesję.

Sesje są identyfikowane przez nagłówek `Mcp-Session-Id`, który serwer ustawia w pierwszej odpowiedzi, a klient powtarza w każdym kolejnym żądaniu. Identyfikatory sesji MUSZĄ być kryptograficznie losowe (128+ bitów); identyfikatory wybrane przez klienta są odrzucane dla bezpieczeństwa.

### Pojedynczy punkt końcowy vs dwa

Tryb dwóch punktów końcowych ze starej specyfikacji jest nadal wywoływalny w 2026 — specyfikacja deklaruje go jako "zgodny ze starszymi wersjami". Ale wszystkie nowe serwery powinny być jednopunktowe. Oficjalne SDK emitują jeden punkt końcowy; używaj trybu starszego tylko wtedy, gdy rozmawiasz z niezmigrowanym zdalnym serwerem.

### Walidacja `Origin` i DNS-rebinding

Przeglądarki nie są klientami MCP (dzisiaj), ale osoba atakująca może stworzyć stronę internetową, która przekona przeglądarkę do wykonania POST na `localhost:1234/mcp` — gdzie nasłuchuje lokalny serwer MCP użytkownika. Jeśli serwer nie sprawdza `Origin`, polityka samego pochodzenia przeglądarki go nie uratuje, ponieważ `Origin: http://evil.com` jest prawidłowe dla różnych źródeł.

Specyfikacja 2025-11-25 wymaga, aby serwery odrzucały żądania, których `Origin` nie znajduje się na liście dozwolonych. Lista dozwolonych zazwyczaj zawiera host klienta MCP (`https://claude.ai`, `vscode-webview://*`) i warianty localhost dla lokalnych interfejsów.

### Cykl życia identyfikatora sesji

1. Klient wysyła pierwsze żądanie bez `Mcp-Session-Id`.
2. Serwer przypisuje losowy identyfikator, ustawia `Mcp-Session-Id` w nagłówku odpowiedzi.
3. Klient powtarza ten nagłówek we wszystkich kolejnych żądaniach i w `GET /mcp` dla strumienia.
4. Sesja może zostać odwołana przez serwer; klient widzi 404 przy kolejnych żądaniach i musi ponownie zainicjować.
5. Klient może jawnie DELETE'ować sesję dla czystego zamknięcia.

### Keepalive i ponowne łączenie

Połączenia SSE zrywają się. Klient przywraca je, wykonując ponownie GET z tym samym `Mcp-Session-Id`. Serwer MUSI kolejkować zdarzenia pominięte podczas przerwy (w rozsądnym oknie) i powtarzać je przez nagłówek `last-event-id`, który klient powtarza.

Faza 13 · 13 obejmuje zadania (Tasks), które pozwalają długotrwałej pracy przetrwać nawet pełne ponowne połączenie sesji.

### Sonda kompatybilności wstecznej

Klient, który chce obsługiwać zarówno stare, jak i nowe serwery:

1. POST na `/mcp`.
2. Jeśli odpowiedź to `200 OK` z JSON lub SSE, to jest Streamable HTTP.
3. Jeśli odpowiedź to `200 OK` z `Content-Type: text/event-stream` ORAZ nagłówkiem `Location` wskazującym na drugorzędny punkt końcowy, to jest legacy HTTP+SSE; podążaj za `Location`.

### Cloudflare, ngrok i hosting

Produkcyjne zdalne serwery MCP w 2026 roku działają na Cloudflare Workers (z ich MCP Agents SDK), Vercel Functions lub w konteneryzowanym Node/Python. Klucz: twój hosting musi obsługiwać długożyciowe połączenia HTTP dla SSE GET. Darmowy tier Vercel ogranicza do 10 sekund i jest nieodpowiedni. Cloudflare Workers obsługują nieograniczone strumienie.

### Kompozycja bramy

Gdy stawiasz wiele serwerów MCP za bramą (Faza 13 · 17), brama jest pojedynczym punktem końcowym Streamable HTTP, który przepisuje identyfikatory sesji i multipleksuje upstream. Narzędzia są scalane na warstwie bramy; klient widzi jeden logiczny serwer.

### Tryby awarii transportu

- **stdio SIGPIPE.** Śmierć procesu potomnego w trakcie zapisu podnosi SIGPIPE; serwery powinny zakończyć czysto. Klienci powinni wykryć EOF i oznaczyć sesję jako martwą.
- **HTTP 502 / 504.** Cloudflare, nginx i inne proxy emitują je przy awarii upstream. Klienci Streamable HTTP powinni ponowić próbę raz po krótkim backoffie.
- **Upadek połączenia SSE.** TCP RST, limit czasu proxy lub zmiana sieci klienta zamyka strumień. Klient łączy się ponownie z `Mcp-Session-Id` i opcjonalnym `last-event-id`, aby wznowić.
- **Odwołanie sesji.** Serwer unieważnia identyfikator sesji; klient widzi 404 przy następnym żądaniu. Klient musi ponownie przeprowadzić uzgadnianie.
- **Rozbieżność zegara.** Obliczenia TTL zasobu po stronie klienta rozchodzą się z serwerem. Klient powinien traktować znaczniki czasu serwera jako autorytatywne.

### Kiedy ominąć Streamable HTTP

Niektóre przedsiębiorstwa wdrażają serwery MCP za transportami gRPC lub kolejek komunikatów we własnych sieciach. Jest to niestandardowe — specyfikacja MCP formalnie ich nie definiuje. Bramy mogą udostępniać powierzchnię Streamable HTTP klientom MCP, używając wewnętrznie gRPC. Utrzymuj zewnętrzną powierzchnię zgodną ze specyfikacją; bama odpowiada za tłumaczenie.

## Użyj Tego

`code/main.py` implementuje minimalny punkt końcowy Streamable HTTP przy użyciu `http.server` (stdlib). Obsługuje POST, GET i DELETE na `/mcp`, ustawia `Mcp-Session-Id` przy pierwszej odpowiedzi, waliduje `Origin` i odrzuca żądania z nieautoryzowanych źródeł. Handler ponownie używa logiki dyspozytora serwera notatek z Lekcji 07.

Na co zwrócić uwagę:

- Handler POST odczytuje treść JSON-RPC, dysponuje i zapisuje odpowiedź JSON (wariant pojedynczej odpowiedzi; wariant SSE jest strukturalnie podobny).
- Sprawdzenie `Origin` odrzuca domyślną sondę `http://evil.example`, ale akceptuje `http://localhost`.
- Identyfikatory sesji to losowe 128-bitowe ciągi szesnastkowe; serwer przechowuje stan na sesję w pamięci.

## Dostarcz

Ta lekcja produkuje `outputs/skill-mcp-transport-migrator.md`. Mając serwer MCP HTTP+SSE (legacy), umiejętność produkuje plan migracji do Streamable HTTP z ciągłością identyfikatora sesji, kontrolami Origin i obsługą sondy kompatybilności wstecznej.

## Ćwiczenia

1. Uruchom `code/main.py`. Wyślij POST z `initialize` z `curl` i zaobserwuj nagłówek odpowiedzi `Mcp-Session-Id`. Wyślij drugie żądanie POST powtarzające nagłówek i sprawdź ciągłość sesji.

2. Dodaj handler GET, który otwiera strumień SSE. Wyślij jedno zdarzenie `notifications/progress` co pięć sekund. Połącz się ponownie, wykonując ponownie GET z tym samym identyfikatorem sesji i potwierdź, że serwer je akceptuje.

3. Zaimplementuj logikę powtarzania `last-event-id`. Przy ponownym połączeniu powtórz wszystkie zdarzenia wygenerowane od tego identyfikatora.

4. Rozszerz walidację `Origin` o obsługę wzorca wieloznacznego (`https://*.example.com`) i potwierdź, że akceptuje `https://app.example.com`, ale odrzuca `https://evil.example.com.attacker.net`.

5. Weź stary serwer HTTP+SSE z oficjalnego rejestru (jest ich kilka) i naszkicuj migrację: co zmienia się w obsłudze punktów końcowych, generowaniu identyfikatorów sesji i semantyce nagłówków.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Transport stdio | "Lokalny proces potomny" | JSON-RPC przez stdin/stdout, oddzielany znakiem nowej linii |
| Streamable HTTP | "Zdalny transport" | Jeden punkt końcowy POST + GET + opcjonalne SSE, specyfikacja 2025-03-26 |
| HTTP+SSE | "Legacy" | Model dwóch punktów końcowych usuwany w połowie 2026 roku |
| `Mcp-Session-Id` | "Nagłówek sesji" | Losowy identyfikator przypisany przez serwer, powtarzany w każdym kolejnym żądaniu |
| Lista dozwolonych `Origin` | "Obrona DNS-rebinding" | Odrzucaj żądania, których Origin nie jest zatwierdzone |
| Pojedynczy punkt końcowy | "Jeden URL" | `/mcp` obsługuje POST / GET / DELETE dla wszystkich operacji sesji |
| `last-event-id` | "Powtarzanie SSE" | Nagłówek używany do wznowienia zerwanego strumienia bez pomijania zdarzeń |
| Sonda kompatybilności wstecznej | "Wykrywanie stare vs nowe" | Sprawdzenie kształtu odpowiedzi klienta, które automatycznie wybiera transport |
| Długożyciowe HTTP | "Strumieniowanie SSE" | Serwer wypycha zdarzenia przez minuty lub godziny na jednym połączeniu TCP |
| Odwołanie sesji | "Wymuś ponowną inicjalizację" | Serwer unieważnia identyfikator sesji; klient musi ponownie przeprowadzić uzgadnianie |

## Dalsza Lektura

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — kanoniczna referencja dla stdio i Streamable HTTP
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — rewizja, która wprowadziła Streamable HTTP
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — wzorce Streamable HTTP hostowane na Workers
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — porównanie dla różnych kształtów wdrożenia
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — konkretny przykład terminu migracji