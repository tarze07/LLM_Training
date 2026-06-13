# Interfejs narzędzia — Dlaczego agenci potrzebują strukturalnego We/Wy

> Model językowy produkuje tokeny. Program wykonuje akcje. Przepaść między nimi to interfejs narzędzia: kontrakt, pozwalający modelowi zażądać akcji, a hostowi ją wykonać. Każdy stos z 2026 roku — function calling w OpenAI, Anthropic i Gemini; `tools/call` w MCP; części zadań w A2A — to inna wersja tego samego czteroetapowego cyklu. Ta lekcja nazywa ten cykl i pokazuje minimalne mechanizmy do jego uruchomienia.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## Learning Objectives

- Wyjaśnij, dlaczego LLM, który potrafi tylko generować tekst, nie może samodzielnie wykonywać akcji w rzeczywistym świecie.
- Narysuj czteroetapowy cykl wywołania narzędzia (opisz → zdecyduj → wykonaj → obserwuj) i nazwij, kto jest właścicielem każdego kroku.
- Napisz opis narzędzia jako trzy części: nazwę, wejście JSON Schema i deterministyczną funkcję wykonawczą.
- Odróżnij narzędzia czyste i powodujące skutki uboczne oraz uzasadnij, dlaczego ten podział ma znaczenie dla bezpieczeństwa.

## The Problem

LLM emituje rozkład prawdopodobieństwa nad następnym tokenem. To cała powierzchnia wyjściowa. Jeśli zapytasz model czatu "jaka jest teraz pogoda w Bengaluru", może napisać prawdopodobne zdanie, ale nie może zadzwonić do API pogodowego. To zdanie może być poprawne przez przypadek lub nieaktualne o trzy dni.

Zamknięcie tej luki jest celem interfejsu narzędzia. Program hosta — Twoje środowisko uruchomieniowe agenta, Claude Desktop, ChatGPT, Cursor lub niestandardowy skrypt — reklamuje modelowi listę wywoływalnych narzędzi. Model, gdy zdecyduje, że potrzebna jest akcja, emituje strukturalny ładunek nazywający narzędzie i jego argumenty. Host analizuje ten ładunek, uruchamia prawdziwe narzędzie i przekazuje wynik z powrotem. Cykl trwa, aż model zdecyduje, że nie są potrzebne dalsze wywołania.

Pierwsza wersja tego kontraktu została wydana w czerwcu 2023 jako parametr "functions" w OpenAI. Anthropic odpowiedział blokami `tool_use` w Claude 2.1. Gemini dodał `functionDeclarations` kilka miesięcy później. Każdy dostawca udostępnia teraz ten sam kształt: listę narzędzi w formacie JSON Schema na wejściu i ładunek JSON na wyjściu. Model Context Protocol (listopad 2024) uogólnił kontrakt, aby jeden rejestr narzędzi mógł obsługiwać każdy model. A2A (kwiecień 2026, v1.0) nałożył ten sam prymityw dla delegacji agent-agent.

Czteroetapowy cykl jest niezmiennikiem pod wszystkim powyższym. Wszystko inne w Fazie 13 jest rozwinięciem.

## The Concept

### Krok pierwszy: opisz

Host deklaruje każde narzędzie za pomocą trzech pól.

- **Nazwa.** Stabilny, maszynowo czytelny identyfikator. `get_weather`, nie "rzecz pogodowa".
- **Opis.** Jednoakapitowe, naturalne streszczenie. "Użyj, gdy użytkownik pyta o aktualne warunki dla konkretnego miasta. Nie używaj do danych historycznych."
- **Schemat wejściowy.** Obiekt JSON Schema (draft 2020-12) opisujący argumenty narzędzia.

Model otrzymuje listę. Nowocześni dostawcy serializują te deklaracje do prompta systemowego przy użyciu szablonu specyficznego dla dostawcy, więc Ty jako wywołujący masz do czynienia tylko ze sformatowaną formą.

### Krok drugi: zdecyduj

Mając wiadomość użytkownika i dostępne narzędzia, model wybiera jedno z trzech zachowań.

1. **Odpowiedz bezpośrednio** tekstem. Brak wywołania narzędzia.
2. **Wywołaj jedno lub więcej narzędzi.** Wyemituj strukturalne obiekty wywołań. Przy `parallel_tool_calls: true` (domyślnie w OpenAI i Gemini, opcjonalnie w Anthropic) model może wyemitować wiele wywołań w jednej turze.
3. **Odmów.** Ściśle strukturalne wyjścia (strict-mode) mogą wyprodukować typowany blok `refusal` zamiast wywołania.

Ładunek wywołania narzędzia ma trzy stabilne pola: identyfikator `id`, nazwę narzędzia `name` i obiekt JSON `arguments` z argumentami. Id istnieje, aby host mógł skorelować późniejszy wynik z konkretnym wywołaniem, co ma znaczenie, gdy równoległe wywołania wracają w nieuporządkowanej kolejności.

### Krok trzeci: wykonaj

Host otrzymuje wywołanie, waliduje argumenty względem zadeklarowanego schematu i uruchamia wykonawcę. Nieprawidłowe argumenty oznaczają, że model zhallucynował pole lub użył złego typu — bardzo częsty tryb awarii na słabszych modelach. Produkcyjne hosty robią jedną z trzech rzeczy w przypadku nieprawidłowych argumentów: szybko zawodzą i przekazują błąd do modelu, naprawiają JSON za pomocą ograniczonego parsera lub ponawiają model z błędem walidacji dołączonym do prompta.

Sam wykonawca to zwykły kod. Python, TypeScript, polecenie powłoki, zapytanie do bazy danych. Produkuje wynik, który zazwyczaj jest ciągiem znaków, ale może być dowolną wartością JSON lub strukturalnym blokiem treści (tekst, obraz lub referencja do zasobu w MCP). Wynik musi być serializowalny.

### Krok czwarty: obserwuj

Host dołącza wynik narzędzia do konwersacji (jako wiadomość z rolą `tool` z pasującym `id`) i ponownie wywołuje model. Model ma teraz wynik narzędzia w kontekście i może wyprodukować ostateczną odpowiedź lub zażądać dalszych wywołań. Trwa to, aż model przestanie emitować wywołania lub host osiągnie limit bezpieczeństwa liczby iteracji.

### Podział zaufania

Narzędzia występują w dwóch wariantach, które mają znaczenie dla bezpieczeństwa.

- **Czyste.** Tylko do odczytu, deterministyczne, bez skutków ubocznych. `get_weather`, `search_docs`, `get_current_time`. Bezpieczne do spekulacyjnego wywoływania.
- **Konsekwencjalne.** Modyfikuje stan, wydaje pieniądze, dotyka danych użytkownika. `send_email`, `delete_file`, `execute_trade`. Muszą być blokowane.

Zasada "Dwóch" Meta z 2026 dla bezpieczeństwa agentów mówi, że pojedyncza tura może łączyć co najwyżej dwa z: niezaufanych wejść, wrażliwych danych, konsekwencjalnych działań. Interfejs narzędzia jest miejscem, gdzie egzekwujesz tę zasadę — przez odrzucanie wywołań, wymaganie potwierdzenia użytkownika lub eskalację zakresów. Zobacz Fazę 13 · 15 dla pełnego rozdziału o bezpieczeństwie i Fazę 14 · 09 dla zasad uprawnień na poziomie agenta.

### Gdzie żyje cykl

| Kontekst | Kto opisuje | Kto decyduje | Kto wykonuje |
|---------|---------------|-------------|--------------|
| Jednoturowe function calling (OpenAI/Anthropic/Gemini) | Deweloper aplikacji | LLM | Deweloper aplikacji |
| MCP | Serwer MCP | LLM przez klienta MCP | Serwer MCP |
| A2A | Wydawca Agent Card | Wywołujący agent | Wywoływany agent |
| Przeglądarka internetowa (agent function-calling) | Rozszerzenie przeglądarki / WebMCP | LLM | Środowisko przeglądarki |

Wszędzie te same cztery kroki. Nazwy kolumn się zmieniają; struktura nie.

### Dlaczego nie po prostu nakłonić model do emitowania JSON?

"Poproś model o odpowiedź w JSON" był wzorcem sprzed function calling. Zawodzi ~5 do 15 procent przypadków na granicznych modelach i znacznie więcej na mniejszych. Tryby awarii obejmują brakujące nawiasy klamrowe, końcowe przecinki, zhallucynowane pola i niewłaściwe typy. Potem potrzebujesz poprawki JSON, ponowienia lub ograniczonego dekodera.

Natywne function calling jest lepsze z trzech powodów. Po pierwsze, dostawca trenuje model end-to-end na dokładnym kształcie wywołania, więc wskaźnik poprawnego JSON wzrasta do 98 do 99 procent w trybie ścisłym. Po drugie, ładunek wywołania siedzi we własnym gnieździe protokołu, a nie w wolnym tekście — więc wywołanie narzędzia nigdy nie wycieka do widocznej dla użytkownika odpowiedzi. Po trzecie, dostawcy egzekwują zgodność schematu za pomocą ograniczonego dekodowania (strict mode OpenAI, `tool_use` Anthropic, `responseSchema` Gemini). Wynik jest gwarantowany do walidacji.

Faza 13 · 02 porównuje trzy API dostawców obok siebie. Faza 13 · 04 zagłębia się w strukturalne wyjścia.

### Bezpieczniki

Cykl kończy się, gdy model przestaje emitować wywołania lub host osiągnie maksymalną liczbę tur. Produkcyjne hosty ustawiają to na między 5 a 20 tur. Powyżej tego prawie na pewno jesteś w pętli, z której model nie może wyjść. Claude Code domyślnie ustawia 20; OpenAI Assistants 10; tryb agenta Cursor 25.

Alternatywa — nieograniczone pętle — pojawia się co sześć miesięcy w postaci raportów "agent wydał 400$ na wywołania API przez noc". Nie wdrażaj bez ograniczenia.

Faza 14 · 12 szczegółowo omawia odzyskiwanie błędów i samonaprawę; Faza 17 obejmuje produkcyjne limity szybkości.

### Dokąd zmierza Faza 13 stąd

- Lekcje 02 do 05 szlifują powierzchnię wywołań narzędzi na poziomie dostawcy.
- Lekcje 06 do 14 uogólniają cykl w MCP.
- Lekcje 15 do 18 bronią cyklu przed wrogimi serwerami, adversarialnymi użytkownikami i nieuwierzytelnionymi zdalnymi powierzchniami auth.
- Lekcje 19 do 22 rozszerzają wzorzec na współpracę agent-agent, obserwowalność, routing i pakowanie.
- Lekcja 23 dostarcza kompletny ekosystem używający każdego prymitywu.

Każda pozostała lekcja jest rozwinięciem tego czteroetapowego cyklu. Miej go w pamięci jako niezmiennik.

## Use It

`code/main.py` uruchamia czteroetapowy cykl bez LLM. Fałszywa funkcja "decydent" symuluje model przez dopasowywanie wzorców na wiadomości użytkownika; wykonawca, walidator schematu i harness kroku obserwacji są prawdziwe. Uruchom go, aby zobaczyć pełną choreografię żądanie/odpowiedź z możliwością wydrukowania stanu pośredniego, a następnie zastąp fałszywego decydenta dowolnym prawdziwym dostawcą w późniejszej lekcji.

Na co zwrócić uwagę:

- Rejestr narzędzi zawiera trzy pola na narzędzie: nazwę, opis, schemat i referencję do wykonawcy.
- Walidator to minimalny podzbiór JSON Schema (typy, required, enum, min/max) napisany tylko w stdlib. Faza 13 · 04 zawiera pełniejszy.
- Cykl ogranicza liczbę iteracji do pięciu. Agenci produkcyjni potrzebują dokładnie takiego bezpiecznika.

## Ship It

Ta lekcja produkuje `outputs/skill-tool-interface-reviewer.md`. Mając szkic definicji narzędzia (nazwa + opis + schemat + zarys wykonawcy), umiejętność audytuje go pod kątem gotowości cyklu: czy nazwa jest stabilna maszynowo, czy opis jest kompletnym streszczeniem użycia, czy schemat poprawnie używa JSON Schema 2020-12 i czy klasyfikacja czysty-konsekwencjalny jest jawna.

## Exercises

1. Dodaj czwarte narzędzie do `code/main.py` o nazwie `get_stock_price(ticker)`. Napisz jego opis jako "Użyj, gdy użytkownik pyta o aktualną cenę akcji po tickerze. Nie używaj do historycznych cen ani podsumowań rynkowych." Uruchom harness i potwierdź, że fałszywy decydent kieruje zapytania wspominające tickery do nowego narzędzia.

2. Zepsuj walidator schematu. Przekaż wywołanie, którego obiekt `arguments` nie ma wymaganego pola, i potwierdź, że host odrzuca je przed wykonaniem. Następnie przekaż wywołanie z dodatkowym nieznanym polem. Zdecyduj: czy host powinien odrzucić czy zignorować? Uzasadnij swój wybór argumentem bezpieczeństwa.

3. Sklasyfikuj każde narzędzie w harnessie jako czyste lub konsekwencjalne. Dodaj flagę `consequential: true` do wpisów rejestru, które jej potrzebują, i zmień cykl, aby wyświetlał linię "potwierdziłbym z użytkownikiem" za każdym razem, gdy wybrane zostanie konsekwencjalne narzędzie. To jest kształt bramki potwierdzenia, której potrzebuje każdy produkcyjny host.

4. Narysuj czteroetapowy cykl na papierze z powyższą tabelą kolumn dostawcy wypełnioną dla Twojego ulubionego klienta (Claude Desktop, Cursor, ChatGPT lub niestandardowego stosu). Porównaj z wariantem specyficznym dla MCP w Fazie 13 · 06.

5. Przeczytaj przewodnik function-calling OpenAI od deski do deski. Zidentyfikuj jedno pole, które znajduje się w żądaniu, ale nie w czteroetapowym cyklu przedstawionym tutaj. Wyjaśnij, co dodaje i dlaczego jest wygodne, a nie niezbędne.

## Key Terms

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Narzędzie (Tool) | "Rzecz, którą model może wywołać" | Trójka: nazwa + wejście JSON Schema + funkcja wykonawcza |
| Function calling | "Natywne użycie narzędzia" | Wsparcie API na poziomie dostawcy do emitowania strukturalnych wywołań narzędzi zamiast prozy |
| Wywołanie narzędzia (Tool call) | "Żądanie modelu do działania" | Ładunek JSON z `id`, `name`, `arguments` wyemitowany przez model |
| Wynik narzędzia (Tool result) | "Co zwróciło narzędzie" | Wynik wykonawcy, opakowany w wiadomość z rolą `tool` z pasującym id |
| Równoległe wywołania narzędzi (Parallel tool calls) | "Wiele wywołań naraz" | Wiele obiektów wywołań w jednej turze modelu, niezależnych i możliwych do uporządkowania po id |
| Tryb ścisły (Strict mode) | "Gwarantowany JSON" | Ograniczone dekodowanie, które wymusza zgodność wyjścia modelu z zadeklarowanym schematem |
| Czyste narzędzie (Pure tool) | "Narzędzie tylko do odczytu" | Brak skutków ubocznych; bezpieczne do ponownego uruchomienia |
| Konsekwencjalne narzędzie (Consequential tool) | "Narzędzie akcji" | Modyfikuje stan zewnętrzny; wymaga bramki, audytu lub potwierdzenia użytkownika |
| Czteroetapowy cykl (Four-step loop) | "Cykl wywołania narzędzia" | opisz → zdecyduj → wykonaj → obserwuj |
| Host | "Środowisko agenta" | Program, który przechowuje rejestr narzędzi, wywołuje model i uruchamia wykonawcę |

## Further Reading

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — kanoniczna referencja dla deklaracji narzędzi i kształtów wywołań w stylu OpenAI
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — format bloków `tool_use` / `tool_result` w Claude
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — `functionDeclarations` i semantyka równoległych wywołań w Gemini
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — niezależna od dostawcy generalizacja interfejsu narzędzia
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — dialekt schematu, którym mówi każde nowoczesne API narzędzi
