# Agenci Przeglądarkowi i Długohoryzontowe Zadania Webowe

> Agent ChatGPT (lipiec 2025) połączył Operatora i deep research w jednego agenta przeglądarkowego/terminalowego i ustanowił SOTA na BrowseComp na 68.9%. OpenAI zamknął Operatora 31 sierpnia 2025 — konsolidacja na poziomie produktu. Przejęcie Vercept przez Anthropic przesunęło Claude Sonnet na OSWorld z poniżej 15% do 72.5%. WebArena-Verified (ServiceNow, ICLR 2026) naprawiło 11.3 punktów procentowych wskaźnika fałszywie negatywnych w oryginalnym WebArena i dostarczyło 258-zadaniowy podzbiór Hard. Liczby są prawdziwe. Tak samo jak powierzchnia ataku: szef działu przygotowania OpenAI stwierdził publicznie, że pośrednie wstrzyknięcie promptu do agentów przeglądarkowych „nie jest błędem, który można w pełni załatać". Udokumentowane ataki 2025–2026: Tainted Memories (Atlas CSRF), HashJack (Cato Networks) i przejęcia jednym kliknięciem w Perplexity Comet.

**Type:** Learn
**Languages:** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## Problem

Agent przeglądarkowy to długohoryzontowy agent, który czyta niezaufane treści i podejmuje doniosłe działania. Każda strona, którą agent odwiedza, to dane wejściowe, których użytkownik nie napisał. Każdy formularz na każdej stronie to potencjalny kanał poleceń. Korpus ataków 2025–2026 pokazuje, że to nie jest hipotetyczne: Tainted Memories pozwala atakującemu związać złośliwe instrukcje z pamięcią agenta poprzez spreparowaną stronę; HashJack ukrywa polecenia we fragmentach URL, które agent odwiedza; przejęcia Perplexity Comet trafiają w jednym kliknięciu.

Obraz obrony jest niewygodny. Szef działu przygotowania OpenAI powiedział głośno to, co cicho: pośrednie wstrzyknięcie promptu „nie jest błędem, który można w pełni załatać". Dzieje się tak, ponieważ atak żyje na granicy czytania-a-działania agenta, która jest architektonicznie rozmyta — każdy token, który model czyta, może w zasadzie być odczytany jako instrukcja.

Ta lekcja nazywa powierzchnię ataku, nazywa krajobraz benchmarków (BrowseComp, OSWorld, WebArena-Verified) i modeluje minimalny scenariusz pośredniego wstrzyknięcia promptu, abyś mógł rozumować o prawdziwych obronach w Lekcjach 14 i 18.

## Koncepcja

### Krajobraz 2026, jeden akapit na system

**Agent ChatGPT (OpenAI).** Uruchomiony w lipcu 2025. Unifikuje Operatora (przeglądanie) i Deep Research (wielogodzinne badania). Zamknął samodzielnego Operatora 31 sierpnia 2025. SOTA na BrowseComp na 68.9%; mocne wyniki na OSWorld i WebArena-Verified.

**Claude Sonnet + Vercept (Anthropic).** Przejęcie Vercept przez Anthropic skupiło się na zdolnościach używania komputera. Przesunęło Claude Sonnet na OSWorld z <15% do 72.5%. Claude Computer Use jest dostarczany jako API narzędzia.

**Gemini 3 Pro z Browser Use (DeepMind).** Integracja Browser Use dostarcza kontroli używania komputera; FSF v3 (kwiecień 2026, Lekcja 20) śledzi autonomię w domenie ML R&D.

**WebArena-Verified (ServiceNow, ICLR 2026).** Naprawia dobrze udokumentowany problem: oryginalne WebArena miało ~11.3% wskaźnika fałszywie negatywnych (zadania oznaczone jako nieudane, które faktycznie były rozwiązane). Wydanie Verified ponownie ocenia z ręcznie wyselekcjonowanymi kryteriami sukcesu i dodaje 258-zadaniowy podzbiór Hard (artykuł ICLR 2026, openreview.net/forum?id=94tlGxmqkN).

### BrowseComp vs OSWorld vs WebArena

| Benchmark | Co mierzy | Horyzont |
|---|---|---|
| BrowseComp | Znajdowanie konkretnych faktów w otwartej sieci pod presją czasu | minuty |
| OSWorld | Agent obsługujący pełny pulpit (mysz, klawiatura, shell) | dziesiątki minut |
| WebArena-Verified | Transakcyjne zadania webowe w symulowanych witrynach | minuty |
| Podzbiór Hard | Zadania WebArena-Verified z wielostronicowymi przejściami stanu | dziesiątki minut |

Różne osie. Wysoki wynik BrowseComp mówi, że agent znajduje fakty; nie mówi, że agent może zarezerwować lot. Wynik OSWorld jest bliższy „czy działa na moim pulpicie". WebArena-Verified jest bliższe „czy może ukończyć przepływ". Każda decyzja produkcyjna potrzebuje benchmarku, który pasuje do rozkładu zadań.

### Powierzchnia ataku, nazwana

1. **Pośrednie wstrzyknięcie promptu.** Niezaufana treść strony zawiera instrukcje. Agent je czyta. Agent je wykonuje. Publiczne przykłady: 2024 Kai Greshake et al., 2025 artykuł Tainted Memories, 2026 HashJack (Cato Networks).
2. **Wstrzyknięcie w fragment URL / zapytanie.** `#fragment` lub ciąg zapytania przeszukanego URL zawiera polecenia. Nigdy nie renderowane widocznie; wciąż w kontekście agenta.
3. **Ataki wiązania pamięci.** Strona instruuje agenta, aby zapisał trwałą pamięć (Lekcja 12 opisuje trwały stan). Następna sesja, pamięć uruchamia ładunek bez widocznego wyzwalacza.
4. **Ataki w stylu CSRF na uwierzytelnione sesje.** Klasa Tainted Memories: agent jest gdzieś zalogowany; strona atakującego wysyła żądania zmiany stanu, które agent wykonuje z ciasteczkami użytkownika.
5. **Przejęcie jednym kliknięciem.** Wizualnie niepozorny przycisk niesie ładunek, który agent wykonuje. Klasa Comet.
6. **Dziury Content-Security-Policy w powierzchni hosta agenta.** Warstwy renderowania i narzędzi mogą same być wektorami ataku; stos przeglądarki-w-agenci-przeglądarkowym jest szeroki.

### Dlaczego „nie w pełni łatwalne"

Atak jest izomorficzny do zdolności agenta. Agent musi czytać niezaufane treści, aby wykonywać swoją pracę. Każda treść, którą agent czyta, może zawierać instrukcje. Każde instrukcje, które agent wykonuje, mogą być niezgodne z rzeczywistym żądaniem użytkownika. Obrony (granice zaufania, klasyfikatory, listy dozwolonych narzędzi, HITL na doniosłych działaniach) podnoszą koszt ataku i zmniejszają jego promień rażenia. Nie zamykają klasy.

To ten sam wzorzec rozumowania co twierdzenie Loba (Lekcja 8): agent nie może udowodnić, że następny token jest bezpieczny; może tylko ustawić system, w którym niebezpieczne tokeny są bardziej wykrywalne.

### Postawa obronna, która faktycznie trafia do produkcji

- **Granica odczytu/zapisu.** Czytanie nigdy nie jest doniosłe. Pisanie (wysyłanie formularza, publikowanie treści, wywołanie narzędzia z efektami ubocznymi) wymaga świeżej zgody człowieka, jeśli inicjująca treść pochodzi spoza granicy zaufania.
- **Lista dozwolonych narzędzi na zadanie.** Agent może przeglądać; nie może inicjować przelewu bankowego, chyba że to narzędzie zostało jawnie włączone dla zadania. Lekcja 13 obejmuje budżety.
- **Izolacja sesji.** Sesje agenta przeglądarkowego działają z ograniczonymi poświadczeniami. Brak produkcyjnego uwierzytelnienia, brak osobistej poczty. Logi każdego żądania HTTP przechowywane do audytu.
- **Dezynfekcja treści.** Pobrane HTML jest pozbawione znanych-złych wzorców przed połączeniem w kontekst modelu. (Zmniejsza łatwe ataki; nie zatrzymuje wyrafinowanych ładunków.)
- **HITL na doniosłych działaniach.** Wzorzec proponuj-następnie-zatwierdź (Lekcja 15).
- **Tokeny kanarkowe w pamięci.** Jeśli wpis pamięci zostanie uruchomiony, użytkownik go widzi (Lekcja 14).

## Użyj Tego

`code/main.py` modeluje malutkie uruchomienie agenta przeglądarkowego przeciwko trzem syntetycznym stronom. Jedna strona jest benigna, jedna ma bezpośredni blob wstrzyknięcia promptu w widocznym tekście, jedna ma wstrzyknięcie w fragmencie URL (niewidoczne, ale wewnątrz kontekstu agenta). Skrypt pokazuje (a) co naiwny agent by zrobił, (b) co łapie granica odczytu/zapisu, (c) co łapie dezynfektor, (d) czego nie łapie żadne.

## Wdróż To

`outputs/skill-browser-agent-trust-boundary.md` zakresuje proponowane wdrożenie agenta przeglądarkowego: które strefy zaufania dotyka, co jest autoryzowane do zapisu i które obrony muszą być na miejscu przed pierwszym uruchomieniem.

## Ćwiczenia

1. Uruchom `code/main.py`. Zidentyfikuj, który atak łapie dezynfektor, ale nie granica odczytu/zapisu, i który atak łapie tylko granica odczytu/zapisu.

2. Rozszerz dezynfektor, aby wykrywał jedną klasę wstrzyknięcia w fragment URL w stylu HashJack. Zmierz wskaźnik fałszywie pozytywnych na benignych URL z legalnymi fragmentami.

3. Wybierz jeden prawdziwy przepływ pracy agenta przeglądarkowego, który znasz (np. „zarezerwuj lot"). Wypisz każdy odczyt i każdy zapis. Oznacz, które zapisy wymagają HITL i dlaczego.

4. Przeczytaj artykuł WebArena-Verified ICLR 2026. Zidentyfikuj jedną kategorię zadań, w której oryginalne punktowanie WebArena było zawodne, i wyjaśnij, jak podzbiór Verified to rozwiązuje.

5. Zaprojektuj kanarek pamięci dla ustawienia agenta przeglądarkowego. Co byś przechowywał, gdzie i co uruchamia alarm?

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|---|---|---|
| Pośrednie wstrzyknięcie promptu | „Zły tekst strony" | Niezaufana treść na stronie, którą agent czyta, zawiera instrukcje, które agent wykonuje |
| Tainted Memories | „Atak na pamięć" | Agent zapisuje instrukcję dostarczoną przez atakującego do trwałej pamięci; uruchomiona w następnej sesji |
| HashJack | „Atak na fragment URL" | Ładunek ukryty w fragmencie/ciągu zapytania URL jest w kontekście agenta, ale nie jest widocznie renderowany |
| Przejęcie jednym kliknięciem | „Zły przycisk" | Widoczna funkcja niesie następny ładunek, który agent wykonuje |
| BrowseComp | „Benchmark wyszukiwania w sieci" | Znajdowanie konkretnych faktów w otwartej sieci; horyzont minutowy |
| OSWorld | „Benchmark pulpitu" | Pełna kontrola OS; wieloetapowe zadania GUI |
| WebArena-Verified | „Naprawiony benchmark zadań webowych" | Przeskalowane WebArena ServiceNow z podzbiorem Hard |
| Granica odczytu/zapisu | „Bramka efektów ubocznych" | Czytanie nigdy doniosłe; zapis wymaga świeżego zatwierdzenia, jeśli treść jest poza zaufaniem |

## Dalsza Lektura

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — połączenie Operatora i deep research; SOTA BrowseComp.
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — linia Operatora i architektura, która stała się agentem ChatGPT.
- [Zhou et al. — WebArena](https://webarena.dev/) — oryginalny benchmark.
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — artykuł o naprawionym podzbiorze ICLR 2026.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — zawiera dyskusję powierzchni ataku dla agentów używających komputera.
