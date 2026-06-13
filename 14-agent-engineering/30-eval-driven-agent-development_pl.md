# Rozwój agentów sterowany ewaluacją

> Wytyczne Anthropic: "zacznij od prostych promptów, optymalizuj je za pomocą kompleksowej ewaluacji i dodawaj wieloetapowe systemy agentowe tylko wtedy, gdy są potrzebne." Ewaluacja nie jest ostatnim krokiem. To zewnętrzna pętla, która napędza każdy inny wybór w Fazie 14.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** All of Phase 14.
**Time:** ~60 minutes

## Learning Objectives

- Wymień trzy warstwy ewaluacji — statyczne benchmarki, niestandardowe offline, produkcyjne online — i do czego każda służy.
- Wyjaśnij ciasną pętlę evaluator-optimizer.
- Opisz najlepszą praktykę 2026: ewaluacje żyją obok kodu, działają w CI, blokują PR.
- Połącz każdą lekcję Fazy 14 z przypadkiem ewaluacyjnym, który generuje.

## Problem

Agenci przechodzą dema. Zawodzą w produkcji w sposób, którego dema nie mogą przewidzieć. Benchmarki odpowiadają "czy ten model jest ogólnie zdolny?" a nie "czy ten agent wysyła właściwe łatki dla mojego produktu?" Odpowiedź: ewaluacja na trzech warstwach, działająca ciągle, z każdym zabezpieczeniem i wyuczoną regułą mapowaną na przypadek ewaluacyjny.

## Koncepcja

### Trzy warstwy ewaluacji

1. **Statyczne benchmarki** — SWE-bench Verified dla kodu (Lekcja 19), WebArena/OSWorld dla przeglądania/pulpitu (Lekcja 20), GAIA dla generalisty (Lekcja 19), BFCL V4 dla użycia narzędzi (Lekcja 06). Używaj do porównań między modelami i bramkowania regresji. Kontaminacja jest realna: SWE-bench+ znalazł 32.67% wycieku rozwiązań. Zawsze raportuj wyniki Verified / +-audytowane.

2. **Niestandardowe ewaluacje offline** — kształt twojego produktu:
   - LLM-jako-sędzia (Langfuse, Phoenix, Opik — Lekcja 24).
   - Oparte na wykonaniu (uruchom łatkę, sprawdź testy).
   - Oparte na trajektorii (porównaj sekwencje akcji ze złotymi; OSWorld-Human pokazuje najlepszych agentów 1.4-2.7x nad złotem).

3. **Ewaluacje online** — produkcja:
   - Odtworzenia sesji (Langfuse).
   - Alerty wyzwalane przez zabezpieczenia (Lekcja 16, 21).
   - Śledzenie kosztu/opóźnienia na krok (spany OTel Lekcja 23).

### Evaluator-optimizer (Anthropic)

Ciasna pętla:

1. Proponent generuje wyjście.
2. Evaluator ocenia.
3. Udoskonalaj, aż evaluator przejdzie.

To jest Self-Refine (Lekcja 05) uogólnione. Każdy przepływ agenta, na którym ci zależy, może być owinięty w evaluator-optimizer dla niezawodności.

### Najlepsza praktyka 2026

- Ewaluacje żyją obok kodu.
- Działają w CI na każdym PR.
- Blokują scalanie na podstawie wyników ewaluacji (np. "brak regresji > 5% vs main").
- Każde zabezpieczenie mapuje się na przypadek ewaluacyjny.
- Każda wyuczona reguła (Reflexion, pro-workflow learn-rule) mapuje się na przypadek awarii.

### Łączenie Fazy 14

Każda lekcja w Fazie 14 generuje przypadki ewaluacyjne:

| Lekcja | Przypadek ewaluacyjny, który generuje |
|--------|------------------------|
| 01 Agent Loop | Wyczerpany budżet, zabezpieczenie nieskończonej pętli |
| 02 ReWOO | Planista poprawnie planuje ponownie, gdy narzędzie zawiedzie |
| 03 Reflexion | Wyuczone refleksje stosują się przy ponowieniu |
| 05 Self-Refine/CRITIC | Sędzia przechodzi udoskonalone wyjście |
| 06 Tool Use | Koercja argumentów działa; nieznane narzędzia odrzucone |
| 07-10 Memory | Cytaty wyszukiwania pasują do źródeł; nieaktualne fakty unieważniają |
| 12 Workflow Patterns | Każdy wzorzec produkuje poprawne wyjście |
| 13 LangGraph | Wznowienie odtwarza stan dokładnie |
| 14 AutoGen Actors | DLQ łapie awarie handlerów |
| 16 OpenAI Agents SDK | Zabezpieczenie wyzwala się na właściwych wejściach |
| 17 Claude Agent SDK | Wyniki podagentów wracają do orkiestratora |
| 19-20 Benchmarks | Wynik SWE-bench Verified, wskaźnik sukcesu WebArena, wydajność OSWorld |
| 21 Computer Use | Bezpieczeństwo na krok łapie wstrzyknięty DOM |
| 23 OTel | Spany emitują wymagane atrybuty |
| 26 Failure Modes | Detektory oznaczają znane awarie |
| 27 Prompt Injection | PVE odrzuca zatrute pobrania |
| 28 Orchestration | Supervisor kieruje do właściwego specjalisty |
| 29 Runtime Shapes | DLQ obsługuje N% awarii |

Jeśli twój zestaw ewaluacyjny ma przypadki dla każdego, pokryłeś Fazę 14.

### Gdzie rozwój sterowany ewaluacją zawodzi

- **Brak punktu odniesienia.** Ewaluacje bez ostatniego znanego dobrego są nieczytelne. Przechowuj punkty odniesienia.
- **LLM-sędzia bez ugruntowania.** Sędziowie też halucynują. Wzór CRITIC (Lekcja 05) — sędzia ugruntowuje na zewnętrznych narzędziach.
- **Przedoptymalizowanie do ewaluacji.** Optymalizacja pod ewaluację odbiega od użyteczności produkcyjnej. Rotuj przypadki.
- **Niestabilne ewaluacje.** Niedeterministyczne przypadki powodują fałszywe alarmy. Ustalaj ziarna, migawkuj stan.

## Build It

`code/main.py` to uprząż ewaluacyjna w stdlib:

- Rejestr przypadków z kategoriami (benchmark, niestandardowy, online).
- Skryptowy agent pod testem.
- Pętla evaluator-optimizer: proponuj, oceń, udoskonalaj, aż do zaliczenia lub maksymalnej liczby rund.
- Bramka CI: zagregowany wskaźnik zaliczeń + regresja względem punktu odniesienia.

Uruchom:

```
python3 code/main.py
```

Wynik: zaliczenie/porażka na przypadek, flaga regresji, werdykt bramki CI.

## Use It

- Pisz przypadki ewaluacyjne w tym samym repozytorium co kod agenta.
- Uruchamiaj je na każdym PR przez CI.
- Odrzucaj kompilację przy regresji.
- Śledź wskaźnik zaliczeń w czasie.
- Łącz każdą awarię produkcyjną z nowym przypadkiem.

## Ship It

`outputs/skill-eval-suite.md` buduje trójwarstwowy zestaw ewaluacyjny dla produktu agenta z bramkami CI i śledzeniem regresji.

## Exercises

1. Weź jedną ze swoich awarii produkcyjnych. Napisz przypadek ewaluacyjny, który ją odtwarza. Czy twój agent przechodzi go teraz?
2. Zbuduj rubrykę LLM-jako-sędzia dla swojej domeny z trzema wymiarami (faktyczność, ton, zakres). Ocen 50 sesji.
3. Podłącz zestaw ewaluacyjny do CI. Odrzuć kompilację przy >=5% regresji.
4. Dodaj metrykę wydajności trajektorii: ile kroków agent wykonał względem złotej trajektorii?
5. Mapuj każdą lekcję Fazy 14 na przypadek ewaluacyjny w swoim zestawie. Brakuje jakiegoś? To luka do zamknięcia.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| Static benchmark | "Gotowa ewaluacja" | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| Custom offline eval | "Ewaluacja domenowa" | LLM-jako-sędzia / exec / trajektoria na kształcie twojego produktu |
| Online eval | "Ewaluacja produkcyjna" | Odtworzenie sesji, alerty zabezpieczeń, śledzenie kosztu/opóźnienia |
| Evaluator-optimizer | "Proponuj-oceń-udoskonalaj" | Iteruj, aż sędzia przejdzie |
| CI gate | "Bloker scalania" | Odrzuć kompilację przy regresji ewaluacji |
| Baseline | "Ostatni znany dobry" | Wynik referencyjny do wykrywania regresji |
| Trajectory efficiency | "Kroki nad złotem" | Liczba kroków agenta podzielona przez minimum ludzkiego eksperta |

## Further Reading

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "zacznij prosto, optymalizuj z ewaluacjami"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — wyselekcjonowany benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — benchmark użycia narzędzi
- [Langfuse docs](https://langfuse.com/) — ewaluacje + odtwarzanie sesji w praktyce