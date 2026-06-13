# Współdzielona Pamięć i Wzorce Tablicowe

> W systemach wieloagentowych w 2026 roku współistnieją dwa podejścia: **pula wiadomości** (każdy widzi wiadomości wszystkich, jak w AutoGen GroupChat lub MetaGPT) oraz **tablica z subskrypcją** (agenci subskrybują istotne zdarzenia, jak w Context-Aware MCP lub frameworku Matrix). Oba są jedyną stanową częścią systemu wieloagentowego — co oznacza, że oba są miejscem, gdzie żyją interesujące błędy. Referencyjnym trybem awarii jest **zatrucie pamięci**: jeden agent halucynuje "fakt", inni agenci traktują go jako zweryfikowany, a dokładność pogarsza się stopniowo w sposób znacznie trudniejszy do debugowania niż natychmiastowa awaria. Ta lekcja buduje obie struktury z stdlib, wstrzykuje atak zatrucia i pokazuje trzy mitygacje, które faktycznie działają w produkcji.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Problem

Systemy wieloagentowe potrzebują miejsca, w którym agenci mogą dzielić się faktami. Dosłowną opcją jest "przekazuj wszystko w wiadomościach" — ale to wynajduje na nowo współdzielony stan z dodatkowym kopiowaniem. Inną jest "daj wszystkim globalny log" — ale globalne logi rosną bez ograniczeń i łatwo ulegają zatruciu. Trzecią jest "projektuj widok na agenta" — skalowalne, ale ciężkie schematowo.

Gdy jeden z agentów halucynuje i zapisuje halucynację do współdzielonego stanu, każdy downstreamowy agent czytający ten stan przejmuje halucynację jako fakt. Zanim człowiek zauważy, łańcuch rozumowania jest pięć kroków głęboki, a pierwotną przyczyną jest trzecia kiedykolwiek zapisana wiadomość. Debugowanie pogarszania się dokładności w systemach wieloagentowych jest trudniejsze niż debugowanie awarii.

To jest zatrucie pamięci. Jest to druga najlepiej udokumentowana rodzina awarii w taksonomii MAST (Cemri i in., arXiv:2503.13657) i jest strukturalna: każdy projekt współdzielonej pamięci bez proweniencji i niezapisywalnego weryfikatora prędzej czy później go wykaże.

## Koncepcja

### Dwie główne topologie

**Pełna pula wiadomości.** Każdy agent czyta każdą wiadomość. AutoGen GroupChat i MetaGPT używają tego. Prosta, przejrzysta, możliwa do inspekcji, ale nie skaluje się powyżej ~10 agentów, ponieważ kontekst każdego agenta wypełnia się pracą innych agentów.

```
agent-A ──zapis──▶ ┌────────────────┐ ◀──odczyt── agent-D
                   │ pula          │
agent-B ──zapis──▶ │ wiadomości    │ ◀──odczyt── agent-E
                   │ (globalny log)│
agent-C ──zapis──▶ └────────────────┘ ◀──odczyt── agent-F
```

**Tablica z subskrypcją.** Agenci deklarują zainteresowanie tematami; podłoże kieruje tylko istotne wiadomości. CA-MCP (arXiv:2601.11595) i zdecentralizowany framework Matrix (arXiv:2511.21686) używają tego. Skaluje się dalej, ale wymaga wcześniejszego projektu schematu, aby subskrypcje były znaczące.

```
                   ┌─ temat: ceny ──┐
agent-A ──pub───▶ │                 │ ──▶ agent-D (subskrybent)
                   ├─ temat: zamówienia ─┤
agent-B ──pub───▶ │                 │ ──▶ agent-E (subskrybent)
                   ├─ temat: alerty ─┤
agent-C ──pub───▶ │                 │ ──▶ agent-F (subskrybent)
                   └─────────────────┘
```

### Kiedy które wygrywa

- **Pełna pula** wygrywa, gdy agentów jest mało (< 10), są heterogeniczni, a konwersacja jest krótkoterminowa. Rozumowanie o tym, kto co powiedział, jest banalne, gdy wszyscy widzą wszystko.
- **Tablica** wygrywa, gdy agentów jest wielu, jednorodni w roli, ale liczni w instancjach (roje), a konwersacja jest długoterminowa. Routing oszczędza koszt tokenów i zanieczyszczenie kontekstu.

Systemy produkcyjne często mieszają: mała pełna pula na górze (warstwa planowania), tablice poniżej (warstwa pracowników).

### Zatrucie pamięci, w jednym scenariuszu

Trzech agentów pracuje nad zadaniem badawczym. Agent A to agent wyszukiwania. Agent B to agent podsumowujący. Agent C to analityk.

1. A pobiera stronę i zapisuje wiadomość do współdzielonego stanu: "Badanie raportuje 42% poprawę dokładności."
2. Pobrana strona faktycznie mówiła "4.2% poprawy." A zrobił halucynację dziesiętną.
3. B, czytając współdzielony stan, zapisuje: "Duży 42% wzrost dokładności (źródło: A)."
4. C, czytając współdzielony stan, zapisuje: "Zalecam adopcję — 42% wzrost jest transformacyjny."
5. Końcowy raport cytuje liczbę 42%, która nigdy nie istniała.

Żaden agent się nie zawiesił. Żaden test nie zawiódł. System "działał." Halucynacja przeniosła się z kontekstu jednego agenta do rozumowania każdego downstreamowego agenta przez współdzielony stan.

### Dlaczego to jest strukturalne

Bez współdzielonego stanu, halucynacja agenta A pozostaje w kontekście A. Downstreamowi agenci ponownie by pobrali lub wyprowadzili i mogliby wychwycić błąd. Z naiwnym współdzielonym stanem, kontekst A staje się kontekstem wszystkich, a halucynacja zostaje wyprana w fakt.

Problemem nie jest sam współdzielony stan — to współdzielony stan **bez proweniencji i bez niezależnego weryfikatora**. Trzy mitygacje rozwiązują to:

1. **Przypisz proweniencję do każdego zapisu.** Każdy wpis we współdzielonym stanie rejestruje, kto go napisał, kiedy, pod jakim promptem i (jeśli dotyczy) jakie źródło agent cytował. Downstreamowi agenci czytają ze sceptycyzmem kluczowanym do proweniencji.
2. **Wersjonuj zapisy; traktuj je jako dopisywalne.** Korekta to nowy wpis, który zastępuje stary, a nie aktualizacja w miejscu. Ślad audytu jest zachowany.
3. **Zachowaj co najmniej jednego agenta, który nie może zapisywać do współdzielonego stanu.** Tylko-do-odczytu agent weryfikator próbkuje wpisy, ponownie pobiera źródła i flaguje niespójności. Ponieważ nie może zapisywać do puli, nie może być zatruty przez pulę.

### Precedens tablicy (Hayes-Roth, 1985)

Wzorzec tablicy poprzedza agenty LLM o cztery dekady. Hayes-Roth (1985, "A Blackboard Architecture for Control") opisał wyspecjalizowane Źródła Wiedzy, które obserwują globalną tablicę, wnoszą częściowe rozwiązania i wyzwalają inne źródła. Tablica z 2026 roku (CA-MCP, Matrix) to ten sam wzorzec z agentami LLM jako Źródłami Wiedzy i blobami JSON jako częściowymi rozwiązaniami. Stara literatura ma udokumentowane rozwiązania dla rywalizacji o zapis, oportunistycznego sterowania i spójności, które nowoczesne systemy odkrywają na nowo.

### Projekcja a pełny widok

Czysta tablica daje każdemu subskrybentowi tę samą projekcję (zakres tematyczny). Bardziej agresywnym projektem jest **projekcja na agenta**: każdy agent otrzymuje widok dostosowany do swojej roli. Reduktory stanu LangGraph są kanoniczną implementacją z 2026 roku — funkcja reduktora składa globalny stan w wycinek specyficzny dla roli.

Projekcja na agenta skaluje się dalej, ale potrzebuje schematu. Bez niego odbudowujesz ad-hoc projekcję w prompcie każdego agenta.

### Wzorce rywalizacji o zapis

Wielu agentów zapisujących jednocześnie to problem współbieżności, nie tylko problem LLM. Trzy wzorce działają:

- **Sekwencyjny zapisujący (pojedynczy producent).** Wszystkie zapisy przechodzą przez jednego agenta koordynatora, który serializuje. Proste, ale wąskie gardło.
- **Optymistyczna współbieżność z wersjonowaniem.** Każdy wpis ma wersję; zapisujący zawodzą przy niezgodności wersji i ponawiają. Klasyczna technika baz danych.
- **Partycjonowanie tematyczne.** Różni agenci posiadają różne tematy. Brak rywalizacji między tematami. Wymaga zaprojektowanych granic partycji.

Większość frameworków z 2026 roku domyślnie używa sekwencyjnego zapisującego, ponieważ wywołania LLM są na tyle wolne, że rywalizacja jest rzadka, a wąskie gardło nie boli.

### Niezapisywalny weryfikator

Najbardziej kluczową mitygacją jest weryfikator tylko-do-odczytu. Zasady implementacji:

- Weryfikator współdzieli stan z zespołem (czyta tablicę lub pulę).
- Weryfikator nie ma uchwytu do zapisu do współdzielonego stanu — tylko do osobnego kanału weryfikacji.
- Weryfikator niezależnie pobiera źródła cytowane w zapisach. Flaguje niezgodności.
- Wyniki weryfikatora są kierowane do człowieka lub osobnego agenta decyzyjnego, nigdy z powrotem do puli.

Bez tego rozdzielenia, wyniki weryfikatora stają się nowymi wpisami w puli, co oznacza, że zatruta pula zatruwa weryfikatora, co zatruwa jego weryfikacje.

## Zbuduj To

`code/main.py` implementuje obie topologie w stdlib Python plus zabawkowy atak zatrucia i trzy mitygacje.

- `MessagePool` — bezpieczny wątkowo, dopisywalny log z pełnym odczytem.
- `Blackboard` — pub/sub kluczowany tematami z subskrypcjami na agenta.
- `ProvenanceEntry` — każdy zapis rejestruje (writer, timestamp, prompt_hash, source_uri).
- `PoisoningScenario` — uruchamia trzyagentowe zadanie badawcze, w którym agent A halucynuje dziesiętną. Wyświetla końcowy raport.
- `Verifier` — agent tylko-do-odczytu, który ponownie pobiera źródła i flaguje niespójności. Uruchamia ten sam scenariusz z obecnym weryfikatorem.

Uruchom:

```
python3 code/main.py
```

Oczekiwane wyjście:
- Uruchomienie 1 (bez weryfikatora): halucynacja 42% propaguje się do końcowego raportu.
- Uruchomienie 2 (z weryfikatorem): weryfikator flaguje niespójność, pula jest oznaczona "flagged", końcowy raport zawiera wycofanie.

## Użyj Tego

`outputs/skill-memory-auditor.md` to umiejętność, która audytuje projekt współdzielonej pamięci dowolnego systemu wieloagentowego pod kątem proweniencji, wersjonowania i separacji weryfikatora. Uruchom ją na nowych architekturach wieloagentowych przed produkcją.

## Wdróż To

Dla każdego projektu współdzielonej pamięci:

- Rejestruj proweniencję przy każdym zapisie: `(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`.
- Uczyń log dopisywalnym. Korekty to nowe wpisy odnoszące się do zastępowanego.
- Wdróż co najmniej jednego weryfikatora tylko-do-odczytu z niezależnym dostępem do źródeł.
- Kieruj wyniki weryfikatora do osobnego kanału, nie z powrotem do współdzielonej puli.
- Loguj stosunek zapisów, które są zastąpieniami — rosnący stosunek to wczesny dowód wzorców halucynacji.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że uruchomienie 1 propaguje halucynację, a uruchomienie 2 ją wychwytuje.
2. Dodaj drugą halucynację: agent B wymyśla rozmiar zestawu danych. Weryfikator powinien wychwycić obie bez ręcznego dostrajania do którejkolwiek.
3. Zamień pełną pulę na tablicę z partycjami tematycznymi (`prices`, `summaries`, `analyses`). Które scenariusze zatrucia partycjonowanie tematyczne utrudnia, a którym nie pomaga?
4. Przeczytaj Hayes-Roth (1985, "A Blackboard Architecture for Control"). Zidentyfikuj dwa wzorce sterowania z publikacji, które nie są omówione w tej lekcji, a z których systemy z 2026 roku skorzystałyby.
5. Przeczytaj CA-MCP (arXiv:2601.11595). Zmapuj jego Shared Context Store na klasę MessagePool lub Blackboard w `code/main.py`. Które prymitywy CA-MCP dodaje na wierzchu?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|----------------|------------------------|
| Pula wiadomości | "Współdzielona historia czatu" | Dopisywalny log, który każdy agent czyta. Pełna przejrzystość, słabe skalowanie. |
| Tablica | "Współdzielona przestrzeń robocza" | Pub/sub kluczowany tematami. Agenci subskrybują istotne tematy. Skaluje się dalej. |
| Proweniencja | "Kto co napisał" | Metadane każdego zapisu: pisarz, znacznik czasu, prompt, źródła. |
| Zatrucie pamięci | "Halucynacje się rozprzestrzeniają" | Błąd jednego agenta wchodzi do współdzielonego stanu, downstreamowi agenci przejmują go jako fakt. |
| Dopisywalność | "Brak aktualizacji w miejscu" | Korekty to nowe wpisy, które zastępują. Zachowuje ślad audytu. |
| Niezapisywalny weryfikator | "Niezależny audytor" | Agent tylko-do-odczytu, który ponownie pobiera źródła i flaguje niespójności. |
| Projekcja | "Zakresowy widok" | Widok na agenta obliczany ze stanu globalnego. Reduktory LangGraph to kanoniczny przypadek. |
| Źródło Wiedzy | "Agent specjalista" | Termin Hayes-Roth z 1985 roku na uczestnika tablicy. |

## Dalsza Literatura

- [Cemri i in. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — taksonomia MAST; zatrucie pamięci to podrodzina awarii koordynacji
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — Shared Context Store dla skoordynowanych serwerów MCP
- [Matrix — zdecentralizowany framework wieloagentowy](https://arxiv.org/abs/2511.21686) — tablica oparta na kolejce wiadomości bez centralnego orkiestratora
- [Stan i reduktory LangGraph](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — wzorzec projekcji na agenta w produkcji
- [Anthropic — Jak zbudowaliśmy nasz wieloagentowy system badawczy](https://www.anthropic.com/engineering/multi-agent-research-system) — notatki o proweniencji i weryfikacji z produkcyjnego wdrożenia