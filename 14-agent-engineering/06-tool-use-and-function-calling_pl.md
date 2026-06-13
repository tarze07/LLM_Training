# Użycie narzędzi i wywoływanie funkcji

> Toolformer (Schick et al., 2023) zapoczątkował samonadzorowaną adnotację narzędzi. Berkeley Function Calling Leaderboard V4 (Patil et al., 2025) ustawia poprzeczkę 2026: 40% agentyczne, 30% wieloturowe, 10% na żywo, 10% nie na żywo, 10% halucynacje. Pojedyncza tura jest rozwiązana. Pamięć, dynamiczne podejmowanie decyzji i długoterminowe łańcuchy narzędzi nie są.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## Learning Objectives

- Wyjaśnij samonadzorowany sygnał treningowy Toolformer: zachowuj adnotacje narzędzi tylko wtedy, gdy wykonanie zmniejsza stratę następnego tokena.
- Wymień pięć kategorii ewaluacyjnych BFCL V4 i co każda mierzy.
- Zaimplementuj rejestr narzędzi w stdlib z walidacją schematu, koercją argumentów i piaskownicą wykonania.
- Zdiagnozuj trzy otwarte problemy 2026: długoterminowe łańcuchowanie narzędzi, dynamiczne podejmowanie decyzji i pamięć.

## The Problem

Wczesne użycie narzędzi pytało: czy model może przewidzieć poprawny fragment funkcji? Nowoczesne użycie narzędzi pyta: czy model może łączyć narzędzia przez 40 kroków, z pamięcią, z częściową obserwowalnością, z odzyskiwaniem po awariach narzędzi, bez halucynowania narzędzi, które nie istnieją?

Toolformer ustanowił linię bazową: modele mogą nauczyć się, kiedy wywoływać narzędzia z samonadzorem. BFCL V4 definiuje cel ewaluacyjny 2026. Luka między nimi to przestrzeń, w której żyją produkcyjni agenci.

## The Concept

### Toolformer (Schick et al., NeurIPS 2023)

Pomysł: pozwól modelowi adnotować własny korpus przedtreningowy kandydackimi wywołaniami API. Dla każdego kandydata, wykonaj go. Zachowaj adnotację tylko wtedy, gdy dołączenie wyniku narzędzia zmniejsza stratę na następnym tokenie. Dostrój na przefiltrowanym korpusie.

Narzędzia objęte: kalkulator, system QA, wyszukiwarki, tłumacz, kalendarz. Sygnał samonadzoru dotyczy czysto tego, czy narzędzie pomaga przewidywać tekst — żadnych ludzkich etykiet.

Wynik skali: użycie narzędzi pojawia się w skali. Mniejsze modele cierpią z powodu adnotacji narzędzi; większe modele zyskują. Dlatego graniczne modele 2026 mają wbudowane silne użycie narzędzi, podczas gdy większość modeli 7B potrzebuje jawnego dostrajania do użycia narzędzi, aby być niezawodnymi.

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025)

BFCL to de facto ewaluacja 2026. Skład V4:

- **Agencyczne (40%)** — pełne trajektorie agenta: pamięć, wieloturowość, dynamiczne decyzje.
- **Wieloturowe (30%)** — interaktywne rozmowy z łańcuchami narzędzi.
- **Na żywo (10%)** — prawdziwe prompty od użytkowników (trudniejsza dystrybucja).
- **Nie na żywo (10%)** — syntetyczne przypadki testowe.
- **Halucynacje (10%)** — wykryj, gdy nie należy wywoływać żadnego narzędzia.

V3 wprowadziło ewaluację opartą na stanie: po sekwencji narzędzi sprawdź rzeczywisty stan API (np. "czy plik został utworzony?") zamiast dopasowywać AST wywołań narzędzi. V4 dodało kategorie wyszukiwania internetowego, pamięci i wrażliwości na format.

Kluczowe ustalenie 2026: wywoływanie funkcji w pojedynczej turze jest prawie rozwiązane. Awarie koncentrują się w pamięci (przenoszenie kontekstu między turami), dynamicznym podejmowaniu decyzji (wybieranie narzędzi na podstawie wcześniejszych wyników), długoterminowych łańcuchach (dryf po 20+ krokach) i wykrywaniu halucynacji (odmowa wywołania, gdy żadne narzędzie nie pasuje).

### Schemat narzędzia

Każdy dostawca ma schemat. Różnią się w szczegółach, ale dzielą ten sam kształt:

```
name: string
description: string (co robi, kiedy go użyć)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic używa `input_schema` bezpośrednio. OpenAI używa `function.parameters`. Oba akceptują JSON Schema. Opisy są nośne — model czyta je, aby wybrać właściwe narzędzie. Złe opisy narzędzi to główna przyczyna awarii "wybrano złe narzędzie".

### Walidacja argumentów

Nie ufaj żadnemu wywołaniu narzędzia. Waliduj:

1. **Koercja typów.** Model może zwrócić string "5", gdzie schemat mówi int. Koercyj, jeśli jednoznaczne; odrzuć, jeśli nie.
2. **Walidacja enum.** Jeśli schemat mówi `status in {"open", "closed"}` a model emituje `"in_progress"`, odrzuć z opisowym błędem.
3. **Wymagane pola.** Brakujące wymagane pole -> natychmiastowa obserwacja błędu z powrotem do modelu, a nie crash.
4. **Walidacja formatu.** Daty, emaile, URL-e — waliduj konkretnymi parserami, a nie regexem.

Każda nieudana walidacja powinna zwrócić ustrukturyzowaną obserwację, aby model mógł spróbować ponownie z poprawnym kształtem.

### Równoległe wywołania narzędzi

Nowocześni dostawcy obsługują równoległe wywołania narzędzi w jednej turze asystenta. Pętla:

1. Model emituje 3 wywołania narzędzi z odrębnymi `tool_use_id`.
2. Środowisko wykonawcze je wykonuje (równolegle, jeśli niezależne).
3. Każdy wynik wraca jako blok `tool_result` skorelowany przez `tool_use_id`.

Reguła inżynieryjna: traktuj identyfikatory korelacji jako nośne. Zamień je, a dostaniesz błędne kierowanie narzędzie-do-błędnego-wyniku.

### Piaskownica

Wykonanie narzędzia jest granicą piaskownicy. Zobacz Lekcję 09 dla szczegółów. Krótka wersja: każde narzędzie powinno określać powierzchnię odczytu/zapisu, dostęp do sieci, limit czasu, limit pamięci. Generyczne `run_shell(cmd)` to czerwona flaga; specyficzne `git_status()` jest bezpieczniejsze.

```figure
tool-routing
```

## Build It

`code/main.py` implementuje rejestr narzędzi o kształcie produkcyjnym:

- Walidator podzbioru JSON Schema (tylko stdlib).
- Rejestracja narzędzi z opisem, schematem wejściowym, limitem czasu i executorem.
- Koercja argumentów i walidacja enum.
- Równoległe wysyłanie narzędzi z identyfikatorami korelacji.
- Obserwacje błędów jako ustrukturyzowane stringi.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje mini agenta wywołującego trzy narzędzia w jednej turze, z jednym celowo źle sformułowanym wywołaniem, które jest odrzucane z opisowym błędem, na którym model może działać.

## Use It

Każdy dostawca ma własny schemat narzędzi — Anthropic, OpenAI, Gemini, Bedrock. Użyj warstwy translacyjnej (OpenAI Agents SDK, Vercel AI SDK, adapter narzędzi LangChain), jeśli potrzebujesz wielu dostawców. BFCL to referencyjny benchmark — uruchom go przeciwko swojemu agentowi przed wdrożeniem, jeśli użycie narzędzi jest kluczowe dla produktu.

## Ship It

`outputs/skill-tool-registry.md` generuje katalog narzędzi, schemat i rejestr dla danej domeny zadania. Zawiera kontrole jakości opisu (czy opis każdego narzędzia mówi modelowi, kiedy go użyć?).

## Exercises

1. Dodaj narzędzie "no-op", które pozwala modelowi wyraźnie odmówić użycia jakiegokolwiek innego narzędzia. Zmierz na teście halucynacji podobnym do BFCL.
2. Zaimplementuj koercję argumentów dla int-jako-string i float-jako-string. Gdzie koercja zaczyna ukrywać prawdziwe błędy?
3. Dodaj limit czasu per-narzędzie i wyłącznik obwodu (odmawiaj narzędzia przez 60s po 3 kolejnych awariach). Co to zmienia w sposobie odzyskiwania modelu?
4. Przeczytaj opis BFCL V4. Wybierz jedną kategorię (np. "wieloturowe") i uruchom 10 przykładowych promptów przez swojego agenta. Raportuj wskaźnik zaliczeń.
5. Przenieś walidator stdlib do Pydantic lub Zod. Co Pydantic/Zod złapało, czego zabawka nie złapała?

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|------|----------------|------------------------|
| Function calling | "Użycie narzędzi" | Wywołanie narzędzia ze strukturalnym wyjściem i zweryfikowanym schematem |
| Toolformer | "Samonadzorowana adnotacja narzędzi" | Schick 2023 — zachowaj wywołania narzędzi, których wyniki redukują stratę następnego tokena |
| BFCL | "Berkeley Function Calling Leaderboard" | Benchmark 2026: 40% agentyczne, 30% wieloturowe, 10% na żywo, 10% nie na żywo, 10% halucynacje |
| Schemat narzędzia | "Sygnatura funkcji dla modelu" | nazwa, opis, JSON Schema argumentów |
| tool_use_id | "Identyfikator korelacji" | Łączy wywołanie narzędzia z jego wynikiem; niezbędny dla równoległego wysyłania |
| Wykrywanie halucynacji | "Wiedz, kiedy nie wywoływać" | Kategoria V4: odmów wywołania, gdy żadne narzędzie nie pasuje |
| Koercja argumentów | "Naprawa string-na-int" | Wąskie poprawki dla przewidywalnych niedopasowań schematu; odrzuć, jeśli niejednoznaczne |
| Piaskownica | "Granica wykonania narzędzia" | Powierzchnia odczytu/zapisu per-narzędzie, sieć, limit czasu, limit pamięci |

## Further Reading

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — samonadzorowana adnotacja narzędzi
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — benchmark ewaluacyjny 2026
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — produkcyjny schemat narzędzi w Claude Agent SDK
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — typ narzędzia funkcji i Guardrails

(End of file - total 140 lines)