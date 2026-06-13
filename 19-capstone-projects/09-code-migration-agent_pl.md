# Capstone 09 — Agent Migracji Kodu (Aktualizacja Języka / Środowiska Wykonawczego na Poziomie Repozytorium)

> MigrationBench Amazonu (Java 8 do 17) i migrator Py2-do-Py3 Google App Engine wyznaczyły standard 2026. Moderne's OpenRewrite wykonuje deterministyczne przepisywania AST na skalę. Grit celuje w ten sam problem z DSL w stylu codemod. Produkcyjny wzór łączy oba: deterministyczne podłoże dla bezpiecznych przepisywań plus warstwę agenta dla niejednoznacznych przypadków, sandbox dla budów na gałęzi i harness testowy, który robi się zielony przed otwarciem PR. Capstone polega na migracji 50 prawdziwych repozytoriów i opublikowaniu wskaźnika zdawalności z taksonomią awarii.

**Type:** Capstone
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## Problem

Migracja kodu na dużą skalę jest jedną z najczystszych produkcyjnych aplikacji agentów programistycznych w 2026 roku. Ground truth jest oczywisty (czy zestaw testów przechodzi po migracji?), nagrody są realne (migracja floty Java-8 to projekt na skalę etatów), a benchmarki są publiczne (podzbiór 50 repozytoriów MigrationBench). Moderne's OpenRewrite obsługuje deterministyczną stronę. Warstwa agenta obsługuje wszystko, czego przepisy OpenRewrite nie mogą: niejednoznaczne przepisywania, dryft systemu budowania, długi ogon składni, przełamania zależności tranzytywnych.

Zbudujesz agenta, który bierze repozytorium Java 8 (lub Python 2) i produkuje zmigrowaną gałąź z zielonym CI. Zmierzysz wskaźnik zdawalności, zachowanie pokrycia testów, koszt na repozytorium i zbudujesz taksonomię awarii. Porównanie obok baseline tylko-deterministycznego mówi, gdzie faktycznie leży wartość agenta.

## Koncepcja

Potok ma dwie warstwy. **Deterministyczne podłoże** (OpenRewrite dla Javy, libcst dla Pythona) uruchamia większość mechanicznych przepisywań bezpiecznie: importy, sygnatury metod, edycje null-safety, try-with-resources, zamiany przestarzałych API. Jest szybkie i produkuje audytowalne diffy. **Warstwa agenta** (OpenAI Agents SDK lub LangGraph na Claude Opus 4.7 i GPT-5.4-Codex) obsługuje przypadki, których przepisy nie mogą: aktualizacje plików budowania (Maven/Gradle/pyproject), konflikty zależności tranzytywnych, flaky testy, niestandardowe adnotacje.

Każde repozytorium otrzymuje sandbox Daytona z preinstalowanym docelowym środowiskiem wykonawczym. Agent iteruje: uruchom budowę, sklasyfikuj awarie, zastosuj naprawę, uruchom ponownie. Twarde limity: 30 minut na repozytorium, $8 na repozytorium, 20 kroków agenta. Jeśli wszystkie testy przechodzą, a różnica pokrycia nie jest ujemna, gałąź otwiera PR. Jeśli nie, repozytorium trafia do klasy awarii z dowodami.

Taksonomia awarii jest rezultatem. W 50 repozytoriach, co się zepsuło? Zależności tranzytywne? Niestandardowe adnotacje? Wersja narzędzia budowania? Flaky testy niezwiązane z migracją? Każda klasa dostaje licznik i przykładowy diff. Przyszli autorzy przepisów mogą celować w top 3.

## Architektura

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## Stack

- Deterministyczne podłoże: OpenRewrite (Java) lub libcst (Python)
- Agent: OpenAI Agents SDK lub LangGraph na Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox: Daytona devcontainery na gałąź, preinstalowane docelowe środowisko wykonawcze (Java 17 / Python 3.12)
- Systemy budowania: Maven, Gradle, uv (Python)
- Benchmarki: Amazon MigrationBench podzbiór 50 repozytoriów (Java 8 do 17), Google App Engine Py2-do-Py3 repos
- Harness testowy: równoległy runner, pokrycie przez Jacoco (Java) lub coverage.py (Python)
- Obserwowalność: Langfuse + pakiet śladu na repozytorium z każdym fragmentem diffa
- Kokpit: kokpit taksonomii awarii z licznikami na klasę i przykładowymi diffami

## Build It

1. **Przejście przepisów.** Uruchom najpierw przepisy OpenRewrite (Java) lub libcst (Python). Złap 70-80% migracji, które są mechaniczne. Zatwierdź jako commit "recipe".

2. **Próba budowania.** Sandbox Daytona: zainstaluj docelowe środowisko wykonawcze, uruchom budowę. Jeśli zielona, przejdź do testów. Jeśli czerwona, przekaż agentowi.

3. **Pętla agenta.** LangGraph z narzędziami: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. Agent klasyfikuje awarię (dep, składnia, test, narzędzie budowania) i stosuje ukierunkowaną naprawę. Uruchom ponownie.

4. **Ograniczenia budżetu.** 30 minut ściennego czasu na repozytorium, $8 kosztu, 20 kroków agenta. Każde przekroczenie zatrzymuje i archiwizuje pod "budget_exhausted" z bieżącym diffem.

5. **Bramka testów + pokrycia.** Po tym jak budowa stanie się zielona, uruchom zestaw testów. Porównaj pokrycie do bazowego repozytorium. Jeśli pokrycie spadło o więcej niż 2%, archiwizuj pod "coverage_regression".

6. **Otwarcie PR.** Po sukcesie, wypchnij gałąź, otwórz PR z diffem i podsumowaniem, które przepisy zostały zastosowane i które commity agent napisał.

7. **Taksonomia awarii.** Dla każdego nieudanego repozytorium, oznacz klasą: `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. Zbuduj kokpit.

8. **Przebieg 50 repozytoriów.** Wykonaj na podzbiorze MigrationBench. Raportuj wskaźnik zdawalności na klasę, koszt-na-repozytorium, zachowanie pokrycia i porównanie z tylko-deterministycznym baseline.

## Use It

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## Ship It

`outputs/skill-migration-agent.md` jest rezultatem. Mając repozytorium, wykonuje deterministyczne przepisy, a następnie pętlę agenta, aby wyprodukować zieloną zmigrowaną gałąź lub archiwizuje repozytorium pod klasą taksonomii.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Wskaźnik zdawalności MigrationBench | 50-repo podzbiór pass@1 |
| 20 | Zachowanie pokrycia testów | Średnia różnica pokrycia vs baza |
| 20 | Koszt na zmigrowane repozytorium | $/repo na przechodzących przebiegach |
| 20 | Integracja agenta / narzędzia deterministycznego | Frakcja napraw obsłużonych przez OpenRewrite vs napisanych przez agenta |
| 15 | Opracowanie analizy awarii | Kompletność taksonomii z przykładami |
| **100** | | |

## Ćwiczenia

1. Uruchom potok migracji z samym OpenRewrite (bez agenta). Porównaj wskaźnik zdawalności do pełnego potoku. Zidentyfikuj przypadki, gdzie sam agent robi różnicę.

2. Zaimplementuj kontrolę "lint-clean": po migracji, uruchom linter stylu (spotless dla Javy, ruff dla Pythona). Odrzuć PR, jeśli pojawią się nowe błędy lint. Zmierz wskaźnik zachowanego pokrycia, ale pogorszonego stylu.

3. Dodaj optymalizator "minimal-diff": po tym, jak gałąź agenta przejdzie testy, przytnij niepotrzebne zmiany drugim przejściem. Raportuj redukcję rozmiaru diff.

4. Rozszerz na trzecią migrację: Node 18 do Node 22. Wykorzystaj ponownie otoczkę sandboksa; zamień warstwę przepisów na niestandardowy codemod.

5. Zmierz czas-do-pierwszej-zielonej-budowy (TTFGB) jako metrykę UX. Cel: p50 poniżej 10 minut.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Deterministic substrate | "Silnik przepisów" | OpenRewrite / libcst: deklaratywne przepisywania AST z gwarancjami bezpieczeństwa |
| Codemod | "Program modyfikujący kod" | Reguła przepisywania, która zmienia kod źródłowy mechanicznie |
| Build drift | "Rozbieżność wersji narzędzi" | Subtelne zmiany zachowania Maven / Gradle / uv między głównymi wersjami |
| Failure class | "Wiadro taksonomii" | Oznaczony powód, dla którego repozytorium nie zmigrowało: dep, składnia, test, narzędzie budowania, budżet |
| Coverage delta | "Zachowanie pokrycia" | Zmiana % pokrycia testów od bazy do zmigrowanej gałęzi |
| Agent turn | "Runda wywołania narzędzia" | Jeden cykl plan -> działaj -> obserwuj w pętli agenta |
| Budget exhaustion | "Osiągnięcie limitu" | Repozytorium wyczerpało limit 30-min / $8 / 20 kroków bez przejścia |

## Dalsza lektura

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — the canonical 2026 benchmark
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — the deterministic substrate reference
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe authoring
- [Grit.io](https://www.grit.io) — alternate codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — the Agents SDK reference
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — alternate migration benchmark
- [libcst](https://github.com/Instagram/LibCST) — Python deterministic substrate
- [Daytona sandboxes](https://daytona.io) — reference per-branch sandbox