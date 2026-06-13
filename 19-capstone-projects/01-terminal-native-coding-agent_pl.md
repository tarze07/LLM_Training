# Capstone 01 — Terminalowo-Natywny Agent Programistyczny

> Rok 2026 ustalił kształt agenta programistycznego. Harness TUI, stanowy plan, sandboksowana powierzchnia narzędziowa, pętla planowania, działania, obserwacji i odtwarzania. Claude Code, Cursor 3 i OpenCode wyglądają tak samo z dystansu. Ten capstone polega na zbudowaniu takiego agenta od początku do końca — CLI na wejściu, pull request na wyjściu — i zmierzeniu go względem mini-swe-agent i Live-SWE-agent na SWE-bench Pro. Dowiesz się, że trudną częścią nie jest wywołanie modelu, ale pętla narzędziowa, sandbox i ograniczenie kosztów w 50-zwojowym przebiegu.

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (eval scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and protocols), Phase 14 (agents), Phase 15 (autonomous systems), Phase 17 (infrastructure)
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 hours

## Problem

Agenci programistyczni stali się dominującą kategorią aplikacji AI w 2026 roku. Claude Code (Anthropic), Cursor 3 z Composer 2 i Agent Tabs (Cursor), Amp (Sourcegraph), OpenCode (112k gwiazdek), Factory Droids i Google Jules dostarczają wariacje tej samej architektury: terminalowy harness, autoryzowaną powierzchnię narzędziową, sandbox i pętlę planuj-działaj-obserwuj zbudowaną wokół modelu granicznego. Granica jest wąska — Live-SWE-agent osiągnął 79,2% na SWE-bench Verified z Opus 4.5 — ale inżynieryjne rzemiosło jest szerokie. Większość trybów awarii to nie błędy modelu. To niestabilność pętli narzędziowej, zatrucie kontekstu, niekontrolowane koszty tokenów i destrukcyjne operacje na systemie plików.

Nie możesz zrozumieć tych agentów z zewnątrz. Musisz go zbudować, zobaczyć, jak pętla się wysypuje przy 47. kroku, gdy ripgrep zwraca 8MB dopasowań, i przebudować warstwę przycinania. O to chodzi w tym capstone.

## Koncepcja

Harness ma cztery powierzchnie. **Plan** utrzymuje stanowy obiekt w stylu TodoWrite, który model przepisuje w każdym kroku. **Act** wysyła wywołania narzędzi (czytaj, edytuj, uruchom, szukaj, git). **Observe** przechwytuje stdout/stderr/kody wyjścia, przycina i przekazuje podsumowanie z powrotem. **Recover** obsługuje błędy narzędzi bez wysadzania okna kontekstu lub zapętlania się. Kształt z 2026 roku dodaje jeszcze jedną rzecz: **hooki**. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop` i `PreCompact` — konfigurowalne punkty rozszerzeń, w których operator wstrzykuje politykę, telemetrię i zabezpieczenia.

Sandbox to E2B lub Daytona. Każde zadanie działa w świeżym devcontainerze z zamontowanym do odczytu-zapisu git worktree. Harness nigdy nie dotyka systemu plików hosta. Worktree jest usuwany po sukcesie lub porażce. Kontrola kosztów jest egzekwowana na trzech poziomach: limit tokenów na krok, budżet dolarowy na sesję i twardy limit kroków (zazwyczaj 50). Warstwa obserwowalności to spany OpenTelemetry z semantycznymi konwencjami GenAI, wysyłane do samodzielnie hostowanego Langfuse.

## Architektura

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## Stack

- Środowisko wykonawcze harnessu: Bun 1.2 + Ink 5 (React-in-terminal)
- Dostęp do modelu: OpenRouter unified API z Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (dla najtrudniejszych zadań)
- Transport narzędzi: Model Context Protocol StreamableHTTP (MCP 2026 revision)
- Sandbox: E2B sandboxy (JS SDK) lub Daytona devcontainery
- Wyszukiwanie w kodzie: ripgrep subprocess, tree-sitter parsery dla 17 języków (pre-kompilowane)
- Izolacja: `git worktree add` na zadanie, czyszczenie po sukcesie/porażce
- Harness ewaluacyjny: SWE-bench Pro (zweryfikowany podzbiór) + Terminal-Bench 2.0 + własny zestaw 30 zadań
- Obserwowalność: OpenTelemetry SDK z `gen_ai.*` semconv → samodzielnie hostowany Langfuse
- Publikacja PR: GitHub App z precyzyjnym tokenem, zakres ograniczony do docelowego repozytorium

## Build It

1. **TUI i pętla komend.** Stwórz szkielet projektu Bun z Ink. Akceptuj `agent run <repo> "<task>"". Wyświetl podzielony widok: panel planu (góra), strumień wywołań narzędzi (środek), budżet tokenów (dół). Dodaj anulowanie przez Ctrl-C, które uruchamia hook `SessionEnd` przed wyjściem.

2. **Stan planu.** Zdefiniuj typowany schemat TodoWrite (elementy oczekujące / w_trakcie / zrobione z notatkami). Model przepisuje cały stan w każdym kroku jako wywołanie narzędzia — nie pozwalaj na mutację przyrostową. Zapisz plan do `.agent/state.json`, aby awarie mogły być wznawiane.

3. **Powierzchnia narzędzi.** Zdefiniuj sześć narzędzi: `read_file`, `edit_file` (z podglądem diff), `ripgrep`, `tree_sitter_symbols`, `run_shell` (z timeoutem), `git` (status/diff/commit/push). Udostępnij przez MCP StreamableHTTP, aby harness był niezależny od transportu. Każde narzędzie zwraca przycięte wyjście (limit 4k tokenów na wywołanie).

4. **Otoczka sandboksa.** Każde zadanie uruchamia sandbox E2B. `git worktree add -b agent/$TASK_ID` nową gałąź. Wszystkie wywołania narzędzi wykonują się wewnątrz sandboksa. System plików hosta jest nieosiągalny.

5. **Hooki.** Zaimplementuj wszystkie osiem typów hooków z 2026 roku. Podłącz co najmniej cztery hooki autorstwa użytkownika: (a) `PreToolUse` strażnik przed destrukcyjnymi komendami blokujący `rm -rf` poza worktree, (b) `PostToolUse` rozliczanie tokenów, (c) `SessionStart` inicjalizacja budżetu, (d) `Stop` zapisuje końcowy pakiet śledzenia.

6. **Pętla ewaluacyjna.** Sklonuj 30-zadaniowy podzbiór SWE-bench Pro Python. Uruchom swój harness na każdym. Porównaj z mini-swe-agent (minimalne baseline) na pass@1, kroki-na-zadanie i $-na-zadanie. Zapisz wyniki do `eval/results.jsonl`.

7. **Kontrola kosztów.** Twarde ograniczenia: 50 kroków, 200k kontekstu, $5 na zadanie. Hook `PreCompact` podsumowuje starsze kroki do bloku poprzedniego stanu w momencie osiągnięcia 150k, zwalniając miejsce na nowe obserwacje bez utraty planu.

8. **Publikacja PR.** Po sukcesie, ostatnim krokiem jest `git push` + wywołanie API GitHub, które otwiera PR z planem i podsumowaniem diffa w treści.

## Use It

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Ship It

Umiejętność będąca rezultatem znajduje się w `outputs/skill-terminal-coding-agent.md`. Mając ścieżkę repozytorium i opis zadania, uruchamia pełną pętlę planuj-działaj-obserwuj w sandboksie i zwraca URL PR oraz pakiet śledzenia. Rubryka dla tego capstone:

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | Twój harness vs mini-swe-agent na 30 dopasowanych zadaniach Python |
| 20 | Przejrzystość architektury | Separacja plan/act/observe, powierzchnia hooków, schemat narzędzi — oceniane względem układu Live-SWE-agent |
| 20 | Bezpieczeństwo | Testy ucieczki z sandboksa, monity o uprawnienia, strażnik przed destrukcyjnymi komendami przechodzi red-team |
| 20 | Obserwowalność | Kompletność śledzenia (100% wywołań narzędzi objętych spanem), rozliczanie tokenów na krok |
| 15 | UX deweloperski | Cold-start < 2s, odtwarzanie po awarii wznawia plan, Ctrl-C anuluje mid-tool czysto |
| **100** | | |

## Ćwiczenia

1. Zamień model bazowy z Claude Sonnet 4.7 na Qwen3-Coder-30B serwowany na vLLM. Porównaj pass@1 i $-na-zadanie. Raportuj, gdzie otwarty model osiąga gorsze wyniki.

2. Dodaj sub-agenta `reviewer`, który czyta diff przed publikacją PR i może zażądać pętli poprawek. Zmierz, czy fałszywie pozytywne recenzje obniżają wskaźnik zdawalności SWE-bench poniżej baseline pojedynczego agenta (wskazówka: zazwyczaj tak).

3. Przeprowadź test wytrzymałościowy sandboksa: napisz zadanie, które próbuje `curl` zewnętrzny URL i zadanie, które pisze poza worktree. Potwierdź, że oba są blokowane przez hook PreToolUse. Zaloguj próby.

4. Zaimplementuj podsumowanie `PreCompact` z mniejszym modelem (Haiku 4.5). Zmierz, jak dużo wierności planu jest tracone przy 3-krotnej kompresji.

5. Zamień transport MCP StreamableHTTP na stdio. Porównaj benchmarki cold-start i opóźnienia na wywołanie. Wybierz zwycięzcę dla użycia lokalnego.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Harness | "Pętla agenta" | Kod otaczający model, który wysyła narzędzia, utrzymuje stan planu i egzekwuje budżety |
| Hook | "Nasłuchiwacz zdarzeń agenta" | Skrypt napisany przez użytkownika uruchamiany na jednym z ośmiu zdarzeń cyklu życia przez harness |
| Worktree | "Git sandbox" | Połączone checkout gita w osobnej ścieżce; jednorazowego użytku bez dotykania głównego klonu |
| TodoWrite | "Stan planu" | Typowana lista elementów oczekujących/w trakcie/zrobionych, którą model przepisuje w każdym kroku |
| StreamableHTTP | "Transport MCP" | Rewizja MCP 2026: długożyciowe połączenie HTTP z dwukierunkowym strumieniowaniem; zastępuje SSE |
| Token ceiling | "Budżet kontekstu" | Limit tokenów wejścia+wyjścia na krok lub sesję; uruchamia kompakcję lub zakończenie |
| pass@1 | "Wskaźnik zdawalności przy jednej próbie" | Frakcja zadań SWE-bench rozwiązanych za pierwszym razem bez ponawiania lub podglądania zestawów testowych |

## Dalsza lektura

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — reference harness from Anthropic
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs and Composer 2 product notes
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — minimal baseline for SWE-bench harness comparison
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 79.2% SWE-bench Verified with Opus 4.5
- [OpenCode](https://opencode.ai) — open harness, 112k stars
- [SWE-bench Pro leaderboard](https://www.swebench.com) — the evaluation this capstone targets
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, capability metadata
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — span schema for tool calls and token usage