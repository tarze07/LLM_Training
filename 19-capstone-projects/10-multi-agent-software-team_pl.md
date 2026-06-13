# Capstone 10 — Wieloagentowy Zespół Inżynierii Oprogramowania

> Architektura fabryki SWE-AF, MetaGPT's role-based prompting, AutoGen 0.4's typed actor graph, Cognition's Devin i Factory's Droids wszystkie zbiegły się do tego samego kształtu w 2026 roku: architekt planuje, N programistów pracuje równolegle w osobnych worktree, recenzent bramkuje, tester weryfikuje. Równoległe worktree zamieniają czas ścienny w przepustowość. Wspólny stan i protokoły przekazywania stają się powierzchnią awarii. Capstone polega na zbudowaniu zespołu, ewaluacji na SWE-bench Pro i raportowaniu, które przekazania się psują i jak często.

**Type:** Capstone
**Languages:** Python / TypeScript (agents), Shell (worktree scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17
**Time:** 40 hours

## Problem

Jednoagentowe harnessy programistyczne napotykają sufit na dużych zadaniach. Nie dlatego, że jakikolwiek pojedynczy agent jest słaby, ale dlatego, że 200k-tokenowy kontekst nie może pomieścić planu architektury plus czterech równoległych wycinków bazy kodu plus komentarzy recenzenta plus wyników testów. Wieloagentowe fabryki dzielą problem: architekt posiada plan, programiści posiadają implementację w równoległych worktree, recenzent bramkuje, tester weryfikuje. Architektura "fabryki" SWE-AF, role MetaGPT, typowany graf aktorów AutoGen — wszystkie trzy ramy opisują ten sam kształt.

Powierzchnia awarii to przekazanie. Architekt planuje coś, czego programiści nie mogą zaimplementować. Programiści produkują konfliktujące diffy. Recenzent zatwierdza halucynacyjną naprawę. Tester ściga się z wciąż piszącym programistą. Zbudujesz jeden z tych zespołów, uruchomisz go na 50 zadaniach SWE-bench Pro, prześledzisz każde przekazanie i opublikujesz analizę powłamaniową.

## Koncepcja

Role to typowane agenty. **Architekt** (Claude Opus 4.7) czyta zadanie, pisze plan i dzieli je na podzadania z jawnymi interfejsami. **Programiści** (Claude Sonnet 4.7, N równoległych instancji, każda w `git worktree` + sandbox Daytona) implementują podzadania niezależnie. **Recenzent** (GPT-5.4) czyta scalony diff i albo zatwierdza, albo żąda konkretnych zmian. **Tester** (Gemini 2.5 Pro) uruchamia zestaw testów w izolacji i raportuje pass/fail z artefaktami.

Komunikacja odbywa się przez wspólną tablicę zadań (opartą na plikach lub Redis). Każda rola konsumuje zadania, które może obsłużyć. Przekazania to wiadomości typowane protokołem A2A. Zagadnienia koordynacji: rozwiązywanie konfliktów scalania (rola koordynatora lub automatyczne trójstronne scalanie), synchronizacja wspólnego stanu (plan jest zamrożony, gdy programiści zaczynają; replany to osobne zdarzenia) i bramkowanie recenzenta (recenzent nie może zatwierdzać własnych zmian ani zmian, które zaproponował).

Amplifikacja tokenów to ukryty koszt. Każda granica roli dodaje podsumowujące prompty i kontekst przekazania. 40-krokowy jednoagentowy przebieg staje się 160 całkowitych kroków w czterech rolach. Rubryka szczególnie waży wydajność tokenów vs jednoagentowe baseline, ponieważ pytanie brzmi nie "czy multi-agent działa", ale "czy wygrywa za dolara".

## Architektura

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## Stack

- Orkiestracja: LangGraph ze wspólnym stanem + sub-grafy na agenta
- Wiadomości: Protokół A2A (Google 2025) dla typowanych wiadomości międzyagentowych
- Modele: Opus 4.7 (architekt), Sonnet 4.7 (programiści), GPT-5.4 (recenzent), Gemini 2.5 Pro (tester)
- Izolacja worktree: `git worktree add` na programistę + sandbox Daytona
- Koordynator scalania: niestandardowe trójstronne scalanie + rozwiązywanie konfliktów przez LLM
- Ewaluacja: SWE-bench Pro (50 zadań), scenariusze SWE-AF, HumanEval++ dla testów jednostkowych
- Obserwowalność: Langfuse z spanami oznaczonymi rolami, rozliczanie tokenów na agenta
- Wdrożenie: K8s z każdą rolą jako osobnym Deployment + HPA na zaległościach

## Build It

1. **Tablica zadań.** JSONL oparty na plikach z typowanymi wiadomościami: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. Agenci subskrybują tagi.

2. **Architekt.** Czyta zadanie GitHub, uruchamia Opus 4.7 z szablonem planu wymagającym jawnych interfejsów podzadań (dotknięte pliki, funkcje publiczne, wpływ na testy). Emituje jeden `plan_request` z DAG-iem podzadań.

3. **Programiści.** N równoległych pracowników, każdy zgłasza się po jedno podzadanie z tablicy. Każdy tworzy świeżą gałąź `git worktree add` plus sandbox Daytona. Implementuje podzadanie. Emituje `diff_ready` z łatką + zmianami testów.

4. **Koordynator scalania.** Po zakończeniu wszystkich programistów, trójstronnie scala N gałęzi do gałęzi stagingowej. Rozwiązywanie konfliktów przez LLM tylko przy nakładaniu się na poziomie plików.

5. **Recenzent.** GPT-5.4 czyta scalony diff. Nie może zatwierdzać diffów, które sam napisał. Emituje `approved` (no-op) lub `review_feedback` z konkretnymi żądaniami zmian skierowanymi z powrotem do odpowiedniego programisty.

6. **Tester.** Gemini 2.5 Pro uruchamia zestaw testów w czystym sandboksie. Przechwytuje artefakty. Emituje `test_passed` lub `test_failed` ze śladami stosu. Nieudane testy wracają do programisty posiadającego zawodzące podzadanie.

7. **Rozliczanie przekazań.** Każda wiadomość przekraczająca granicę roli otrzymuje span w Langfuse z rozmiarem ładunku i użytym modelem. Oblicz amplifikację tokenów na podzadanie (tokeny_programisty + tokeny_recenzenta + tokeny_testera + udział_architekta / tokeny_programisty).

8. **Ewaluacja.** Uruchom na 50 zadaniach SWE-bench Pro. Porównaj pass@1 i $-na-rozwiązane-zadanie przeciwko jednoagentowemu baseline (jeden Sonnet 4.7 w jednym worktree).

9. **Analiza powłamaniowa.** Dla każdego nieudanego zadania, zidentyfikuj, które przekazanie się zepsuło (plan zbyt ogólny, konflikt scalania, fałszywe zatwierdzenie recenzenta, flaky tester). Wyprodukuj histogram awarii przekazań.

## Use It

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Ship It

`outputs/skill-multi-agent-team.md` jest rezultatem. Mając URL zadania i poziom równoległości, zespół produkuje gotowy do scalenia PR z rozliczaniem tokenów na rolę.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | Dopasowany podzbiór 50 zadań, pass@1 |
| 20 | Przyspieszenie równoległe | Czas ścienny vs jednoagentowe baseline |
| 20 | Jakość recenzji | Wskaźnik fałszywych zatwierdzeń na teście z wstrzykniętym błędem |
| 20 | Wydajność tokenów | Całkowite tokeny na rozwiązane zadanie vs jednoagentowe |
| 15 | Inżynieria koordynacji | Rozwiązywanie konfliktów scalania, histogram awarii przekazań |
| **100** | | |

## Ćwiczenia

1. Wstrzyknij oczywisty błąd do diffa w trakcie przebiegu (dodatkowy `return None` przed głównym ciałem). Zmierz wskaźnik fałszywych zatwierdzeń recenzenta. Dostrój prompt recenzenta, aż fałszywe zatwierdzenie będzie poniżej 5%.

2. Zmniejsz do dwóch programistów (architekt + programista + recenzent + tester, programista wykonuje dwa podzadania sekwencyjnie). Porównaj czas ścienny i wskaźnik zdawalności.

3. Zastąp koordynatora scalania ograniczeniem pojedynczego piszącego (podzadania dotykają rozłącznych zestawów plików). Zmierz obciążenie planowaniem architekta.

4. Zamień recenzenta z GPT-5.4 na Claude Opus 4.7. Zmierz wskaźnik fałszywych zatwierdzeń i różnicę kosztu tokenów.

5. Dodaj piątą rolę: dokumentalista (Haiku 4.5). Po recenzji, produkuje wpis do changeloga. Zmierz, czy jakość dokumentacji uzasadnia dodatkowy wydatek tokenów.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Parallel worktree | "Izolowana gałąź" | `git worktree add` produkujące świeże drzewo robocze na programistę |
| Task board | "Wspólna magistrala wiadomości" | Przechowywany w pliku lub Redis zbiór typowanych wiadomości, które agenci subskrybują |
| Handoff | "Granica roli" | Każda wiadomość przechodząca z kontekstu jednej roli do drugiej |
| Token amplification | "Narzut wieloagentowy" | Całkowite tokeny we wszystkich rolach / tokeny jednoagentowe dla tego samego zadania |
| A2A protocol | "Agent-to-agent" | Specyfikacja Google z 2025 dla typowanych wiadomości międzyagentowych |
| Merge coordinator | "Integrator" | Komponent, który uruchamia trójstronne scalanie i mediuje konflikty |
| False approval | "Halucynacja recenzenta" | Recenzent zatwierdza diffa ze znanymi błędami |

## Dalsza lektura

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — the reference 2026 multi-agent factory
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — role-based multi-agent framework
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft's typed actor framework
- [Cognition AI (Devin)](https://cognition.ai) — reference product
- [Factory Droids](https://www.factory.ai) — alternate reference product
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — inter-agent messaging spec
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — the isolation substrate
- [SWE-bench Pro](https://www.swebench.com) — the evaluation target