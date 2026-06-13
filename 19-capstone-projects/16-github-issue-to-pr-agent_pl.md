# Capstone 16 — Autonomiczny Agent GitHub Issue-to-PR

> AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud i Google Jules wszystkie dostarczają ten sam produktowy kształt 2026: oznacz zadanie, otrzymaj PR. Uruchom agenta w chmurowym sandboksie, zweryfikuj, że testy przechodzą i opublikuj gotowy do recenzji PR z uzasadnieniem. Trudne części to automatyczne odtworzenie środowiska budowania repozytorium, zapobieganie wyciekowi poświadczeń, egzekwowanie budżetów na repozytorium i upewnienie się, że agent nie może wymusić pusha. Ten capstone buduje samodzielnie hostowaną wersję i porównuje ją pod względem kosztów i wskaźnika zdawalności z hostowanymi alternatywami.

**Type:** Capstone
**Languages:** Python (agent), TypeScript (GitHub App), YAML (Actions)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## Problem

Asynchroniczny chmurowy agent programistyczny to osobna kategoria produktu od interaktywnych agentów programistycznych (capstone 01). UX to etykieta GitHub. Oznaczasz zadanie `@agent fix this`, worker uruchamia się w chmurowym sandboksie, klonuje repozytorium, uruchamia testy, edytuje pliki, weryfikuje i otwiera PR z uzasadnieniem agenta w treści. Żadnej interaktywnej pętli, żadnego terminala. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules i Factory Droids wszystkie zbiegają się do tego.

Wyzwania inżynieryjne są konkretne: odtworzenie środowiska (agent musi zbudować repozytorium od zera bez cache'owanego obrazu deweloperskiego), flaky testy (muszą być ponownie uruchomione lub wyizolowane), zakresowanie poświadczeń (GitHub App z minimalnymi precyzyjnymi uprawnieniami), egzekwowanie budżetu na repozytorium na dzień i polityka zakazu force-push. Capstone mierzy wskaźnik zdawalności, koszt i bezpieczeństwo vs hostowane alternatywy.

## Koncepcja

Wyzwalaczem jest webhook GitHub (etykieta zadania lub komentarz PR). Dyspozytor umieszcza pracę w kolejce do ECS Fargate lub Lambda. Worker pobiera repozytorium do sandboksa Daytona lub E2B z generycznym Dockerfile wywnioskowanym z repozytorium (język, framework). Agent uruchamia pętlę mini-swe-agent lub SWE-agent v2 przeciwko Claude Opus 4.7 lub GPT-5.4-Codex. Iteruje: czytaj kod, zaproponuj naprawę, zastosuj łatkę, uruchom testy.

Weryfikacja jest krokiem bramkującym. Pełne CI musi przejść w sandboksie przed otwarciem PR. Różnica pokrycia jest obliczana; jeśli ujemna powyżej progu, PR otwiera się, ale dostaje etykietę `needs-review`. Agent publikuje uzasadnienie jako opis PR plus wątek `@agent`, który recenzent może pingować w celu uzupełnień.

Bezpieczeństwo jest zakresowane przez dwie różne powierzchnie GitHub: App zapewnia krótkożyciowy token instalacyjny z `workflows: read` i wąskimi zakresami treści repozytorium/PR; ochrona gałęzi (nie uprawnienia aplikacji) egzekwuje "brak bezpośrednich zapisów do `main`" i "brak force-push" — aplikacja nigdy nie jest dodawana do listy pomijania. Dostęp tylko do odczytu z zakresem ścieżki do `.github/workflows` nie jest prymitywem GitHub App, więc lista dozwolonych plików agenta do edycji musi egzekwować to na workerze. Sufity budżetu na repozytorium na dzień są egzekwowane na dyspozytorze (np. max 5 PR na repozytorium na dzień, $20 na PR).

## Architektura

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## Stack

- Wyzwalacz: GitHub App z precyzyjnym tokenem; odbiornik webhook przez Lambda lub Fly.io
- Worker: Zadanie ECS Fargate (lub GitHub Actions self-hosted runner)
- Sandbox: devcontainer Daytona lub sandbox E2B na zadanie
- Pętla agenta: mini-swe-agent baseline lub SWE-agent v2 na Claude Opus 4.7 / GPT-5.4-Codex
- Wyszukiwanie: tree-sitter repo-map + ripgrep
- Weryfikacja: pełne CI w sandboksie + bramka różnicy pokrycia
- Obserwowalność: Langfuse z archiwum śledzenia na PR połączonym z treścią PR
- Budżet: dzienny sufit dolarowy na repozytorium; max PR na repozytorium na dzień

## Build It

1. **GitHub App.** Precyzyjny token instalacyjny: issues read+write, pull_requests write, contents read+write, workflows read. Ochrona gałęzi (jedyna powierzchnia, która może to zrobić) egzekwuje "brak bezpośredniego pusha do `main`" i "brak force-push"; aplikacja nie jest na liście pomijania. Worker egzekwuje "brak zapisów w `.github/workflows`" jako sprawdzenie listy dozwolonych na proponowanym diffie, ponieważ uprawnienia GitHub App nie są zakresowane ścieżkowo.

2. **Odbiornik webhook.** Funkcja Lambda akceptuje webhooki etykiety zadania / komentarza PR. Filtruje według etykiety `@agent fix this`. Umieszcza w kolejce SQS.

3. **Dyspozytor.** Pobiera zadania z SQS. Egzekwuje budżet na repozytorium na dzień. Uruchamia zadanie ECS Fargate z URL repozytorium, treścią zadania i świeżym sandboksem Daytona.

4. **Wnioskowanie środowiska.** Wykryj język (Python, Node, Go, Rust) i menedżer pakietów (uv, pnpm, go mod, cargo). Wygeneruj Dockerfile na bieżąco, jeśli nie istnieje.

5. **Pętla agenta.** mini-swe-agent lub SWE-agent v2 z Claude Opus 4.7. Narzędzia: ripgrep, tree-sitter repo-map, read_file, edit_file, run_tests, git. Twarde limity: $20 kosztu, 30 min czasu ściennego, 30 kroków agenta.

6. **Weryfikacja.** Po zakończeniu pętli, uruchom pełny zestaw testów w sandboksie. Oblicz różnicę pokrycia przez jacoco / coverage.py. Jeśli CI czerwone: zatrzymaj, nie otwieraj PR. Jeśli pokrycie spadnie o więcej niż 2%: otwórz PR z etykietą `needs-review`.

7. **Publikacja PR.** Wypchnij gałąź agenta. Otwórz PR przez API GitHub z: tytułem, uzasadnieniem, podsumowaniem diffa, URL śledzenia, kosztem, krokami.

8. **Higiena poświadczeń.** Worker działa z krótkożyciowym tokenem instalacyjnym GitHub App. Logi są czyszczone z sekretów przed archiwizacją.

9. **Ewaluacja.** 30 zasianych wewnętrznych zadań o różnym poziomie trudności. Zmierz wskaźnik zdawalności, jakość PR (rozmiar diffa, styl, pokrycie), koszt, opóźnienie. Porównaj z Cursor Background Agents i AWS Remote SWE Agents na tych samych zadaniach.

## Use It

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## Ship It

`outputs/skill-issue-to-pr.md` jest rezultatem. GitHub App + asynchroniczny chmurowy worker, który zamienia oznaczone zadania w gotowe do recenzji PR z ograniczonym kosztem i zakresowanymi poświadczeniami.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Wskaźnik zdawalności na 30 zadaniach | Sukces end-to-end (CI zielone + pokrycie OK) |
| 20 | Jakość PR | Rozmiar diffa, różnica pokrycia, zgodność stylu |
| 20 | Koszt i opóźnienie na rozwiązane zadanie | $ i czas ścienny na PR |
| 20 | Bezpieczeństwo | Zakresowany token, budżet na repozytorium, brak force-push, higiena poświadczeń |
| 15 | UX operatora | Komentarze uzasadnienia, możliwość ponowienia, follow-up przez @-wzmiankę |
| **100** | | |

## Ćwiczenia

1. Dodaj tryb "napraw flaky test": etykieta `@agent stabilize-flake TestX` uruchamia test 50 razy w sandboksie i proponuje minimalną zmianę, która go stabilizuje.

2. Porównaj koszt vs Cursor Background Agents na trzech wspólnych zadaniach. Raportuj, które narzędzia wygrywają gdzie.

3. Zaimplementuj kokpit budżetu: koszt na repozytorium na dzień, koszt na użytkownika. Alarmuj na anomalie.

4. Zbuduj tryb "dry-run", który otwiera draft PR bez uruchamiania CI, aby recenzenci mogli tanio przejrzeć plan.

5. Dodaj politykę przechowywania: gałęzie PR starsze niż 7 dni bez scalenia są automatycznie usuwane.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| GitHub App | "Zakresowana tożsamość bota" | Aplikacja z precyzyjnymi uprawnieniami + krótkożyciowym tokenem instalacyjnym |
| Async cloud agent | "Agent w tle" | Nieinteraktywny worker działający w chmurowym sandboksie, nie terminalu |
| Environment inference | "Synteza Dockerfile" | Wykryj język + menedżer pakietów, wygeneruj Dockerfile, jeśli brak |
| Verification | "CI-w-sandboksie" | Uruchom pełny zestaw testów wewnątrz workera przed otwarciem PR |
| Coverage delta | "Zachowanie pokrycia" | Zmiana % pokrycia testów od bazy do gałęzi agenta |
| Per-repo budget | "Dzienny sufit" | Limit dolarowy i liczby PR egzekwowany na dyspozytorze |
| Rationale | "Wyjaśnienie w treści PR" | Podsumowanie agenta co się zmieniło i dlaczego; wymagane w treści PR |

## Dalsza lektura

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — the canonical async cloud agent reference
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI reference
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — commercial alternative
- [OpenAI Codex (cloud)](https://openai.com/codex) — hosted competitor
- [Google Jules](https://jules.google) — Google's hosted version
- [Factory Droids](https://www.factory.ai) — alternate commercial reference
- [GitHub App documentation](https://docs.github.com/en/apps) — scoped bot identity
- [Daytona cloud sandboxes](https://daytona.io) — reference sandbox