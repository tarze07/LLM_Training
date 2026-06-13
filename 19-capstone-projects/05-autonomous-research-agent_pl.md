# Capstone 05 — Autonomiczny Agent Badawczy (Klasa AI-Scientist)

> AI-Scientist-v2 od Sakana opublikował pełne artykuły. Agent Laboratory przeprowadził eksperymenty. Allen AI udostępnił ślady. Kształt w 2026 roku to przeszukiwanie drzewa plan-wykonaj-zweryfikuj po eksperymentach, budżetowany koszt, sandboksowane wykonanie kodu, wizyjny edytor LaTeX i zautomatyzowany zespół recenzentów w stylu NeurIPS. Capstone polega na zbudowaniu takiego systemu, uruchomieniu go od końca do końca w budżecie $30 na artykuł i przetrwaniu audytu bezpieczeństwa sandboksa, który udokumentowała Sakana.

**Type:** Capstone
**Languages:** Python (agent + sandbox), LaTeX (output)
**Prerequisites:** Phase 2 (ML), Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 18 (safety)
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Time:** 40 hours

## Problem

Autonomiczne agenty badawcze przekroczyły próg w 2026 roku. AI-Scientist-v2 od Sakana AI został opublikowany w Nature z wygenerowanymi artykułami, które przeszły warsztatową recenzję. ShinkaEvolve (ICLR 2026) rozszerzył linię na ewoluujące hipotezy. Agent Laboratory od AMD dostarczył powtarzalne ślady. Agenty nie są magiczne — to pętla plan-wykonaj-zweryfikuj działająca na drzewie kandydackich eksperymentów, z ograniczeniami kosztów, sandboksami z ustalonym seedem i automatyczną recenzją. Rzemiosło tkwi w pętli, budżecie i historii bezpieczeństwa.

Uczysz się pętli implementując ją przeciwko seedowemu pomysłowi w wąskiej domenie (na przykład ablacje rzadkości uwagi na 100M-parametrowym transformerze). Wartość nie leży w odkryciu czegoś nowego przy pierwszym uruchomieniu. Wartość leży w infrastrukturze: przeszukiwaniu drzewa, sandboksie eksperymentów, pętli pisarz-recenzent, raporcie red team. Zespół Sakana udokumentował awarie ucieczki z sandboksa; twój agent musi przejść ten sam red team.

## Koncepcja

Agent to przeszukiwanie drzewa w trybie best-first. Węzły to specyfikacje eksperymentów: (hipoteza, konfiguracja, kod, oczekiwany wynik). Krok rozwijania proponuje dzieci z małymi edycjami (zamień optymalizator, zmień rozmiar batcha, usuń komponent). Każde dziecko działa w świeżym sandboksie z twardym limitem zasobów. Wyniki wracają do funkcji punktującej, która rankuje węzły według (nowość × jakość × pozostały budżet). Drzewo rośnie aż budżet się wyczerpie, a następnie najlepsza gałąź jest opisywana.

Pisarz jest multimodalny. Generuje szkic LaTeX, kompiluje go, renderuje figury i przekazuje wyrenderowany PDF z powrotem do trybu wizji Claude Opus 4.7 w celu krytyki układu, czytelności figur i dopasowania twierdzeń do dowodów. Zespół recenzentów złożony z pięciu sędziów LLM emituje oceny w stylu NeurIPS (nowość, rzetelność, jasność, powtarzalność, wpływ); jeśli średnia spadnie poniżej progu, artykuł wraca do pisarza z krytyką.

Bezpieczeństwo jest kluczowe. Każdy eksperyment działa w sandboksie E2B lub Daytona bez egressu sieciowego, z ograniczonym czasem i przypiętymi limitami zasobów. Krok generowania kodu przez agenta przechodzi przez warstwę polityki, która blokuje wywołania systemowe uciekające z sandboksa. Raport red team odtwarza udokumentowaną przez Sakana powierzchnię ataku (fork bomby, ucieczki systemu plików, wywołania sieciowe napisane przez LLM).

## Architektura

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## Stack

- Orkiestracja: LangGraph z checkpointingiem i bramkami zatwierdzania przez człowieka
- Przeszukiwanie drzewa: własne best-first po węzłach eksperymentów (w stylu AB-MCTS z Sakana v2)
- Sandbox: E2B na eksperyment, Docker-in-Docker jako zapasowy; limity zasobów przez cgroups
- Literatura: Semantic Scholar Graph API + OpenAlex + lokalny cache FAISS abstraktów
- Pisarz: szablon LaTeX + Claude Opus 4.7 (tryb wizji) dla krytyki figur i układu
- Recenzent: ensemble 5 sędziów (Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max) z ważoną agregacją
- Framework eksperymentów: PyTorch 2.5 dla fizycznych eksperymentów, W&B do logowania
- Obserwowalność: Langfuse dla śladów agenta, twardy budżet $30 na artykuł

## Build It

1. **Seed i zakres domeny.** Weź seedowy pomysł (np. "zbadaj wzorce rzadkości w mapach uwagi transformerów poniżej 1B"). Zdefiniuj przestrzeń poszukiwań: modele, zestawy danych, budżet obliczeniowy.

2. **Przegląd literatury.** Zapytaj Semantic Scholar + OpenAlex o 50 najczęściej cytowanych istotnych artykułów; cache'uj abstrakty lokalnie; wygeneruj 1-stronicowe podsumowanie domeny.

3. **Szkielet drzewa.** Zainicjuj korzeń z seedową hipotezą. Zaimplementuj `expand(node) -> children` z propozycjami małych edycji (jedna zmiana konfiguracji na dziecko). Zaimplementuj `score(node)` jako ważony termin nowość × jakość × budżet.

4. **Otoczka sandboksa.** Każdy eksperyment działa `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` (lub równoważna polityka E2B). Seedy są zapisywane do sandboksa; wyniki są montowane tylko do odczytu z powrotem.

5. **Pętla plan-wykonaj-zweryfikuj.** `plan` proponuje dzieci. `execute` uruchamia sandbox, przechwytuje logi i metryki. `verify` uruchamia sprawdzenia jednostkowe na metrykach (czy strata spadła? czy ablacja wyizolowała efekt?). Nieudane węzły otrzymują przyczynę awarii zapisaną na drzewie.

6. **Pisarz.** Po wyczerpaniu budżetu, wybierz najlepszą gałąź. Wyrenderuj figury matplotlib. Wygeneruj szkic LaTeX przez Claude Opus 4.7 ze śladem gałęzi w kontekście. Skompiluj. Przekaż skompilowany PDF z powrotem do Opus 4.7 vision w celu krytyki. Iteruj.

7. **Zespół recenzentów.** Pięciu sędziów punktuje szkic za (nowość, rzetelność, jasność, powtarzalność, wpływ) z rubrykami w stylu NeurIPS. Jeśli średnia < 4.0/5, zwróć do pisarza z krytyką. Twarde zatrzymanie po 3 przepisaniach.

8. **Red team.** Zbuduj lub zintegruj zestaw zadań adversarialnych celujących w sandbox: fork bomby, próby eksfiltracji sieciowej, ucieczki systemu plików, metaznaki shellowe napisane przez LLM. Potwierdź, że wszystkie są blokowane. Sporządź raport.

9. **Powtarzalność.** Każdy artykuł jest dostarczany z JSON-em śladu przeszukiwania drzewa, seedami, linkami do przebiegów W&B, konfiguracjami sandboksa i README odtwarzającym go od końca do końca.

## Use It

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## Ship It

`outputs/skill-ai-scientist.md` jest rezultatem. Mając seedowy pomysł + domenę + budżet $30, uruchamia pełny potok i emituje recenzowalny artykuł plus pakiet powtarzalności.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Jakość artykułu | Ślepa ocena rubrykowa względem opublikowanych artykułów warsztatowych |
| 20 | Rzetelność eksperymentalna | Baseliney, seedy, ablacje; każde twierdzenie poparte komórką w tabeli wyników |
| 20 | Dyscyplina kosztowa i obliczeniowa | Limit $30/artykuł egzekwowany, śledzony przez Langfuse |
| 20 | Bezpieczeństwo | Red team sandboksa przechodzi; polityka sieciowa i kill-switch zweryfikowane |
| 15 | Powtarzalność | Ponowne uruchomienie jednym poleceniem z identycznymi seedami odtwarza artykuł |
| **100** | | |

## Ćwiczenia

1. Uruchom potok przeciwko trzem różnym seedowym pomysłom w tej samej domenie. Porównaj, które części przeszukiwania drzewa nakładają się. Zidentyfikuj zduplikowany zmarnowany koszt obliczeniowy.

2. Dodaj bramkę człowiek-w-pętli przed wykonaniem eksperymentu dla węzłów o szacowanym koszcie powyżej $5. Zmierz, jak bardzo spada całkowity koszt.

3. Zamień zespół recenzentów na pojedynczego sędziego. Zmierz wskaźnik fałszywych akceptacji na wstrzymanym zestawie znanych złych artykułów.

4. Wprowadź test red team eksfiltracji sieciowej: agent pisze kod, który próbuje `curl` zewnętrzny adres. Potwierdź, że polityka `--network=none` blokuje go. Zaloguj próbę.

5. Porównaj swoje przeszukiwanie drzewa z płaskim baseline losowym (ten sam budżet, bez strategii rozwijania). Raportuj zysk nowość × jakość.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Tree search | "Ekspansja w stylu AB-MCTS" | Eksploracja best-first po węzłach eksperymentów z punktacją nowość×jakość×budżet |
| Sandbox | "Izolacja eksperymentu" | Kontener bez sieci, ograniczonym CPU/pamięcią, przypiętymi seedami, wejściami tylko do odczytu |
| Vision critique | "Renderuj-potem-czytaj" | Skompiluj artykuł do PDF, przekaż PDF z powrotem do VLM w celu krytyki układu i dopasowania twierdzeń do dowodów |
| Reviewer ensemble | "Zautomatyzowana recenzja" | Wielu sędziów LLM punktujących artykuł z rubryką NeurIPS; ważona agregacja bramkuje potok |
| Novelty score | "Czy to jest nowe?" | Heurystyka karząca bliskość do cache'u 50 artykułów z literatury |
| Cost ceiling | "Budżet $" | Twardy limit całkowitego wydatku na artykuł; liczniki Langfuse + oszacowania przed uruchomieniem |
| Red team | "Audyt ucieczki z sandboksa" | Zadania adversarialne, które uciekłyby z sandboksa jeśli polityka jest błędna |

## Dalsza lektura

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — the reference production research agent
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — the original methodology
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — evolutionary extension
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — multi-role research-lab framework
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — reference orchestration layer
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — literature search
- [E2B sandboxes](https://e2b.dev) — reference experiment isolation
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — the rubric the reviewer ensemble encodes