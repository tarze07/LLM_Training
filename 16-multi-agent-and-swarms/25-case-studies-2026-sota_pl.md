# Studia Przypadków i Stan Sztuki w 2026

> Trzy produkcyjne referencje do kompleksowego przestudiowania, każda ilustrująca inny wycinek inżynierii wieloagentowej. **System badawczy Anthropic** (orkiestrator-worker, 15x tokenów, +90.2% nad pojedynczym agentem Opus 4, wdrożenia tęczowe) — kanoniczny przypadek nadzorcy. **MetaGPT / ChatDev** (specjalizacja ról zakodowana w SOP dla inżynierii oprogramowania; "komunikatywna dehallucynacja" ChatDev; rozszerzenie MacNet do >1000 agentów przez DAGi, arXiv:2406.07155) — kanoniczny przypadek dekompozycji ról. **OpenClaw / Moltbook** (pierwotnie Clawdbot autorstwa Petera Steinbergera, listopad 2025; przemianowany dwukrotnie; 247k gwiazdek GitHub do marca 2026; lokalne agenty ReAct-loop; Moltbook jako sieć społecznościowa tylko dla agentów z ~2.3M kont agentów w ciągu dni od uruchomienia, przejęty przez Meta 2026-03-10) ilustruje, co dzieje się w skali populacji: emergentna aktywność ekonomiczna, ryzyko wstrzykiwania promptów, regulacje na szczeblu państwowym (Chiny ograniczyły OpenClaw na komputerach rządowych, marzec 2026). **Krajobraz frameworków kwiecień 2026:** LangGraph i CrewAI prowadzą w produkcji; AG2 to kontynuacja społecznościowa AutoGen; Microsoft AutoGen jest w trybie utrzymania (włączony do Microsoft Agent Framework, RC luty 2026); OpenAI Agents SDK to produkcyjny następca Swarm; Google ADK (kwiecień 2025) to natywny dla A2A nowy uczestnik. Każdy główny framework ma teraz wsparcie MCP; większość ma A2A. Ta lekcja czyta każdy przypadek kompleksowo i destyluje wspólne wzorce, abyś mógł wybrać właściwą referencję dla swojego następnego systemu produkcyjnego.

**Type:** Learn (capstone)
**Languages:** —
**Prerequisites:** all of Phase 16 (Lessons 01-24)
**Time:** ~90 minutes

## Problem

Inżynieria wieloagentowa to młoda dyscyplina. Referencje produkcyjne są nieliczne, a każda pokrywa inną część przestrzeni. Czytanie ich pojedynczo jest użyteczne; porównanie ich jako zestawu jest bardziej użyteczne. Ta lekcja traktuje trzy kanoniczne studia przypadków z 2026 jako kompleksową listę lektur, wskazuje wspólne wzorce i mapuje krajobraz frameworków, abyś mógł dokonywać wyborów frameworków z wiedzy, a nie z marketingu.

## Koncepcja

### System badawczy Anthropic

Produkcyjny przypadek nadzorca-worker. Claude Opus 4 planuje i syntetyzuje; Claude Sonnet 4 subagenci badają równolegle. Opublikowany post inżynieryjny: https://www.anthropic.com/engineering/multi-agent-research-system.

Kluczowe zmierzone wyniki:

- **+90.2%** poprawa w porównaniu z pojedynczym agentem Opus 4 w wewnętrznych ewaluacjach badawczych.
- **80% wariancji BrowseComp** wyjaśnione przez **samo użycie tokenów** — wiele agentów wygrywa głównie dlatego, że każdy subagent dostaje świeże okno kontekstowe.
- **15x tokenów na zapytanie** vs pojedynczy agent.
- **Wdrożenie tęczowe**, ponieważ agenci są długotrwałe i stanowe.

Ugruntowane lekcje projektowe:

1. **Dostosuj nakład do złożoności zapytania.** Proste → 1 agent z 3-10 wywołaniami narzędzi. Średnie → 3 agentów. Złożone badania → 10+ subagentów.
2. **Najpierw szeroko, potem wąsko.** Subagenci przeprowadzają szerokie wyszukiwania; prowadzący syntetyzuje; subagenci uzupełniający wykonują ukierunkowane głębokie analizy.
3. **Wdrożenia tęczowe.** Utrzymuj stare wersje środowiska uruchomieniowego przy życiu, dopóki ich aktywni agenci nie zakończą.
4. **Weryfikacja nie jest opcjonalna.** Zaobserwowano, że bez jawnych ról weryfikatorów system ulegał halucynacji.

To jest referencyjny przypadek topologii nadzorca-pracownik (Faza 16 · 05) w skali produkcyjnej.

### MetaGPT / ChatDev

Produkcyjny przypadek dekompozycji ról przez SOP. Obejmuje arXiv:2308.00352 (MetaGPT) i arXiv:2307.07924 (ChatDev).

MetaGPT koduje SOP inżynierii oprogramowania jako prompty ról: Product Manager, Architect, Project Manager, Engineer, QA Engineer. Ramowanie artykułu: `Code = SOP(Team)`. Każda rola ma wąski, wyspecjalizowany prompt; przekazania między rolami niosą ustrukturyzowane artefakty (dokumenty PRD, dokumenty architektury, kod).

Wkład ChatDev: **komunikatywna dehallucynacja**. Agenci proszą o szczegóły przed odpowiedzią — agent projektant pyta programistę, jaki język jest zamierzony przed szkicowaniem UI, zamiast zgadywać. Artykuł raportuje, że to mierzalnie zmniejsza halucynacje w potokach wieloagentowych.

MacNet (arXiv:2406.07155) rozszerza ChatDev do **>1000 agentów przez DAGi**. Każdy węzeł DAG to specjalizacja roli; krawędzie kodują kontrakty przekazania. Skala jest możliwa, ponieważ routing jest jawny i obliczalny offline.

Lekcje projektowe:

1. **Struktura ma znaczenie bardziej niż rozmiar.** Zwarty zespół 5 ról SOP bije nieustrukturyzowaną grupę 50 agentów.
2. **Kontrakty przekazania na piśmie.** Artefakty przekazywane między rolami są zgodne ze schematem.
3. **Komunikatywna dehallucynacja** to tani, nośny wzorzec.
4. **DAGi skalują się dalej niż czat.** Gdy przepływ jest znany, zakoduj go.

To jest referencyjny przypadek specjalizacji ról (Faza 16 · 08) i topologii strukturalnej (Faza 16 · 15).

### Ekosystem OpenClaw / Moltbook

Produkcyjny przypadek skali populacyjnej. Oś czasu:

- **Listopad 2025:** Clawdbot (lokalny agent programistyczny ReAct-loop Petera Steinbergera) zostaje wydany.
- **Grudzień 2025 – Marzec 2026:** przemianowany dwukrotnie (Clawdbot → OpenClaw → kontynuowany jako OpenClaw).
- **Luty 2026:** Moltbook uruchamia się jako sieć społecznościowa tylko dla agentów na tych samych prymitywach; ~2.3M kont agentów w ciągu dni.
- **Marzec 2026 (2026-03-10):** Meta przejmuje Moltbook.
- **Marzec 2026:** Chiny ograniczają OpenClaw na komputerach rządowych.
- **Marzec 2026:** OpenClaw przekracza 247k gwiazdek GitHub.

Tak wygląda wiele agentów, gdy umieścisz miliony agentów na współdzielonym podłożu:

- **Emergentna aktywność ekonomiczna.** Agenci kupują, sprzedają i świadczą sobie usługi za pomocą płatności tokenowych.
- **Ryzyko wstrzykiwania promptów w skali populacyjnej.** Jeden złośliwy prompt w wirusowym profilu agenta propaguje się do tysięcy interakcji agent-agent w ciągu godzin.
- **Odpowiedź regulacyjna na szczeblu państwowym.** W ciągu tygodni od uruchomienia regulacje docierają do ekosystemu.

Lekcje projektowe z tego przypadku są częściowo techniczne, częściowo związane z zarządzaniem:

1. **Skala populacyjna wielu agentów to nowy reżim.** Najlepsze praktyki dla pojedynczego systemu (weryfikacja, jasność ról) wciąż obowiązują, ale nie są wystarczające.
2. **Wstrzykiwanie promptów to nowe XSS.** Traktuj profile agentów i wiadomości między agentami jako niezaufane dane wejściowe domyślnie.
3. **Regulacja jest szybsza niż cykle projektowe.** Planuj to.
4. **Open source + skala wirusowa się kumulują.** 247k gwiazdek w ~4 miesiące jest nietypowe; projektuj na wybuchowe obciążenie wdrożeniowe.

Zobacz [OpenClaw na Wikipedii](https://en.wikipedia.org/wiki/OpenClaw) oraz raporty CNBC / Palo Alto Networks dla szczegółów ekosystemu. Dla podstaw technicznych, repozytoria Clawdbot / OpenClaw ujawniają lokalną pętlę ReAct; publiczne posty Moltbook ujawniają architekturę grafu społecznościowego na wierzchu.

### Krajobraz frameworków kwiecień 2026

| Framework | Status | Najlepszy do | Uwagi |
|---|---|---|---|
| **LangGraph** (LangChain) | Lider produkcji | ustrukturyzowany graf + punkt kontrolny + człowiek-w-pętli | zalecana domyślna dla produkcji |
| **CrewAI** | Lider produkcji | załogi oparte na rolach z procesami Sekwencyjnymi/Hierarchicznymi | silny dla dekompozycji ról |
| **AG2** | Utrzymywany przez społeczność | GroupChat + wybór mówcy | kontynuacja AutoGen v0.2 |
| **Microsoft AutoGen** | Tryb utrzymania (luty 2026) | — | włączony do Microsoft Agent Framework RC |
| **Microsoft Agent Framework** | RC (luty 2026) | wzorce orkiestracji + integracja korporacyjna | nowy uczestnik; obserwuj |
| **OpenAI Agents SDK** | Produkcja | następca Swarm | wzorzec przekazania zwrotu narzędzia |
| **Google ADK** | Produkcja (kwiecień 2025) | natywny A2A | integracja Google Cloud |
| **Anthropic Claude Agent SDK** | Produkcja | pojedynczy agent + rozszerzenie Research | zobacz post o systemie Research |

Każdy główny framework ma teraz wsparcie **MCP**; większość ma **A2A**. Kompatybilność protokołów nie jest już wyróżnikiem.

### Wspólne wzorce we wszystkich trzech przypadkach

1. **Orkiestrator + pracownicy** (Anthropic jawny nadzorca, MetaGPT PM-jako-nadzorca, indywidualni agenci OpenClaw + efekty sieciowe).
2. **Ustrukturyzowane kontrakty przekazania** (opisy zadań subagentów Anthropic, dokumenty PRD/architektury MetaGPT, artefakty A2A OpenClaw).
3. **Weryfikacja jako pierwszorzędna rola** (weryfikator Anthropic, QA Engineer MetaGPT, walidatory w sieci OpenClaw).
4. **Skalowanie to topologia + podłoże, a nie tylko więcej agentów** (wdrożenia tęczowe, DAGi MacNet, podłoża skali populacyjnej).
5. **Koszt jest materialny i ujawniony** (15x tokenów, budżet na rolę w MetaGPT, cena za interakcję w Moltbook).
6. **Postawa bezpieczeństwa jest jawna** (piaskownica Anthropic, ograniczenia ról MetaGPT, wstrzykiwanie promptów jako znana powierzchnia ataku w OpenClaw).

### Wybór referencji dla następnego projektu

- **Produkcyjne zadanie badawcze/wiedzy → Anthropic Research.** Subagenci ze świeżym kontekstem wygrywają.
- **Inżynieryjny/tool-chain przepływ pracy → MetaGPT / ChatDev.** Role + SOP + kontrakty przekazania.
- **Produkt społecznościowy z efektem sieciowym → OpenClaw / Moltbook.** Podłoże + emergentna ekonomia.
- **Klasyczna automatyzacja korporacyjna → CrewAI lub LangGraph** (lider produkcji, stabilne środowisko uruchomieniowe).

### Podsumowanie stanu sztuki w 2026

Gdzie jest dziedzina w kwietniu 2026:

- **Frameworki konwergują.** Wsparcie MCP + A2A to minimalny próg. Semantyka przekazania to pozostały wybór projektowy.
- **Ewaluacja się utwardza.** SWE-bench Pro, MARBLE, benchmarki łagodzenia STRATUS. Pro jest obecną odporną na kontaminację kontrolą rzeczywistości.
- **Produkcyjne wskaźniki awaryjności są mierzalne** (Cemri 2025 MAST; 41-86.7% na rzeczywistych MAS). Dziedzina wyszła z ery "wygląda świetnie w demo."
- **Koszt jest centralnym ograniczeniem inżynieryjnym.** Koszt tokenów na zadanie, czas ścienny na interakcję, narzut wdrożeń tęczowych. Wiele agentów wygrywa na dokładności, ale traci na koszcie — i ten kompromis jest decyzją biznesową.
- **Regulacja jest wejściem krótkoterminowym, a nie tłem.** Jurysdykcje działają szybciej niż indywidualne cykle wdrożeniowe.

## Use It

`outputs/skill-case-study-mapper.md` to umiejętność, która czyta proponowany projekt systemu wieloagentowego i mapuje go na najbliższe studium przypadku, wydobywając decyzje projektowe, które to studium już przetestowało.

## Ship It

Początkowe zasady dla produkcyjnych systemów wieloagentowych w 2026:

- **Zacznij od studium przypadku, a nie od zera.** Wybierz najbliższe z Anthropic Research / MetaGPT / OpenClaw i dostosuj.
- **Adoptuj MCP + A2A.** Przenośność między frameworkami jest cenna; wsparcie protokołów jest darmowe.
- **Mierz względem SWE-bench Pro lub swojego wewnętrznego odpowiednika Pro.** Verified jest skontaminowany.
- **Zapłać podatek weryfikacyjny.** Niezależny weryfikator kosztuje ~20-30% twojego budżetu tokenów i kupuje mierzalną poprawność.
- **Wdrażaj tęczowo długotrwałych agentów.** Oczekuj, że wielogodzinne uruchomienia agentów będą rutynowe.
- **Czytaj WMAC 2026 i artykuły uzupełniające MAST.** Dziedzina porusza się szybko.

## Ćwiczenia

1. Przeczytaj kompleksowo post o systemie badawczym Anthropic. Zidentyfikuj trzy decyzje projektowe, które zmieniłyby się, gdybyś zastąpił Opus 4 mniejszym modelem (np. Haiku 4).
2. Przeczytaj MetaGPT Sekcje 3-4 (arXiv:2308.00352). Zakoduj jeden SOP ze swojej własnej domeny (nie oprogramowania) jako prompty ról. Ile ról implikuje ten SOP?
3. Przeczytaj ChatDev (arXiv:2307.07924). Zidentyfikuj mechanizm "komunikatywnej dehallucynacji." Zaimplementuj go w jednym ze swoich istniejących systemów wieloagentowych.
4. Przeczytaj o OpenClaw i Moltbook. Wybierz jeden konkretny tryb awarii, który pojawił się w skali populacyjnej, a który nie pojawiłby się w systemie 5-agentowym. Jak byś się przed nim zabezpieczył?
5. Wybierz swój obecny projekt wieloagentowy. Które z trzech studiów przypadku jest najbliższą referencją? Których decyzji projektowych z tego studium przypadku JESZCZE NIE przyjąłeś? Zapisz jedną, którą przyjmiesz w tym kwartale.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|----------------|------------------------|
| Anthropic Research | "Referencja nadzorcy" | Claude Opus 4 + subagenci Sonnet 4; 15x tokenów; +90.2% nad pojedynczym agentem. |
| MetaGPT | "SOP jako prompty" | Dekompozycja ról dla inżynierii oprogramowania; `Code = SOP(Team)`. |
| ChatDev | "Agenci jako role" | Projektant / programista / recenzent / tester; komunikatywna dehallucynacja. |
| MacNet | "Skalowanie ChatDev przez DAG" | arXiv:2406.07155; 1000+ agentów przez jawny routing DAG. |
| OpenClaw | "Lokalne agenty ReAct-loop" | Projekt Steinbergera; 247k gwiazdek do marca 2026. |
| Moltbook | "Sieć społecznościowa tylko dla agentów" | 2.3M kont agentów; przejęty przez Meta marzec 2026. |
| Wdrożenie tęczowe | "Wiele wersji równocześnie" | Utrzymuj stare wersje środowiska uruchomieniowego dla aktywnych długotrwałych agentów. |
| Komunikatywna dehallucynacja | "Pytaj przed odpowiadaniem" | Agenci proszą o szczegóły od rówieśników zamiast zgadywać. |
| WMAC 2026 | "Warsztat AAAI" | Punkt centralny społeczności w kwietniu 2026 dla koordynacji wieloagentowej. |

## Dalsza lektura

- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — produkcyjna referencja nadzorca-pracownik
- [MetaGPT — Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) — dekompozycja ról przez SOP
- [ChatDev — Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) — komunikatywna dehallucynacja
- [MacNet — scaling role-based agents to 1000+](https://arxiv.org/abs/2406.07155) — skalowanie oparte na DAG
- [OpenClaw na Wikipedii](https://en.wikipedia.org/wiki/OpenClaw) — przegląd ekosystemu
- [WMAC 2026](https://multiagents.org/2026/) — AAAI 2026 Bridge Program Workshop on Multi-Agent Coordination
- [Dokumentacja LangGraph](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — lider produkcji
- [Dokumentacja CrewAI](https://docs.crewai.com/en/introduction) — framework oparty na rolach