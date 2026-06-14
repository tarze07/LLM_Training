# Ścieżka Nauki: Projektowanie Zespołów Wieloagentowych (Firma Software'owa)

Ten dokument zawiera ustrukturyzowany plan nauki opart na katalogu szkoleń w repozytorium [LLM_Training](file:///home/tarze07/Poligon/LLM_Training). Przygotuje Cię on teoretycznie i praktycznie do zaprojektowania i wdrożenia autonomicznego zespołu agentów (typu *software house* / *agentic swarm*).

---

## Krok 1: Fundamenty pojedynczego agenta (Pętla i Narzędzia)

Przed przejściem do systemów wieloagentowych musisz dokładnie zrozumieć, jak działa pojedynczy agent – w jaki sposób planuje działania, jak korzysta z zewnętrznych narzędzi (shell, git, kompilatory) oraz jak zarządza swoim kontekstem.

* **[14-agent-engineering/01-the-agent-loop_pl.md](file:///home/tarze07/Poligon/LLM_Training/14-agent-engineering/01-the-agent-loop_pl.md)**
  * **Dlaczego warto:** Poznasz podstawową pętlę agenta (obserwacja $\rightarrow$ myślenie $\rightarrow$ działanie).
* **[14-agent-engineering/02-rewoo-plan-and-execute_pl.md](file:///home/tarze07/Poligon/LLM_Training/14-agent-engineering/02-rewoo-plan-and-execute_pl.md)**
  * **Dlaczego warto:** Zrozumiesz wzorce planowania i dekompozycji złożonych zadań (np. ReWOO / Plan-and-Execute).
* **[14-agent-engineering/06-tool-use-and-function-calling_pl.md](file:///home/tarze07/Poligon/LLM_Training/14-agent-engineering/06-tool-use-and-function-calling_pl.md)**
  * **Dlaczego warto:** Nauczysz się poprawnie projektować interfejsy narzędzi, z których korzystać będą agenci programiści i testerzy.
* **[14-agent-engineering/07-memory-virtual-context-memgpt_pl.md](file:///home/tarze07/Poligon/LLM_Training/14-agent-engineering/07-memory-virtual-context-memgpt_pl.md)**
  * **Dlaczego warto:** Opanujesz mechanizmy pamięci wirtualnej (MemGPT), niezbędne do pracy nad dużymi projektami bez utraty kontekstu.

---

## Krok 2: Wieloagentowość i specjalizacja ról

Ten etap odpowiada bezpośrednio na strukturę "firmy". Dowiesz się, dlaczego jeden agent nie poradzi sobie z całą bazą kodu oraz jak przypisywać wąskie, wyspecjalizowane role (np. Product Manager, Architekt, Programista, Tester, Recenzent).

* **[16-multi-agent-and-swarms/01-why-multi-agent_pl.md](file:///home/tarze07/Poligon/LLM_Training/16-multi-agent-and-swarms/01-why-multi-agent_pl.md)**
  * **Dlaczego warto:** Tłumaczy problem nasycenia kontekstu pojedynczego agenta i wskazuje, kiedy podział na role jest opłacalny.
* **[16-multi-agent-and-swarms/08-role-specialization_pl.md](file:///home/tarze07/Poligon/LLM_Training/16-multi-agent-and-swarms/08-role-specialization_pl.md)**
  * **Dlaczego warto:** **Najważniejsza lekcja dla Twojego celu.** Analizuje struktury SOP (Standard Operating Procedures) w inżynierii oprogramowania na przykładzie **MetaGPT** (`Code = SOP(Team)`) oraz **ChatDev** (komunikatywna dehalucynacja – czyli agenci proszący inne role o doprecyzowanie wymagań zamiast wymyślania kodu).
* **[16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern_pl.md](file:///home/tarze07/Poligon/LLM_Training/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern_pl.md)**
  * **Dlaczego warto:** Poznasz wzorzec orkiestratora/nadzorcy, w którym agent-menedżer koordynuje pracę agentów wykonawczych.
* **[16-multi-agent-and-swarms/06-hierarchical-architecture_pl.md](file:///home/tarze07/Poligon/LLM_Training/16-multi-agent-and-swarms/06-hierarchical-architecture_pl.md)**
  * **Dlaczego warto:** Uczy projektowania struktur hierarchicznych (np. CEO $\rightarrow$ PM $\rightarrow$ Programiści).

---

## Krok 3: Protokoły komunikacji i gotowe frameworki

System wieloagentowy potrzebuje ścisłych standardów wymiany informacji, formatów wiadomości i bibliotek uruchomieniowych.

* **[16-multi-agent-and-swarms/03-communication-protocols_pl.md](file:///home/tarze07/Poligon/LLM_Training/16-multi-agent-and-swarms/03-communication-protocols_pl.md)**
  * **Dlaczego warto:** Dowiesz się o protokole komunikacji agent-do-agenta (A2A), standardach MCP (Model Context Protocol) oraz ACP (audytowanie trajektorii rozumowania).
* **[16-multi-agent-and-swarms/13-shared-memory-blackboard_pl.md](file:///home/tarze07/Poligon/LLM_Training/16-multi-agent-and-swarms/13-shared-memory-blackboard_pl.md)**
  * **Dlaczego warto:** Analizuje wzorzec tablicy (Blackboard), czyli współdzielonego stanu pamięci, niezbędnego do wspólnej edycji kodu.
* **[14-agent-engineering/14-autogen-actor-model_pl.md](file:///home/tarze07/Poligon/LLM_Training/14-agent-engineering/14-autogen-actor-model_pl.md)**
  * **Dlaczego warto:** Poznasz framework AutoGen firmy Microsoft, bazujący na modelu aktorów i bezpośredniej komunikacji.
* **[14-agent-engineering/15-crewai-role-based-crews_pl.md](file:///home/tarze07/Poligon/LLM_Training/14-agent-engineering/15-crewai-role-based-crews_pl.md)**
  * **Dlaczego warto:** Szczegółowy opis CrewAI (Agenci, Zadania, Procesy oraz Flows) – bardzo popularnego narzędzia do budowy zespołów opartych na rolach.

---

## Krok 4: Specyfika agentów kodujących i bezpieczeństwo środowiska

Generowanie i uruchamianie kodu przez AI wymaga sandboksowania, trwałości sesji oraz weryfikacji rezultatów.

* **[15-autonomous-systems/09-coding-agent-landscape_pl.md](file:///home/tarze07/Poligon/LLM_Training/15-autonomous-systems/09-coding-agent-landscape_pl.md)**
  * **Dlaczego warto:** Pokazuje współczesną architekturę systemów kodujących (Devin, OpenHands, Cline, Claude Code) oraz różnice między wywołaniami JSON a CodeAct (wykonywaniem kodu bezpośrednio w kontenerze).
* **[15-autonomous-systems/12-durable-execution_pl.md](file:///home/tarze07/Poligon/LLM_Training/15-autonomous-systems/12-durable-execution_pl.md)**
  * **Dlaczego warto:** Uczy tworzenia odpornych na awarie pętli (Durable Execution), co zapobiega przerwaniu wielogodzinnych zadań przy problemach sieciowych.
* **[15-autonomous-systems/15-propose-then-commit_pl.md](file:///home/tarze07/Poligon/LLM_Training/15-autonomous-systems/15-propose-then-commit_pl.md)** oraz **[15-autonomous-systems/16-checkpoints-rollback_pl.md](file:///home/tarze07/Poligon/LLM_Training/15-autonomous-systems/16-checkpoints-rollback_pl.md)**
  * **Dlaczego warto:** Pokazuje, jak wdrożyć bezpieczne zatwierdzanie zmian w repozytorium oraz mechanizmy cofania zmian (rollback), gdy agent wygeneruje wadliwy kod.

---

## Krok 5: Skalowanie, orkiestracja i obsługa błędów

Gdy przenosisz projekt na środowisko produkcyjne, musisz radzić sobie z asynchronicznością i specyficznymi błędami systemów LLM.

* **[16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints_pl.md](file:///home/tarze07/Poligon/LLM_Training/16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints_pl.md)**
  * **Dlaczego warto:** Uczy, jak zarządzać kolejkami wiadomości i asynchronicznymi pętlami w skalowalnych swarmach agentów.
* **[16-multi-agent-and-swarms/23-failure-modes-mast-groupthink_pl.md](file:///home/tarze07/Poligon/LLM_Training/16-multi-agent-and-swarms/23-failure-modes-mast-groupthink_pl.md)**
  * **Dlaczego warto:** Analizuje typy awarii (np. nieskończone pętle wiadomości, kaskada halucynacji czy grupowe myślenie - groupthink) oraz sposoby zapobiegania im.
