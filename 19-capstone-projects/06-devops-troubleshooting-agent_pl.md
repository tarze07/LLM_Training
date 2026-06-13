# Capstone 06 — Agent Rozwiązywania Problemów DevOps dla Kubernetes

> AWS DevOps Agent trafił do GA, Resolve AI opublikował swoje playbooki K8s, NeuBird zademonstrował monitorowanie semantyczne, a Metoro powiązał AI SRE z SLO na usługę. Produkcyjny kształt jest ustalony: webhook alarmu odpala, agent czyta telemetrię, przechodzi graf obiektów K8s, rankuje hipotezy przyczyn źródłowych i publikuje podsumowanie na Slack z przyciskami zatwierdzania. Tylko do odczytu domyślnie. Każda naprawa jest bramkowana przez człowieka. Ten capstone to ten agent, oceniony na 20 syntetycznych incydentach i porównany z Agentem AWS na trzech wspólnych przypadkach.

**Type:** Capstone
**Languages:** Python (agent), TypeScript (Slack integration)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18
**Time:** 30 hours

## Problem

Narracja SRE w latach 2025-2026 brzmiała: "Agenty AI triażują incydenty, ludzie zatwierdzają naprawy." AWS DevOps Agent, Resolve AI, NeuBird, Metoro, PagerDuty AIOps wszystkie dostarczają ten kształt produkcyjnie. Agent czyta metryki Prometheus, logi Loki, ślady Tempo, kube-state-metrics i graf wiedzy obiektów K8s. Produkuje rankingową hipotezę przyczyny źródłowej z cytowaniami telemetrycznymi w mniej niż pięć minut. Nigdy nie wykonuje destrukcyjnych komend bez wyraźnego zatwierdzenia przez człowieka przez Slack.

Większość ciężkiej pracy to zakresowanie i bezpieczeństwo, nie rozumowanie. Agent potrzebuje domyślnie tylko-do-odczytu powierzchni RBAC, utwardzonego serwera narzędzi MCP i logów audytu każdej rozważanej vs. wykonanej komendy. Musi wiedzieć, kiedy jest poza swoją głębokością i eskalować. I musi działać wystarczająco tanio, aby kaskady OOM-kill nie generowały rachunku za agenta na $5k.

## Koncepcja

Agent operuje na grafie wiedzy. Węzły to obiekty K8s (Pody, Deploymenty, Serwisy, Węzły, HPA, PVC) plus źródła telemetrii (serie Prometheus, strumienie Loki, ślady Tempo). Krawędzie kodują własność (Pod -> ReplicaSet -> Deployment), harmonogramowanie (Pod -> Node) i obserwację (Pod -> seria Prometheus). Graf jest utrzymywany świeżo przez synchronizację kube-state-metrics i próbkowany na nowo przy każdym alarmie.

Gdy alarm odpala, agent szuka przyczyn źródłowych zaczynając od dotkniętego obiektu. Przechodzi krawędzie, pobiera odpowiednie wycinki telemetrii (ostatnie 15 minut) i szkicuje hipotezę. Hipoteza jest rankowana według dowodów: ile cytowań telemetrycznych ją wspiera, jak niedawne, jak szczegółowe. Top-3 hipotezy idą na Slack z wizualizacjami ścieżek grafu i przyciskami zatwierdzania dla działań naprawczych.

Naprawa jest bramkowana. Dozwolone domyślne działania są tylko do odczytu. Destrukcyjne działania (skalowanie w dół, wycofywanie, usuwanie Podów) wymagają zatwierdzenia przez Slack; hooki wycofania ArgoCD wymagają tokena autoryzacyjnego, którego agent nigdy nie posiada. Log audytu rejestruje każdą komendę, którą agent *rozważył* — nie tylko wykonaną — więc proces przeglądu wychwytuje bliskie chybienia.

## Architektura

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## Stack

- Źródła obserwowalności: Prometheus, Loki, Tempo, kube-state-metrics
- Graf wiedzy: Neo4j (zarządzany) lub kuzu (osadzony) obiektów K8s + krawędzi telemetrycznych
- Agent: LangGraph z listą dozwolonych narzędzi na narzędzie, domyślnie tylko do odczytu
- Transport narzędzi: FastMCP przez StreamableHTTP; oddzielny serwer dla destrukcyjnych narzędzi za bramką zatwierdzania
- Modele: Claude Sonnet 4.7 do wnioskowania o przyczynach źródłowych, Gemini 2.5 Flash do podsumowywania logów
- Naprawa: webhook wycofania ArgoCD, eskalacja PagerDuty, karta zatwierdzania Slack
- Audyt: dołączany tylko strukturalny log (rozważone, wykonane, zatwierdzone, wynik)
- Wdrożenie: wdrożenie K8s z własną wąską rolą RBAC; osobna przestrzeń nazw

## Build It

1. **Indeksacja grafu.** Synchronizuj kube-state-metrics do Neo4j/kuzu co 30s. Węzły: Pod, Deployment, Node, Service, PVC, HPA. Krawędzie: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES. Nakładkowe krawędzie telemetrii: OBSERVED_BY (Pod jest obserwowany przez serię Prometheus).

2. **Odbiornik alarmów.** Endpoint FastAPI akceptujący webhooki PagerDuty lub Alertmanager. Wyodrębnij dotknięty obiekt(y) i naruszenie SLO.

3. **Powierzchnia narzędzi tylko do odczytu.** Owiń kubectl, zapytanie Prometheus, Loki logql, Tempo traceql przez FastMCP. Każde narzędzie ma wąski czasownik RBAC ("get", "list", "describe"). Brak "delete", "exec", "scale" w domyślnym serwerze.

4. **Agent przyczyn źródłowych.** LangGraph z trzema węzłami: `sample` pobiera wycinek telemetrii z ostatnich 15 minut, `walk` odpytuje graf w poszukiwaniu sąsiednich obiektów, `hypothesize` szkicuje rankingowe kandydatury na przyczyny źródłowe z cytowaniami telemetrycznymi.

5. **Punktacja dowodów.** Każda hipoteza ma wynik = świeżość * szczegółowość * odwrotność długości ścieżki grafu * liczba cytowań. Zwróć top-3.

6. **Podsumowanie Slack.** Opublikuj załącznik z hipotezą, wizualizacją ścieżki grafu (obraz podgrafu wyrenderowany po stronie serwera) i przyciskami zatwierdzania dla co najwyżej jednego działania naprawczego.

7. **Bramka naprawy.** Destrukcyjne narzędzia (skaluj w dół, wycofaj, usuń) znajdują się na drugim serwerze MCP za tokenem zatwierdzania. Agent może je wywołać tylko po zatwierdzeniu karty Slack przez człowieka.

8. **Log audytu.** JSONL tylko do dołączania: dla każdej kandydackiej komendy, zaloguj czy była rozważona, czy wykonana, kto zatwierdził. Wysyłaj do S3 codziennie.

9. **Zestaw syntetycznych incydentów.** Zbuduj 20 scenariuszy: kaskada OOMKill, flapping DNS, thrashing HPA, wypełnienie PVC, hałaśliwy sąsiad, wadliwy sidecar, złe wdrożenie ConfigMap, rotacja certyfikatów, backoff obrazu, itd. Punktuj agenta za dokładność przyczyn źródłowych i czas do hipotezy.

## Use It

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## Ship It

`outputs/skill-devops-agent.md` jest rezultatem. Mając klaster K8s i źródło alarmów, agent produkuje rankingowe hipotezy przyczyn źródłowych i przepływ naprawczy bramkowany przez Slack.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Dokładność RCA na zestawie scenariuszy | ≥80% poprawnych przyczyn źródłowych na 20 syntetycznych incydentach |
| 20 | Bezpieczeństwo | Strażnik destrukcyjnych działań nigdy nie odpala bez zatwierdzenia Slack w logu audytu |
| 20 | Czas do hipotezy | p50 poniżej 5 minut od alarmu do podsumowania Slack |
| 20 | Wyjaśnialność | Każda hipoteza ma ścieżki grafu i cytowania telemetryczne |
| 15 | Kompletność integracji | PagerDuty, Slack, ArgoCD, Prometheus działające od końca do końca |
| **100** | | |

## Ćwiczenia

1. Uruchom swojego agenta na tych samych trzech incydentach, na których demonstrowany jest AWS DevOps Agent. Opublikuj porównanie obok siebie. Raportuj, gdzie agent się różni.

2. Dodaj audyt "bliskiego chybienia", który flaguje każdą komendę, którą agent *rozważył*, a która byłaby destrukcyjna bez zatwierdzenia. Zmierz wskaźnik bliskich chybień przez tydzień.

3. Zamień model hipotez z Claude Sonnet 4.7 na samodzielnie hostowany Llama 3.3 70B. Zmierz różnicę dokładności RCA i dolarów na incydent.

4. Zbuduj filtr przyczynowy: odróżnij skorelowane skoki telemetryczne od prawdziwej przyczyny źródłowej. Wytrenuj mały klasyfikator na etykietach 20 scenariuszy.

5. Dodaj suchy przebieg wycofania: wycofanie ArgoCD na klastrze stagingowym z tym samym manifestem. Zweryfikuj plan wycofania w żywym klastrze przed przyciskiem zatwierdzania Slack.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| K8s knowledge graph | "Graf klastra" | Węzły = obiekty K8s + serie telemetrii; krawędzie = własność, harmonogramowanie, obserwacja |
| Read-only-by-default | "Zakresowane RBAC" | Konto serwisowe agenta ma tylko czasowniki get/list/describe; destrukcyjne czasowniki w osobnym serwerze za zatwierdzeniem |
| Audit log | "Rozważone vs wykonane" | Rekord tylko do dołączania każdej kandydackiej komendy, czy została uruchomiona, kto zatwierdził |
| Hypothesis ranking | "Punktacja dowodów" | Świeżość × szczegółowość × odwrotność długości ścieżki grafu × liczba cytowań |
| Slack approval card | "Bramka HITL" | Interaktywna wiadomość Slack z przyciskami naprawy; agent nie może kontynuować dopóki człowiek nie kliknie |
| Telemetry citation | "Wskaźnik dowodów" | Zapytanie Prometheus, selektor Loki lub URL śladu Tempo, które wspiera twierdzenie |
| MTTR | "Czas do rozwiązania" | Czas od odpalenia alarmu do odzyskania SLO |

## Dalsza lektura

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — the canonical 2026 reference
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — the competitor reference
- [NeuBird semantic monitoring](https://www.neubird.ai) — semantic-graph approach
- [Metoro AI SRE](https://metoro.io) — SLO-first production framing
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — the cluster-state source
- [LangGraph](https://langchain-ai.github.io/langgraph/) — reference agent orchestrator
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP server framework
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — the gated remediation target