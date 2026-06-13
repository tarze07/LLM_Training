# Autoskalowanie GPU na Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling

> Trzy warstwy, nie jedna. Karpenter dynamicznie provisionuje węzły (poniżej minuty, 40% szybciej niż Cluster Autoscaler). KAI Scheduler obsługuje gang scheduling, świadomość topologii i hierarchiczne kolejki — zapobiega pułapce częściowej alokacji 7-z-8, gdzie siedem węzłów czeka i pali pieniądze na brakującym GPU. Skalery na poziomie aplikacji (NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler) skalują na sygnałach specyficznych dla inferencji — głębokość kolejki, wykorzystanie pamięci podręcznej KV — a nie cyklu pracy CPU/DCGM. Klasyczna pułapka HPA polega na tym, że `DCGM_FI_DEV_GPU_UTIL` jest miarą cyklu pracy: 100% może oznaczać 10 żądań lub 100. vLLM prealokuje pamięć podręczną KV, więc pamięć nigdy nie wyzwala skalowania w dół. Ta lekcja uczy komponować trzy warstwy i unikać domyślnej polityki Karpenter `WhenEmptyOrUnderutilized`, która kończy działające zadania GPU w trakcie inferencji.

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## Learning Objectives

- Narysuj diagram trzech warstw skalowania (provisioning węzłów, gang scheduling, poziom aplikacji) i nazwij narzędzie używane na każdej warstwie.
- Wyjaśnij, dlaczego `DCGM_FI_DEV_GPU_UTIL` jest złym sygnałem HPA dla vLLM i wymień dwa zamienniki (głębokość kolejki, wykorzystanie pamięci podręcznej KV).
- Opisz gang scheduling i tryb awarii częściowej alokacji, któremu zapobiega KAI Scheduler (7 z 8 GPU bezczynnych).
- Nazwij politykę konsolidacji Karpenter (`WhenEmptyOrUnderutilized`), która kończy działające zadania GPU i podaj bezpieczną alternatywę na 2026.

## The Problem

Twój zespół wdraża usługę serwowania LLM na Kubernetes. Konfigurujesz HPA z `DCGM_FI_DEV_GPU_UTIL` jako sygnałem. Usługa utrzymuje się na 100% wykorzystania w godzinach pracy. HPA nigdy się nie skaluje w górę — już myśli, że jesteś pełny. Dodajesz replikę ręcznie; TTFT spada. HPA nadal nie skaluje. Sygnał cię okłamuje.

Osobno używasz Cluster Autoscaler dla węzłów. Prompt o 1M tokenów pojawia się o 2 w nocy; klaster spędza 3 minuty na provisionowaniu węzła, a żądanie przekracza limit czasu.

Osobno, wdrażasz model 70B wymagający 8 GPU na 2 węzłach. Klaster ma 7 wolnych GPU i 1 rozłożone na 3 węzłach. Cluster Autoscaler provisionuje węzeł dla 1 brakującego GPU. Siedem węzłów czeka 4 minuty paląc pieniądze, podczas gdy Kubernetes uruchamia ostatnie GPU.

Trzy warstwy, trzy różne tryby awarii. Świadome GPU autoskalowanie w 2026 to nie „włącz HPA". To komponowanie provisionowania węzłów, gang schedulingu i autoskalowania na sygnałach aplikacyjnych.

## The Concept

### Warstwa 1 — provisionowanie węzłów (Karpenter)

Karpenter monitoruje oczekujące pody i provisionuje węzły w ciągu ~45-60 sekund (Cluster Autoscaler zazwyczaj zajmuje 90-120 sekund dla węzłów GPU). Wybiera typy instancji dynamicznie zgodnie z ograniczeniem `NodePool` — jeśli Twój pod potrzebuje 8 H100, a klaster nie ma pasującego węzła, Karpenter provisionuje go bezpośrednio zamiast skalować istniejącą grupę.

**Pułapka konsolidacji**: domyślne `consolidationPolicy: WhenEmptyOrUnderutilized` Karpenter jest niebezpieczne dla pul GPU. Zakończy działający węzeł GPU, aby migrować pody do tańszej, odpowiednio dobranej instancji. Dla obciążeń inferencyjnych oznacza to wywłaszczenie działających żądań i ponowne załadowanie modelu 70B na nowym węźle. Strata to minuty mocy obliczeniowej plus nieudane żądania.

Bezpieczne ustawienie dla pul GPU:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

Pozwala Karpenter konsolidować naprawdę puste węzły po godzinie, ale nigdy nie wywłaszcza działającego zadania.

### Warstwa 2 — gang scheduling (KAI Scheduler)

KAI Scheduler (projekt „Karp" następnie przemianowany) obsługuje to, czego domyślny kube-scheduler nie robi:

**Gang scheduling** — harmonogram wszystko-albo-nic. Rozproszony pod inferencyjny wymagający 8 GPU albo wszystkie 8 startują razem, albo żaden. Bez tego pojawia się pułapka częściowej alokacji: 7 z 8 podów startuje, czeka w nieskończoność, pali pieniądze.

**Świadomość topologii** — wie, które GPU współdzielą NVLink, które są na tym samym racku, które mają InfiniBand między sobą. Umieszcza pody odpowiednio. Obciążenie tensor-parallel DeepSeek-V3 67B musi pozostać w jednej domenie NVLink; KAI Scheduler to respektuje.

**Hierarchiczne kolejki** — wiele zespołów konkuruje o tę samą pulę GPU z priorytetem i limitem. Produkcyjny wąski gardło zespołu A może zostać wywłaszczone przez zadanie treningowe zespołu B tylko jeśli reguły priorytetu na to pozwalają.

KAI jest wdrażany obok kube-scheduler jako dodatkowy scheduler; adnotujesz obciążenia, aby go używać. Zarówno Ray, jak i vLLM production-stack integrują się.

### Warstwa 3 — sygnały na poziomie aplikacji

**Pułapka HPA**: `DCGM_FI_DEV_GPU_UTIL` to metryka cyklu pracy — mierzy, czy GPU wykonywało pracę w każdym przedziale próbkowania. 100% wykorzystania może oznaczać 10 równoczesnych żądań lub 100; GPU było zajęte w obu przypadkach. Skalowanie na cyklu pracy to skalowanie na ślepo.

Co gorsza, vLLM i podobne silniki prealokują pamięć podręczną KV (do `--gpu-memory-utilization`). Użycie pamięci pozostaje blisko 90% nawet przy jednym żądaniu. HPA oparte na pamięci nigdy nie skaluje w dół.

**Sygnały zastępcze 2026**:

- Głębokość kolejki (liczba żądań oczekujących na prefill).
- Wykorzystanie pamięci podręcznej KV (jaka część bloków jest przydzielona do aktywnych sekwencji).
- P99 TTFT na replikę (Twój sygnał SLA).
- Goodput (żądania spełniające wszystkie SLO na sekundę).

NVIDIA Dynamo Planner i llm-d Workload Variant Autoscaler konsumują te sygnały i skalują repliki. Zastępują HPA całkowicie dla serwowania LLM.

### Kiedy czego użyć

| Decyzja skalowania | Narzędzie |
|----------------|------|
| Dodaj/usuń węzły | Karpenter |
| Harmonogram zadań multi-GPU | KAI Scheduler |
| Dodaj/usuń repliki | Dynamo Planner / llm-d WVA (lub niestandardowe HPA na głębokości kolejki) |
| Wybierz typ GPU | Karpenter NodePool |
| Wywłaszcz niski priorytet | Kolejki KAI Scheduler |

### Rozdzielony prefill/decode komplikuje wszystko

Jeśli uruchamiasz rozdzielony prefill/decode (Faza 17 · 17), masz dwie klasy podów z różnymi wyzwalaczami skalowania: pody prefill skalują się na głębokości kolejki, pody decode skalują się na ciśnieniu pamięci podręcznej KV. llm-d udostępnia je jako oddzielne `Services` z HPA na rolę. Nie próbuj umieszczać pojedynczego HPA przed obydwoma.

### Zimny start ma tu znaczenie

Mitigacja zimnego startu (Faza 17 · 10) jest tam, gdzie czas provisionowania węzła staje się widoczny dla użytkownika. Rozgrzewka Karpenter 45-60 sekund plus ładowanie modelu 20 GB plus inicjalizacja silnika oznacza, że żądanie z zera zajmuje 2-5 minut. Utrzymuj ciepłą pulę (`min_workers=1`) dla ścieżek krytycznych dla SLO, lub używaj checkpointingu w stylu Modal na warstwie aplikacji.

### Liczby, które powinieneś zapamiętać

- Provisionowanie węzłów Karpenter: ~45-60s vs Cluster Autoscaler ~90-120s (węzły GPU).
- KAI Scheduler zapobiega marnowaniu częściowej alokacji — pułapka 7-z-8.
- `DCGM_FI_DEV_GPU_UTIL` jako sygnał HPA: uszkodzony; używaj głębokości kolejki lub wykorzystania KV.
- Karpenter `WhenEmptyOrUnderutilized`: kończy działające zadania GPU. Używaj `WhenEmpty + consolidateAfter: 1h` dla inferencji.

```figure
autoscaling
```

## Use It

`code/main.py` symuluje trójwarstwowy autoscaler na zrywnym obciążeniu GPU. Porównuje naiwne HPA (cykl pracy), HPA na głębokości kolejki i skalowanie z gang scheduling KAI. Raportuje niespełnione żądania, minuty bezczynnych GPU i złożony wynik.

## Ship It

Ta lekcja produkuje `outputs/skill-gpu-autoscaler-plan.md`. Biorąc topologię klastra, kształt obciążenia i SLO, projektuje trójwarstwowy plan autoskalowania.

## Exercises

1. Uruchom `code/main.py`. Przy zrywnym obciążeniu, ile żądań naiwne HPA na cyklu pracy upuszcza, które HPA na głębokości kolejki łapie? Skąd bierze się różnica?
2. Zaprojektuj Karpenter NodePool dla klastra serwującego Llama 3.3 70B FP8 na H100 SXM5. Określ `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter` i skażenie, które utrzymuje obciążenia nie-GPU z dala od tych węzłów.
3. Twój zespół zgłasza, że wdrożenia utknęły w stanie Oczekujące, ponieważ „GPU dostępne, ale pod nie chce się zaplanować". Zdiagnozuj — czy to Karpenter, kube-scheduler, czy KAI Scheduler? Które metryki potwierdzają?
4. Wybierz sygnał do autoskalowania rozdzielonych podów prefill i inny sygnał dla podów decode. Uzasadnij oba.
5. Oblicz koszt pułapki konsolidacji `WhenEmptyOrUnderutilized` na usłudze produkcyjnej 24x7, która średnio ma 60 zdarzeń upuszczania żądań/dzień przy P99 TTFT > 10s.

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Karpenter | „provisioner węzłów" | Autoscaler węzłów Kubernetes; provisionowanie poniżej minuty |
| Cluster Autoscaler | „stary skaler" | Poprzednik autoskalera węzłów Kubernetes; wolniejszy, oparty na grupach |
| KAI Scheduler | „scheduler GPU" | Dodatkowy scheduler dla gang + topologia + kolejki |
| Gang scheduling | „wszystko albo nic" | Zaplanuj N podów atomowo lub odłóż wszystkie |
| Topology awareness | „świadomy racka" | Umieszczaj pody na podstawie NVLink/IB/umiejscowienia w racku |
| `DCGM_FI_DEV_GPU_UTIL` | „wykorzystanie GPU" | Metryka cyklu pracy; NIE sygnał skalowania dla LLM |
| Queue depth | „oczekujące żądania" | Poprawny sygnał HPA dla skalowania związanego z prefill |
| KV cache utilization | „ciśnienie pamięci" | Poprawny sygnał HPA dla skalowania związanego z decode |
| Consolidation | „konsolidacja Karpenter" | Zakończenie węzła do tańszego typu instancji |
| `WhenEmpty + 1h` | „bezpieczna konsolidacja" | Polityka, która nie wywłaszcza działających zadań GPU |

## Further Reading

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — dokumentacja projektowa i przykłady konfiguracji.
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — semantyka polityki konsolidacji i bezpieczne domyślne dla GPU.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — sygnały skalowania Dynamo Planner.
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — wzorzec integracji Ray.
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — wskazówki specyficzne dla zarządzanego Kubernetes.
- [llm-d GitHub](https://github.com/llm-d/llm-d) — projekt Workload Variant Autoscaler.

(End of file - total 139 lines)