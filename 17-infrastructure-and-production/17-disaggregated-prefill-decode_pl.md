# Rozdzielone Prefill/Decode — NVIDIA Dynamo i llm-d

> Prefill jest ograniczony obliczeniowo; decode jest ograniczony pamięciowo. Uruchamianie obu na tym samym GPU marnuje jeden zasób. Rozdzielenie dzieli je na osobne pule i przesyła KV cache między nimi przez NIXL (RDMA/InfiniBand lub TCP fallback). NVIDIA Dynamo (ogłoszenie GTC 2025, GA 1.0) działa nad vLLM/SGLang/TRT-LLM — jego Planner Profiler + SLA Planner automatycznie dopasowują proporcje prefill:decode do SLO. NVIDIA publikuje przyrosty przepustowości — developer.nvidia.com (2025-06) pokazuje ~6x poprawę dla DeepSeek-R1 MoE na GB200 NVL72 + Dynamo w reżimie średniej latencji, a strona produktowa Dynamo (developer.nvidia.com, bez daty) reklamuje do 50x przepustowości MoE na GB300 NVL72 + Dynamo vs Hopper. Wartość "30x" to agregat społecznościowy dla pełnego stosu Blackwell + Dynamo + DeepSeek-R1; nie znaleźliśmy pojedynczego źródła mówiącego dokładnie 30x, więc traktuj to jako szacunek kierunkowy. llm-d (Red Hat + AWS) jest natywny dla Kubernetes: prefill / decode / router jako niezależne Service z HPA na rolę. llm-d 0.5 dodaje hierarchiczne odciążanie KV, routing LoRA uwzględniający cache, sieć UCCL, scale-to-zero. Ekonomika: wewnętrzne zestawienie wielu ujawnień klientów sugeruje 30–40% oszczędności na wydatkach inferencyjnych rzędu $2M (tj. $600-800K/rok) przy przejściu z serwowania kolokowanego na rozdzielone z Dynamo przy stałym SLA; konkretna wartość $2M→$600-800K to wewnętrzny agregat, a nie pojedyncze opublikowane studium przypadku — używaj jako kotwicy rzędu wielkości, a nie cytowanego źródła. Krótkie prompty (<512 tokenów, krótkie wyjście) nie uzasadniają kosztu transferu.

**Type:** Learn
**Languages:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75 minutes

## Learning Objectives

- Wyjaśnić, dlaczego prefill i decode mają różne optymalne alokacje GPU i określić ilościowo straty przy kolokacji.
- Narysować diagram architektury rozdzielonej: pula prefill, pula decode, transfer KV przez NIXL, router.
- Wymienić warunek, przy którym rozdzielenie NIE jest opłacalne (krótkie prompty, krótkie wyjścia).
- Odróżnić NVIDIA Dynamo (ponad stosem) od llm-d (natywny Kubernetes) i dopasować każdy do kontekstu operacyjnego.

## The Problem

Uruchamiasz Llama 3.3 70B na 8 H100. Przy mieszanym obciążeniu (długie prompty + krótkie wyjścia) GPU są bezczynne podczas decode, ponieważ większość obliczeń została zużyta na prefill. Przy innym obciążeniu (krótkie prompty + długie wyjścia) dzieje się odwrotnie. Kolokacja prefill + decode oznacza, że nadmiarowo zapewniasz oba.

Wpływ na budżet: 20-40% czasu GPU jest marnowane na niewłaściwym zasobie. Kupujesz obliczenia H100 do uruchamiania decode ograniczonego pamięcią lub przepustowość HBM H100 do uruchamiania prefill ograniczonego obliczeniowo. Oba to kosztowne marnotrawstwo.

Rozdzielenie dzieli prefill i decode na osobne pule, każdą dopasowaną do swojego wąskiego gardła. KV cache jest przesyłane z puli prefill do puli decode przez szybkie łącze.

## The Concept

### Dlaczego wąskie gardła się różnią

**Prefill** — uruchom transformer na pełnym wejściowym prompcie w jednym przejściu w przód. Dominują mnożenia macierzy; ograniczone obliczeniowo. H100 FP8 daje ~2000 TFLOPS użytecznej przepustowości. Wydajność wsadowa jest dobra — jedno przejście przetwarza wiele tokenów.

**Decode** — generuj jeden token na raz, czytając pełne wagi każdej iteracji. Ograniczone przepustowością pamięci. HBM3 daje ~3 TB/s. Wydajność wsadowa jest dobra tylko przy wysokiej współbieżności — odczyt wag amortyzuje się w całej partii.

Kolokacja: kupujesz GPU zoptymalizowane pod oba. H100 jest dobry w obu, ale kosztuje tyle samo. Na dużą skalę chcesz pulę prefill na H100 / obliczeniowo intensywnych; pulę decode na H200 / pamięciowo intensywnych lub z agresywną kwantyzacją.

### Architektura

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL to międzywęzłowy transport NVidii. Używa RDMA/InfiniBand gdy dostępne, TCP jako fallback. Latencja transferu jest realna — typowo 20-80 ms dla KV cache 4K-tokenowego promptu na 70B FP8. Dlatego krótkie prompty nie uzasadniają rozdzielenia: koszt transferu przewyższa oszczędności.

### Dynamo vs llm-d

**NVIDIA Dynamo** (ogłoszone GTC 2025, GA 1.0):
- Działa nad vLLM, SGLang, TRT-LLM jako orkiestrator.
- Planner Profiler mierzy obciążenie, SLA Planner automatycznie konfiguruje proporcje prefill:decode.
- Rdzeń w Rust, rozszerzalność w Pythonie.
- Przyrosty przepustowości: NVIDIA raportuje 6x dla DeepSeek-R1 MoE na GB200 NVL72 + Dynamo w reżimie średniej latencji (developer.nvidia.com, 2025-06); doniesienia społeczności o "do 30x" na pełnym stosie Blackwell + Dynamo + DeepSeek-R1 nie mają pojedynczego źródła i należy je traktować jako kierunkowe.
- GB300 NVL72 + Dynamo: do 50x przepustowości MoE vs Hopper według strony produktowej Dynamo (developer.nvidia.com, bez daty).

**llm-d** (Red Hat + AWS, natywny Kubernetes):
- Prefill / decode / router jako niezależne Kubernetes Services.
- HPA na rolę z sygnałami głębokości kolejki (prefill) / wykorzystania KV (decode).
- `topologyConstraint packDomain: rack` pakuje kliki prefill+decode na tym samym racku dla szybkiego transferu KV.
- llm-d 0.5 (2026): hierarchiczne odciążanie KV, routing LoRA uwzględniający cache, sieć UCCL, scale-to-zero.

Użyj Dynamo, jeśli chcesz zarządzanego orkiestratora ponad stosem. Użyj llm-d, jeśli chcesz prymitywów natywnych dla Kubernetes i jesteś zaangażowany w ekosystem CNCF.

### Ekonomika

Wewnętrzny agregat (nie pojedyncze opublikowane studium przypadku — kotwica rzędu wielkości):

- $2M/rok wydatków na inferencję przy serwowaniu kolokowanym.
- Przejście na rozdzielone z Dynamo.
- Taki sam wolumen żądań, to samo SLA P99 latencji.
- Raportowane oszczędności: $600K–$800K/rok (30–40% redukcji).
- Żaden nowy sprzęt.

Syntetyzujemy tę wartość z wielu ujawnień klientów, a nie z pojedynczego cytowalnego studium przypadku; najbliższym opublikowanym punktem danych jest 2x szybszy TTFT / 61% wyższa przepustowość Basetena z routingiem KV Dynamo (baseten.co, 2025-10) oraz projekcja VAST + CoreWeave o 60–130% więcej tokenów/$ przy 40–60% współczynniku trafień KV (vastdata.com, 2025-12). Oszczędności pochodzą z odpowiedniego wymiarowania każdej puli; obciążenia z dominującym prefillem (RAG z prefiksami 8K+) korzystają bardziej niż zrównoważone.

### Kiedy NIE rozdzielać

- Prompty < 512 tokenów i wyjścia < 200 tokenów: koszt transferu przewyższa zysk.
- Mały klaster (< 4 GPU): niewystarczająca różnorodność pul.
- Zespół nie może operować dwoma pulami GPU ze skalowaniem na rolę: Dynamo pomaga, ale nie trywialnie.
- BrakRDMA: koszt transferu TCP jest wyższy.

### Router integruje się z Phase 17 · 11

Routery rozdzielone są świadome KV cache (Phase 17 · 11). Żądanie trafia do puli decode przechowującej jego prefiks — jeśli brak dopasowania, przepływa prefill → decode. Współczynnik trafień i rozdzielenie kumulują się — router świadomy cache decyduje, czy nowy prefill jest w ogóle potrzebny.

### MoE na Blackwell to miejsce, gdzie są prawdziwe liczby

GB300 NVL72 + Dynamo pokazuje 50x przepustowości MoE w porównaniu z bazowymi Hopper. Routing ekspertów MoE jest intensywny obliczeniowo przy prefill, ale intensywny pamięciowo przy decode (caching ekspertów), więc rozdzielenie to podwójny zysk. Serwowanie modeli granicznych w 2026 jest zdominowane przez MoE (DeepSeek-V3, przyszłe warianty GPT-5).

### Liczby, które warto zapamiętać

Wartości benchmarków dryfują — NVIDIA i stos inferencyjny publikują zaktualizowane wyniki co kwartał. Sprawdź ponownie przed cytowaniem.

- DeepSeek-R1 na GB200 NVL72 + Dynamo: ~6x przepustowości vs bazowa w reżimie średniej latencji (developer.nvidia.com, 2025-06); twierdzenia społeczności o "do 30x" na pełnym stosie Blackwell + Dynamo to agregaty kierunkowe bez pojedynczego źródła.
- GB300 NVL72 + Dynamo: do 50x przepustowości MoE vs Hopper (developer.nvidia.com, bez daty).
- Kotwica oszczędności (wewnętrzny agregat, nie pojedyncze studium przypadku): $600-800K/rok z $2M rocznych wydatków przy stałym SLA.
- Próg rozdzielenia: prompty >512 tokenów + wyjścia >200 tokenów.
- Transfer KV przez NIXL: 20-80 ms dla 4K-promptu KV na 70B FP8.

## Use It

`code/main.py` symuluje serwowanie kolokowane vs rozdzielone. Raportuje przepustowość, koszt na żądanie i punkt przecięcia długości promptu.

## Ship It

Ta lekcja produkuje `outputs/skill-disaggregation-decider.md`. Na podstawie obciążenia i klastra decyduje, czy rozdzielać.

## Exercises

1. Uruchom `code/main.py`. Przy jakiej długości promptu rozdzielenie pokonuje kolokację?
2. Zaprojektuj pulę prefill i pulę decode dla usługi RAG z P99 długością prefiksu 8K, wyjściem 300.
3. Dynamo vs llm-d: wybierz jeden dla sklepu czysto Kubernetesowego bez preferencji runtime'u Python.
4. Oblicz koszt transferu KV: 4K prefill na 70B FP8 = ~500 MB KV. Przy RDMA 100 GB/s, transfer = 5 ms. Przy TCP 10 GB/s = 50 ms. Który ma znaczenie dla Twojego SLA?
5. Routing ekspertów MoE zmienia wzorce dostępu do KV. Jak zachowuje się rozdzielenie z MoE, które aktywuje różnych ekspertów na token?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Disaggregated serving | "split prefill/decode" | Separate GPU pools for each phase |
| NIXL | "NVIDIA transport" | Dynamo's inter-node KV transfer (RDMA/TCP) |
| NVIDIA Dynamo | "the orchestrator" | Stack-above coordinator for vLLM/SGLang/TRT-LLM |
| llm-d | "Kubernetes native" | Red Hat + AWS K8s disaggregated stack |
| Planner Profiler | "Dynamo auto-config" | Measures workload, configures pool ratios |
| SLA Planner | "Dynamo policy" | Auto-rate-matches prefill:decode to meet SLOs |
| `packDomain: rack` | "llm-d topology" | Pack prefill+decode on same rack for fast KV |
| UCCL | "unified collective" | llm-d 0.5 networking layer for scale-to-zero |
| MoE expert routing | "expert per token" | DeepSeek-V3 pattern; disaggregation helps |

## Further Reading

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)