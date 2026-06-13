# Capstone 14 — Serwer Inferencji z Dekodowaniem Spekulatywnym

> EAGLE-3 w vLLM 0.7 zapewnia 2.5-3x przepustowość na prawdziwym ruchu. P-EAGLE (AWS 2026) pchnął równoległą spekulację jeszcze dalej. SpecForge SGLanga trenował głowy szkicu na skalę. Red Hat's Speculators opublikował dopasowane szkice dla popularnych otwartych modeli. TensorRT-LLM uczynił dekodowanie spekulatywne pierwszą klasą na NVIDIA. Produkcyjny stos serwowania 2026 to vLLM lub SGLang ze szkicami rodziny EAGLE, kwantyzacją FP8 lub INT4 i HPA na queue-wait. Ten capstone polega na serwowaniu dwóch otwartych modeli przy 2.5x+ bazowej przepustowości z pełnym raportem opóźnienia ogona.

**Type:** Capstone
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Prerequisites:** Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 17 (infrastructure)
**Phases exercised:** P3 · P7 · P10 · P17
**Time:** 30 hours

## Problem

Dekodowanie spekulatywne stało się towarem w 2026 roku. Głowy szkicu EAGLE-3 trenują na ukrytych stanach modelu docelowego i przewidują N tokenów do przodu; model docelowy weryfikuje w jednym przejściu. Współczynniki akceptacji 60-80% przekładają się na 2-3x przepustowość end-to-end. vLLM 0.7 integruje to natywnie. SGLang + SpecForge daje potok treningowy. Red Hat's Speculators publikuje dopasowane szkice dla Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B.

Rzemiosło leży w operacjach serwowania, nie modelu. Współczynnik akceptacji dryfuje z dystrybucją ruchu (ShareGPT vs kod vs dane domenowe). Opóźnienie ogona przy odrzuceniu jest gorsze niż bez spekulacji — musisz raportować p99 przy wielu rozmiarach batcha, nie tylko tokeny/s w stanie ustalonym. Koszt na 1M tokenów vs Anthropic / OpenAI API jest dźwignią wiarygodności.

## Koncepcja

Dekodowanie spekulatywne ma dwie warstwy. Model **szkicu** (głowa EAGLE-3, ngram lub mniejszy model dopasowany do docelowego) proponuje k tokenów kandydackich na krok. Model **docelowy** weryfikuje wszystkie k w jednym przejściu; każdy zaakceptowany prefiks zastępuje ścieżkę zachłanną. Współczynnik akceptacji zależy od dopasowania szkicu do modelu docelowego i dystrybucji wejściowej.

EAGLE-3 bije szkice ngram na większości ruchu. P-EAGLE uruchamia równoległą spekulację dla głębszych drzew szkicu. Kompromis: opóźnienie P99 przy odrzuceniu jest wyższe, ponieważ przejście weryfikacji jest większe. Konfiguracja serwowania musi raportować opóźnienie pogrupowane według rozmiaru batcha, aby to uwidocznić.

Wdrożenie to Kubernetes. vLLM 0.7 działa jedna replika na GPU lub shard tensor-parallel. HPA automatycznie skaluje na queue-wait, a nie CPU. Kwanty FP8 (Marlin) i INT4 (AWQ) utrzymują pamięć GPU w obrębie obwiedni H100 / H200. Raport end-to-end to przepustowość, współczynnik akceptacji, p50/p99 przy batch 1/8/32 i $/1M tokenów.

## Architektura

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## Stack

- Serwowanie: vLLM 0.7 lub SGLang 0.4
- Metody spekulatywne: głowy szkicu EAGLE-3, równoległa spekulacja P-EAGLE, zapasowy ngram
- Trenowanie szkicu: SpecForge (SGLang) lub Red Hat Speculators
- Modele docelowe: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B
- Kwantyzacja: FP8 (Marlin), INT4 AWQ
- Wdrożenie: Kubernetes + NVIDIA device plugin; HPA na metryce queue-wait
- Ewaluacja: ShareGPT, MT-Bench-v2, GSM8K, HumanEval dla pomiaru akceptacji w różnych domenach
- Referencja: TensorRT-LLM speculative decoding dla baseline od dostawcy

## Build It

1. **Przygotowanie modelu docelowego.** Wybierz Llama 3.3 70B. Skwantyzuj do FP8 przez Marlin. Wdróż pod vLLM 0.7 na 1xH100 (lub 2x tensor-parallel).

2. **Źródło szkicu.** Pobierz dopasowaną głowę szkicu EAGLE-3 z Red Hat Speculators (lub wytrenuj przez SpecForge). Załaduj do konfiguracji dekodowania spekulatywnego vLLM.

3. **Liczby bazowe.** Przed spekulacją: tokeny/s przy batch 1/8/32, opóźnienie p50/p99, wykorzystanie GPU. Opublikuj.

4. **Włącz EAGLE-3.** Odwróć konfigurację; uruchom ponownie ten sam benchmark. Raportuj przyspieszenie, współczynnik akceptacji, różnicę opóźnienia ogona p99.

5. **P-EAGLE.** Włącz równoległą spekulację; zmierz głębsze drzewo szkicu vs szeregowy EAGLE-3. Raportuj punkt przegięcia, gdzie P-EAGLE pomaga vs szkodzi.

6. **Ruch domenowy.** Uruchom ShareGPT vs HumanEval vs ruch specyficzny dla domeny przez ten sam serwer. Zmierz współczynnik akceptacji na dystrybucję. Zidentyfikuj, kiedy szkice dryfują.

7. **Drugi model docelowy.** Uruchom ten sam potok na Qwen3-Coder-30B MoE. Szkic jest trudniejszy (szum routingu MoE). Raportuj.

8. **K8s HPA.** Wdróż pod K8s z HPA śledzącym `queue_wait_ms`. Zademonstruj skalowanie w górę, gdy obciążenie się potraja.

9. **Porównanie kosztów.** Oblicz $/1M tokenów vs Anthropic Claude Sonnet 4.7 i OpenAI GPT-5.4 na tej samej ewaluacji. Opublikuj.

## Use It

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## Ship It

`outputs/skill-inference-server.md` opisuje rezultat. Zmierzony stos serwowania z dekodowaniem spekulatywnym, pełny raport benchmarkowy i wdrożenie K8s.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Zmierzone przyspieszenie vs baseline | 2.5x+ przepustowość przy dopasowanej jakości na dwóch modelach |
| 20 | Współczynnik akceptacji na realistycznym ruchu | Raport współczynnika akceptacji na dystrybucję |
| 20 | Dyscyplina opóźnienia ogona p99 | p99 przy batch 1/8/32 z i bez spekulacji |
| 20 | Operacje | Wdrożenie K8s, HPA na queue-wait, płynne wdrożenie |
| 15 | Opracowanie i metodologia | Jasne wyjaśnienie, co się zmieniło i dlaczego |
| **100** | | |

## Ćwiczenia

1. Zmierz degradację współczynnika akceptacji, gdy szkic jest o jedną wersję za modelem docelowym (np. dryf Llama 3.3 -> 3.4). Zbuduj alarm monitorujący.

2. Zaimplementuj zapasowy ngram: jeśli akceptacja EAGLE-3 spadnie poniżej progu, przełącz na szkice ngram. Raportuj poprawę niezawodności.

3. Przeprowadź kontrolowany eksperyment MoE: ten sam Qwen3-Coder-30B z wstrzykniętym szumem routingu vs bez. Zmierz wrażliwość akceptacji szkicu.

4. Rozszerz na H200 (141 GB). Raportuj uzyskany margines rozmiaru modelu-na-replikę i czy możesz serwować nieskwantyzowany Llama 3.3 70B.

5. Porównaj benchmark TensorRT-LLM speculative decoding na tym samym sprzęcie H100. Raportuj, gdzie wygrywa vs vLLM.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Draft model | "Spekulator" | Mały model proponujący N tokenów do weryfikacji przez model docelowy |
| EAGLE-3 | "Architektura szkicu 2026" | Głowa szkicu trenowana na ukrytych stanach modelu docelowego; ~75% akceptacji |
| P-EAGLE | "Równoległa spekulacja" | Drzewo gałęzi szkicu zweryfikowane w jednym przejściu modelu docelowego |
| Acceptance rate | "Współczynnik trafień" | Frakcja szkicowanych tokenów zaakceptowanych bez ponownego próbkowania |
| Quantization | "FP8 / INT4" | Wagi o niższej precyzji, aby zmieścić więcej modelu w pamięci GPU |
| Queue wait | "Metryka HPA" | Czekanie żądania w kolejce oczekujących przed rozpoczęciem inferencji |
| Speculators hub | "Dopasowane szkice" | Red Hat Neural Magic hub szkiców EAGLE dla popularnych otwartych modeli |

## Dalsza lektura

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — the reference serving stack
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — parallel speculative decoding paper + integration
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft-head training pipeline
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — aligned draft hub
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — vendor alternative
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — commercial reference
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — the method paper
- [vLLM repository](https://github.com/vllm-project/vllm) — code and benchmarks