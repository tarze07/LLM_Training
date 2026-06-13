# Testowanie Obciążenia API LLM — Dlaczego k6 i Locust Kłamią

> Tradycyjne testery obciążenia nie były zaprojektowane dla odpowiedzi strumieniowych, zmiennych długości wyjścia, metryk na poziomie tokenów ani nasycenia GPU. Dwie pułapki gryzą większość zespołów. Pułapka GIL: pomiar na poziomie tokenów w Locust uruchamia tokenizację pod GIL Python, który konkuruje z generowaniem żądań przy dużej współbieżności; backlog tokenizacji zawyża raportowaną latencję między-tokenową — Twój klient jest wąskim gardłem, nie serwer. Pułapka jednolitości promptów: identyczne prompty w pętli testują jeden punkt na dystrybucji tokenów; rzeczywisty ruch ma zmienne długości i różnorodne dopasowania prefiksów. LLMPerf naprawia to z `--mean-input-tokens` + `--stddev-input-tokens`. Mapowanie narzędzi w 2026: specjalizowane LLM (GenAI-Perf, LLMPerf, LLM-Locust, guidellm) dla dokładności na poziomie tokenów; **k6 v2026.1.0** + **k6 Operator 1.0 GA (wrzesień 2025)** — świadome strumieniowania, rozproszone natywnie Kubernetes przez CRD TestRun/PrivateLoadZone, najlepsze dla bramek CI/CD; Vegeta dla nasycenia stałą szybkością w Go; Locust 2.43.3 tylko z rozszerzeniem LLM-Locust dla strumieniowania. Wzorce obciążenia: stan ustalony, narastanie, skok (test autoskalowania), soak (wycieki pamięci).

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 minutes

## Learning Objectives

- Wyjaśnić dwa antywzorce (pułapka GIL, pułapka jednolitości promptów), które powodują, że generyczne testery obciążenia kłamią dla API LLM.
- Wybrać narzędzie dla danego celu: LLMPerf (uruchomienie benchmarkowe), k6 + rozszerzenie strumieniowania (bramka CI), guidellm (syntetyczne na dużą skalę), GenAI-Perf (referencja NVIDIA).
- Zaprojektować cztery wzorce obciążenia (stan ustalony, narastanie, skok, soak) i wymienić tryb awarii, który każdy wychwytuje.
- Zbudować realistyczną dystrybucję promptów używając średniej + odchylenia standardowego tokenów wejściowych zamiast ustalonej długości.

## The Problem

Przetestowałeś k6 swój endpoint LLM przy 500 współbieżnych użytkownikach. Wytrzymał. Wdrożyłeś. W produkcji przy 200 rzeczywistych użytkowników serwis upadł — P99 TTFT eksplodował, GPU przypięte.

Stały się dwie rzeczy. Po pierwsze, k6 wysłał 500 identycznych promptów — Twoje łączenie żądań i cachowanie prefiksów sprawiło, że wyglądało to, jakbyś obsługiwał 500 współbieżnych dekodów, podczas gdy w rzeczywistości obsługiwałeś jeden. Po drugie, k6 nie śledzi latencji między-tokenowej na odpowiedziach strumieniowych tak, jak doświadcza jej oko; widzi jedno połączenie HTTP, nie 500 tokenów przychodzących w różnych odstępach.

Testowanie obciążenia dla LLM to własna dyscyplina.

## The Concept

### Pułapka GIL (Locust)

Locust używa Python i uruchamia tokenizację po stronie klienta pod GIL. Przy dużej współbieżności tokenizator kolejkuje się za generowaniem żądań. Raportowana latencja między-tokenowa zawiera backlog tokenizacji po stronie klienta. Myślisz, że serwer jest wolny; to uprząż testowa.

Naprawa: rozszerzenie LLM-Locust przenosi tokenizację do osobnych procesów lub użyj uprzęży w języku kompilowanym (k6, LLMPerf używający tokenizers.rs).

### Pułapka jednolitości promptów

Wszystkie znane testery obciążenia pozwalają skonfigurować jeden prompt. W pętli testowej 10 000 iteracji ten sam dokładnie prompt jest wysyłany za każdym razem. Serwer widzi ten sam prefiks za każdym razem — trafienia cache prefiksu zbliżają się do 100%, przepustowość wygląda świetnie.

Naprawa: próbkuj z dystrybucji promptów. LLMPerf używa `--mean-input-tokens 500 --stddev-input-tokens 150` — różne długości, różna treść.

### Cztery wzorce obciążenia

1. **Stan ustalony** — stałe RPS przez 30-60 min. Wychwytuje: regresje bazowej wydajności.
2. **Narastanie** — liniowy wzrost RPS od 0 do celu przez 15 min. Wychwytuje: punkt załamania pojemności, anomalie rozgrzewania.
3. **Skok** — nagłe 3-10x RPS na 2 min, potem powrót. Wychwytuje: latencję autoskalowania, nasycenie kolejki, wpływ zimnego startu.
4. **Soak** — stan ustalony przez 4-8 godzin. Wychwytuje: wycieki pamięci, dryf puli połączeń, przepełnienie obserwowalności.

### Mapowanie narzędzi 2026

**LLMPerf** (Anyscale) — Python, ale tokenizacja wspierana przez Rust. Prompty ze średnią/odchyleniem standardowym. Świadomy strumieniowania. Najlepszy domyślny wybór do uruchomień wydajnościowych.

**NVIDIA GenAI-Perf** — referencja NVIDIA. Używa klienta Triton; kompleksowe pokrycie metryk. Uwaga: jego ITL wyklucza TTFT; LLMPerf go zawiera. Dwa narzędzia produkują różne TPOT dla tego samego serwera.

**LLM-Locust** (TrueFoundry) — rozszerzenie Locust naprawiające pułapkę GIL. Znajome DSL Locust + metryki strumieniowania.

**guidellm** — syntetyczne benchmarkowanie na dużą skalę.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (wrzesień 2025)**:
- k6 samo (Go, kompilowane, bez GIL) dodało metryki świadome strumieniowania.
- k6 Operator używa CRD TestRun / PrivateLoadZone dla rozproszonego testowania natywnego Kubernetes.
- Najlepsze dla bramek CI/CD i testowania SLA.

**Vegeta** — Go, prostsze niż k6. Nasycenie HTTP stałą szybkością. Nie jest świadome LLM, ale dobre do testowania bramek / ograniczeń szybkości.

**Locust 2.43.3 stock** — ma pułapkę GIL dla LLM. Tylko z rozszerzeniem LLM-Locust.

### Bramka SLA w CI

Uruchom k6 na PR z:

- 30-50 iteracji każda przy bazowym RPS.
- Bramka: P50/P95 TTFT, 5xx < 5%, TPOT poniżej progu.
- Złam budowę przy naruszeniu.

### Realistyczna dystrybucja promptów

Zbuduj z rzeczywistych próbek ruchu (jeśli je masz) lub z opublikowanych dystrybucji (np. ShareGPT prompty do czatu, HumanEval do kodu). Podaj średnią + odchylenie standardowe do LLMPerf. Unikaj pętli-z-jednym-promptem za wszelką cenę.

### Liczby, które warto zapamiętać

- k6 Operator 1.0 GA: wrzesień 2025.
- k6 v2026.1.0: metryki świadome strumieniowania.
- Typowe uruchomienie LLMPerf: 100-1000 żądań przy współbieżności X.
- Typowa bramka CI: 30-50 iteracji na PR.
- Cztery wzorce: stan ustalony, narastanie, skok, soak.

## Use It

`code/main.py` symuluje test obciążenia z realistyczną dystrybucją promptów, mierzy efektywny TPOT i demonstruje pułapkę jednolitego promptu.

## Ship It

Ta lekcja produkuje `outputs/skill-load-test-plan.md`. Na podstawie obciążenia i SLA wybiera narzędzie i projektuje cztery wzorce obciążenia.

## Exercises

1. Uruchom `code/main.py`. Porównaj jednolitą vs realistyczną dystrybucję — gdzie jest luka?
2. Napisz skrypt k6 dla bramki CI: TTFT P95 < 800 ms przy 100 współbieżnych, czas wykonania 5 minut.
3. Twój test soak pokazuje wzrost pamięci 50 MB/godzinę. Wymień trzy przyczyny i instrumentację do wyboru między nimi.
4. Test skoku z 10 RPS do 100 RPS. Jaki jest oczekiwany czas odzyskiwania, jeśli Karpenter + stos produkcyjny vLLM są na miejscu (Phase 17 · 03 + 18)?
5. GenAI-Perf raportuje TPOT=6ms; LLMPerf raportuje TPOT=11ms na tym samym serwerze. Wyjaśnij.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| LLMPerf | "the LLM harness" | Anyscale benchmark tool, streaming-aware |
| GenAI-Perf | "NVIDIA tool" | NVIDIA reference harness |
| LLM-Locust | "Locust for LLMs" | Locust extension fixing GIL trap |
| guidellm | "synthetic benchmark" | Large-scale synthetic tool |
| k6 Operator | "K8s k6" | CRD-based distributed k6 |
| GIL trap | "Python client overhead" | Tokenization backlog inflates reported latency |
| Prompt-uniformity trap | "single-prompt lie" | Loop with same prompt hits cache, inflates throughput |
| Steady-state | "constant load" | Flat RPS for N minutes |
| Ramp | "linear up" | 0 to target over duration |
| Spike | "burst test" | Sudden multiplier then revert |
| Soak | "long test" | Hours for leak detection |

## Further Reading

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)