# Metryki inferencji — TTFT, TPOT, ITL, Goodput, P99

> Cztery metryki decydują o tym, czy wdrożenie inferencji działa. TTFT to prefill plus kolejka plus sieć. TPOT (równoważnie ITL) to koszt dekodowania ograniczony pamięcią na token. Opóźnienie od końca do końca to TTFT plus TPOT razy długość wyjścia. Przepustowość to tokeny na sekundę zagregowane w całej flocie. Ale ta, która ma znaczenie dla produktu, to goodput — frakcja żądań, które spełniły wszystkie SLO jednocześnie. Wysoka przepustowość przy niskim goodput oznacza, że przetwarzasz tokeny, które nigdy nie docierają do użytkowników na czas. Liczby referencyjne dla Llama-3.1-8B-Instruct na TRT-LLM w 2026: średnia TTFT 162 ms, średnia TPOT 7,33 ms, średnia E2E 1 093 ms. Zawsze raportuj P50, P90, P99 — nigdy tylko średnią. I obserwuj pułapkę pomiarową: GenAI-Perf wyklucza TTFT z obliczeń ITL, LLMPerf go włącza; dwa narzędzia nie zgadzają się co do TPOT dla tego samego uruchomienia.

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## Learning Objectives

- Zdefiniuj precyzyjnie TTFT, TPOT, ITL, E2E, przepustowość i goodput oraz nazwij komponent, który każdy mierzy.
- Wyjaśnij, dlaczego średnia jest złą statystyką dla serwowania LLM i jak czytać P50/P90/P99.
- Skonstruuj SLO z wieloma ograniczeniami (np. TTFT<500 ms ORAZ TPOT<15 ms ORAZ E2E<2 s) i oblicz goodput względem niego.
- Wymień dwa narzędzia benchmarkowe, które nie zgadzają się co do TPOT dla tego samego uruchomienia i wyjaśnij dlaczego.

## The Problem

„Nasza przepustowość to 15 000 tokenów na sekundę." I co z tego? Jeśli 40% żądań przekroczyło 2 sekundy od końca do końca, użytkownicy porzucili sesję. Sama przepustowość nie mówi, czy produkt działa.

Inferencja ma wiele osi latencji i każda zawodzi inaczej. Prefill jest ograniczony obliczeniowo i skaluje się z długością promptu. Dekodowanie jest ograniczone pamięcią i skaluje się z rozmiarem partii. Opóźnienie kolejkowania to problem operacyjny. Sieć to problem odległości fizycznej. Potrzebujesz odrębnych metryk dla każdej, potrzebujesz percentyli i potrzebujesz pojedynczej złożonej, która mówi „czy użytkownik dostał to, czego oczekiwał" — to jest goodput.

## The Concept

### TTFT — czas do pierwszego tokena

`TTFT = queue_time + network_request + prefill_time`

Prefill dominuje, gdy prompty są długie. Na Llama-3.3-70B FP8 na H100, 32k prompt zajmuje ~800 ms czystego prefillu. Czas kolejki to zachowanie schedulera pod obciążeniem. Żądanie sieciowe to czas na przewodzie, w tym TLS. TTFT to latencja, którą użytkownik widzi, zanim cokolwiek zacznie być przesyłane strumieniowo.

### TPOT / ITL — latencja między tokenami

Wiele nazw dla jednej wielkości. `TPOT` (time per output token), `ITL` (inter-token latency), `decode latency per token` — wszystkie to samo. To czas między kolejnymi przesyłanymi strumieniowo tokenami po pierwszym.

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

Na tym samym stosie Llama-3.3-70B H100 z chunked prefill, średnia TPOT ~7 ms. Bez chunked prefill, podczas długiego prefillu na sąsiedniej sekwencji, TPOT może pikować do 50 ms. Obserwuj P99, nie średnią.

### Latencja E2E

`E2E = TTFT + TPOT * output_tokens + network_response`

Dla długich wyjść (>500 tokenów), E2E jest zdominowane przez TPOT. Dla krótkich wyjść z długimi promptami, E2E jest zdominowane przez TTFT. Raportuj E2E warunkowane długością wyjścia.

### Przepustowość

`przepustowość = całkowita_liczba_tokenów_wyjściowych / czas_upływu`

Metryka zagregowana. Mówi o wydajności floty. Nie mówi o kondycji pojedynczego żądania.

### Goodput — metryka, na której faktycznie ci zależy

`goodput = frakcja żądań spełniających (TTFT <= a) ORAZ (TPOT <= b) ORAZ (E2E <= c)`

SLO to wiele ograniczeń. Żądanie jest „dobre" tylko wtedy, gdy każde ograniczenie zostało spełnione. Goodput to udział. Wysoka przepustowość przy 60% goodput to porażka. Niższa przepustowość przy 99% goodput to cel.

W 2026, goodput jest metryką używaną w zgłoszeniach MLPerf Inference v6.0 i wewnętrznym śledzeniu SLA u dostawców platform AI.

### Dlaczego średnia jest złą statystyką

Rozkłady latencji LLM są prawoskośne. Partia dekodowania z jednym długim sąsiadem prefillu może wysłać 500 tokenów z TPOT ~7 ms i 20 tokenów z TPOT ~60 ms. Średnia TPOT to 9 ms. P99 TPOT to 65 ms. Użytkownicy regularnie trafiają P99 — dlatego odchodzą.

Zawsze raportuj trójkę (P50, P90, P99). Dla doświadczenia użytkownika, P99 to ten, który optymalizujesz.

### Liczby referencyjne — Llama-3.1-8B-Instruct na TRT-LLM, 2026

- średnia TTFT: 162 ms
- średnia TPOT: 7,33 ms
- średnia E2E: 1 093 ms
- P99 TPOT: zmienia się 10-25 ms w zależności od konfiguracji chunked-prefill.

To opublikowane punkty referencyjne NVIDIA. Zmieniają się z rozmiarem modelu (70B pokazałoby 3-5x), sprzętem (H100 vs B200 ~3x) i obciążeniem.

### Pułapka pomiarowa

Dwa najczęściej używane narzędzia benchmarkowe 2026 nie zgadzają się co do TPOT dla tego samego uruchomienia:

- **NVIDIA GenAI-Perf**: wyklucza TTFT z obliczeń ITL. ITL zaczyna się od tokena 2.
- **LLMPerf**: włącza TTFT. ITL zaczyna się od tokena 1.

Dla żądania z TTFT 500 ms i 100 tokenami wyjściowymi w 700 ms całkowitego dekodowania, GenAI-Perf raportuje `ITL = 700/99 = 7,07 ms`, LLMPerf raportuje `ITL = 1200/100 = 12,00 ms`. Wybór narzędzia zmienia liczbę.

Zawsze podawaj, które narzędzie. Zawsze publikuj definicję.

### Konstruowanie SLO

Rozsądne SLO skierowane do konsumenta dla modelu czatu 70B w 2026:

- TTFT P99 <= 800 ms.
- TPOT P99 <= 25 ms.
- E2E P99 <= 3 s dla wyjść <300 tokenów.
- Cel goodput >= 99%.

Korporacyjne SLO zaostrzają TTFT (200-400 ms) i luzują E2E. Chodzi o to, aby je zapisać, zmierzyć wszystkie trzy i śledzić goodput jako pojedynczy złożony wskaźnik.

### Jak mierzyć

- Uruchom prawdziwy ruch lub realistyczny syntetyczny (LLMPerf z `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`).
- Celuj w 2x szczytową współbieżność dla uruchomienia benchmarku.
- Uruchom 30-50 iteracji, weź percentyle z połączonej próbki.
- Publikuj z nazwą narzędzia, wersją narzędzia, modelem, sprzętem, współbieżnością, dystrybucją promptów.

```figure
throughput-latency
```

## Use It

`code/main.py` to zabawkowy kalkulator goodput. Wygeneruj syntetyczny rozkład latencji, zastosuj SLO i oblicz goodput. Pokazuje również różnicę TPOT między GenAI-Perf a LLMPerf na tym samym śladzie.

## Ship It

Ta lekcja produkuje `outputs/skill-slo-goodput-gate.md`. Biorąc obciążenie i SLO, produkuje przepis benchmarkowy gotowy do CI/CD, który bramkuje wdrożenia na goodput, a nie przepustowość.

## Exercises

1. Uruchom `code/main.py`. Wygeneruj rozkład z 1% pikiem ogona. Jak zmienia się goodput, gdy zaostrzysz P99 TPOT z 30 ms do 15 ms?
2. Dostawca cytuje „15 000 tok/s na Llama 3.3 70B H100". Wymień trzy pytania, które należy zadać przed zaufaniem.
3. Dlaczego chunked prefill chroni P99 TPOT, ale nie średnią TPOT?
4. Skonstruuj SLO konsumenckie dla asystenta głosowego (pierwszy token jest słyszany, nie czytany). Która metryka jest najbardziej widoczna dla użytkownika?
5. Przeczytaj README LLMPerf i dokumentację GenAI-Perf. Zidentyfikuj trzy inne metryki, w których narzędzia się nie zgadzają.

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| TTFT | „czas do pierwszego tokena" | Kolejka + sieć + prefill; zdominowany przez prefill przy długich promptach |
| TPOT | „czas na token wyjściowy" | Koszt dekodowania ograniczony pamięcią na token po pierwszym |
| ITL | „latencja między tokenami" | To samo co TPOT w większości narzędzi (nie wszystkich — patrz GenAI-Perf) |
| E2E | „od końca do końca" | TTFT + TPOT * output_len; sieć po stronie odpowiedzi na wierzchu |
| Throughput | „tok/s" | Wydajność floty; bezużyteczna bez percentyli latencji |
| Goodput | „wskaźnik spełnienia SLO" | Frakcja żądań spełniających każde ograniczenie SLO jednocześnie |
| P99 | „ogon" | 1-na-100 najgorszy przypadek latencji; metryka doświadczenia użytkownika |
| SLO multi-constraint | „złącze" | ORAZ wszystkich trzech ograniczeń latencji; żądanie nie powiedzie się, jeśli którekolwiek jest naruszone |
| GenAI-Perf vs LLMPerf | „pułapka narzędzi" | Narzędzia nie zgadzają się, czy ITL zawiera TTFT |

## Further Reading

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — kanoniczna definicja TTFT, ITL, TPOT.
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — alternatywne definicje i przepis pomiarowy.
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — pomiar stosowany na rzeczywistych wdrożeniach.
- [LLMPerf](https://github.com/ray-project/llmperf) — otwartoźródłowy benchmark oparty na Ray.
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — narzędzie benchmarkowe NVIDIA.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — akceptowany w branży benchmark oparty na goodput.

(End of file - total 144 lines)