# Stos produkcyjny vLLM z LMCache KV Offloading

> Stos produkcyjny vLLM to referencyjne wdrożenie Kubernetes — router, silniki i obserwowalność połączone razem. LMCache to warstwa odciążania KV, która wyciąga KV cache z pamięci GPU i wykorzystuje go ponownie między zapytaniami i silnikami (CPU DRAM, następnie dysk/Ceph). Łącznik odciążania KV vLLM 0.11.0 (styczeń 2026) czyni to asynchronicznym i wtyczkowym poprzez Connector API (v0.9.0+). Latencja odciążania nie jest widoczna dla użytkownika. LMCache jest wartościowy nawet bez współdzielonych prefiksów — gdy GPU wyczerpuje sloty KV, wywłaszczone żądania mogą zostać przywrócone z CPU zamiast ponownie obliczać prefill. Opublikowane benchmarki na 16x H100 (80GB HBM) na 4 a3-highgpu-4g: gdy KV cache przekracza HBM, zarówno natywne odciążanie CPU, jak i LMCache znacząco poprawiają przepustowość; przy niskim śladzie KV wszystkie konfiguracje dorównują bazowej z małym narzutem.

**Type:** Learn
**Languages:** Python (stdlib, toy KV-spill simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 minutes

## Learning Objectives

- Narysować diagram warstw stosu produkcyjnego vLLM: router, silniki, odciążanie KV, obserwowalność.
- Wyjaśnić działanie Connector API do odciążania KV (v0.9.0+) i jak asynchroniczna ścieżka 0.11.0 ukrywa latencję odciążania.
- Określić ilościowo, kiedy LMCache CPU-DRAM pomaga (KV > HBM) vs dodaje narzut (KV wystarczająco małe, by zmieścić się w HBM).
- Wybrać między natywnym odciążaniem CPU vLLM a łącznikiem LMCache w zależności od ograniczeń wdrożenia.

## The Problem

Twoje serwowanie vLLM pokazuje GPU przy 100% HBM ze zdarzeniami wywłaszczenia przy każdym wzroście współbieżności. Żądania są eksmitowane, ponownie kolejkowane, a Ty ponownie wypełniasz ten sam 2K-tokenowy prompt cztery razy na minutę. Moc obliczeniowa GPU jest marnowana na redundantne prefill; wydajność użyteczna jest znacznie poniżej surowej przepustowości.

Dodawanie kolejnych GPU kosztuje liniowo. Dodanie większej ilości HBM nie jest możliwe. Ale CPU DRAM jest tanie — jeden socket ma 512 GB+ przy latencji o rzędy wielkości gorszej niż HBM, ale wystarczającej dla "tymczasowo ciepłego" KV cache.

LMCache wyodrębnia KV cache do CPU DRAM, więc wywłaszczone żądania odzyskują szybko, a powtarzające się prefiksy między silnikami współdzielą cache bez ponownego wypełniania każdego silnika.

## The Concept

### Stos produkcyjny vLLM

`github.com/vllm-project/production-stack` to referencyjne wdrożenie Kubernetes:

- **Router** — świadomy cache (Phase 17 · 11). Konsumuje zdarzenia KV.
- **Silniki** — workerzy vLLM. Jeden na GPU lub grupę TP/PP.
- **Odciążanie KV cache** — wdrożenie LMCache lub natywny łącznik.
- **Obserwowalność** — scrapowanie Prometheus, dashboardy Grafana, ślady OTel.
- **Płaszczyzna sterowania** — odnajdywanie usług, konfiguracja, aktualizacje kroczące.

Dostarczane jako Helm chart + operator.

### Connector API odciążania KV (v0.9.0+)

vLLM 0.9.0 wprowadził Connector API dla wtyczkowych backendów KV cache. Twój silnik odciąża bloki do łącznika; łącznik je przechowuje (RAM, dysk, przechowywanie obiektów, LMCache). Żądanie potrzebuje bloku, łącznik ładuje go z powrotem.

vLLM 0.11.0 (styczeń 2026) dodaje asynchroniczną ścieżkę odciążania — odciążanie może odbywać się w tle, więc silnik nie blokuje się na nim w typowym przypadku. Końcowa latencja i przepustowość zależą od kształtu obciążenia, współczynnika trafień KV cache i ciśnienia systemowego; notatki vLLM wyraźnie wskazują, że odciążanie z niestandardowym jądrem może pogorszyć przepustowość przy niskich współczynnikach trafień, a planowanie asynchroniczne ma znane problemy interakcji z dekodowaniem spekulatywnym.

### Natywne odciążanie CPU vs LMCache

**Natywne odciążanie CPU vLLM**: lokalne dla silnika. Przechowuje bloki KV w RAM hosta. Szybkie w implementacji, zero skoków sieciowych. Nie krzyżuje silników.

**Łącznik LMCache**: skala klastra. Przechowuje bloki we współdzielonym serwerze LMCache (CPU DRAM + warstwa Ceph/S3). Bloki są dostępne dla każdego silnika. Opublikowane benchmarki na 16x H100.

Wybierz natywne, gdy pojedynczy silnik ma presję HBM. Wybierz LMCache, gdy wiele silników współdzieli prefiksy (RAG ze wspólnymi system promptami, multi-tenant ze współdzielonymi szablonami).

### Zachowanie w benchmarkach

Test 16x H100 (80 GB HBM) rozłożonych na 4 a3-highgpu-4g:

- Niski ślad KV (krótkie prompty, niska współbieżność): wszystkie konfiguracje dorównują bazowej, LMCache dodaje ~3-5% narzutu.
- Umiarkowany ślad: LMCache zaczyna pomagać przy ponownym użyciu prefiksu między silnikami.
- KV przekracza HBM: natywne odciążanie CPU i LMCache oba znacząco poprawiają przepustowość; LMCache większy zysk dzięki współdzieleniu między silnikami.

### Kiedy LMCache jest decydujący

- Serwowanie multi-tenant, gdzie system prompty są współdzielone między tenantami.
- RAG, gdzie fragmenty dokumentów powtarzają się między zapytaniami.
- Warianty dostrojone (LoRA) na tej samej bazie, gdzie ponowne użycie KV modelu bazowego obcina redundantną pracę.
- Obciążenia z dużą liczbą wywłaszczeń: przywrócenie z CPU tańsze niż ponowny prefill.

### Kiedy NIE włączać

- Mała presja HBM — płacisz narzut bez korzyści.
- Krótkie konteksty (<1K tokenów) — czas transferu > ponowny prefill.
- Pojedynczy tenant, pojedynczy prompt — brak możliwości ponownego użycia.

### Integracja z serwowaniem rozdzielonym

Phase 17 · 17 serwowanie rozdzielone + LMCache kumulują się: transfery KV z puli prefill do puli decode lądują w LMCache, jeśli nie są używane; kolejne zapytania pobierają z LMCache. Router świadomy cache z Phase 17 · 11 może kierować do silnika, którego lokalny LUB współdzielony przez LMCache cache pasuje.

### Liczby, które warto zapamiętać

- vLLM 0.9.0: wydano Connector API.
- vLLM 0.11.0 (styczeń 2026): asynchroniczna ścieżka odciążania; wpływ na końcową latencję zależy od obciążenia, współczynnika trafień KV i ciśnienia systemowego (nie bezwzględna gwarancja).
- Benchmark 16x H100: LMCache pomaga, gdy ślad KV przekracza HBM.
- Mała presja HBM: 3-5% narzutu bez korzyści.

```figure
zero-sharding
```

## Use It

`code/main.py` symuluje obciążenie z dużą liczbą wywłaszczeń z LMCache i bez. Raportuje uniknięte ponowne prefill, przyrost przepustowości i próg rentowności wykorzystania HBM.

## Ship It

Ta lekcja produkuje `outputs/skill-vllm-stack-decider.md`. Na podstawie kształtu obciążenia i wdrożenia vLLM decyduje o natywnym vs LMCache vs żadnym.

## Exercises

1. Uruchom `code/main.py`. Przy jakim wykorzystaniu HBM LMCache zaczyna się opłacać?
2. Tenant współdzieli 6K-tokenowy system prompt na 200 zapytań/godzinę. Oblicz oczekiwane oszczędności LMCache na tenanta.
3. Serwer LMCache jest pojedynczym punktem awarii. Zaprojektuj strategię HA (repliki, fallback do natywnego).
4. LMCache przechowuje na Ceph na dysku talerzowym. Dla 4K-tokenowego KV przy 70B FP8 (500 MB), jaki jest czas odczytu vs ponowny prefill?
5. Uzasadnij, czy asynchroniczna ścieżka vLLM 0.11.0 jest "darmowa" — gdzie ukryty jest narzut?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Production-stack | "the reference deployment" | vLLM's Kubernetes Helm chart + operator |
| Connector API | "KV backend interface" | vLLM 0.9.0+ pluggable KV store interface |
| Native CPU offload | "engine-local spill" | Store KV in host RAM of same engine |
| LMCache | "cluster KV cache" | Cross-engine KV cache server on CPU DRAM + disk |
| 0.11.0 async | "non-blocking offload" | Offload hidden behind engine stream |
| Preemption | "evict to make room" | KV cache shuffle when HBM full |
| Prefix reuse | "same system prompt" | Multiple queries share beginning; cache hit |
| Ceph tier | "disk tier" | Durable storage below DRAM in the cache hierarchy |

## Further Reading

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — asynchronous path details.