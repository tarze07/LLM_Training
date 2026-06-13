# Wybór Samodzielnego Serwowania — llama.cpp, Ollama, TGI, vLLM, SGLang

> Cztery silniki dominują samodzielną inferencję w 2026. Wybieraj na podstawie sprzętu, skali i ekosystemu. **llama.cpp** jest najszybsze na CPU — najszersze wsparcie modeli, pełna kontrola nad kwantyzacją i wątkowaniem. **Ollama** to instalacja jednym poleceniem na laptopie deweloperskim, ~15-30% wolniejsze od llama.cpp (Go + CGo + serializacja HTTP), 3x różnica przepustowości pod obciążeniem produkcyjnym. **TGI weszło w tryb utrzymania 11 grudnia 2025** — tylko poprawki błędów, ~10% wolniejsza surowa przepustowość niż vLLM, ale historycznie najlepsza obserwowalność i integracja z ekosystemem HF. Ten status utrzymania czyni go ryzykownym długoterminowym zakładem — SGLang lub vLLM są bezpieczniejszymi domyślnymi wyborami dla nowych projektów. **vLLM** to domyślny wybór ogólnego przeznaczenia do produkcji — v0.15.1 (luty 2026) dodaje PyTorch 2.10, RTX Blackwell SM120, optymalizację H200. **SGLang** to specjalista od agentowych wieloobrotowych / obciążeń z dużym prefiksem — 400 000+ GPU w produkcji (xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). Ograniczenia sprzętowe: tylko CPU → tylko llama.cpp. AMD / non-NVIDIA → tylko vLLM (TRT-LLM jest zablokowany dla NVIDIA). Wzorzec potoku 2026: dev = Ollama, staging = llama.cpp, prod = vLLM lub SGLang. Te same wagi GGUF/HF w całym potoku.

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Time:** ~45 minutes

## Learning Objectives

- Wybrać silnik na podstawie sprzętu (CPU / AMD / NVIDIA Hopper / Blackwell), skali (1 użytkownik / 100 / 10 000) i obciążenia (ogólny czat / agent / długi kontekst).
- Wymienić status trybu utrzymania TGI 2026 (11 grudnia 2025) i dlaczego skłania nowe projekty ku vLLM lub SGLang.
- Opisać potok dev/staging/prod używający tych samych wag GGUF lub HF w całym potoku.
- Wyjaśnić, dlaczego "tylko CPU" wymusza llama.cpp, a "AMD" wyklucza TRT-LLM.

## The Problem

Twój zespół rozpoczyna nowy projekt samodzielnego hostowania LLM. Jeden inżynier mówi Ollama, drugi mówi vLLM, trzeci mówi "czy TGI nie działa po prostu od razu?" Wszyscy trzej mają rację w różnych kontekstach. Żaden nie jest odpowiedni dla wszystkich.

W 2026 drzewo wyboru ma znaczenie: najpierw sprzęt, potem skala, potem obciążenie. I jedno konkretne wydarzenie 2025 — TGI wchodzące w tryb utrzymania 11 grudnia — zmienia domyślny wybór dla nowych projektów.

## The Concept

### Pięć silników

| Silnik | Najlepsze dla | Uwagi |
|--------|----------|-------|
| **llama.cpp** | CPU / brzeg / minimalne zależności / najszersze wsparcie modeli | Najszybsze na CPU, pełna kontrola |
| **Ollama** | Laptopy deweloperskie, pojedynczy użytkownik, instalacja jednym poleceniem | 15-30% wolniejsze od llama.cpp; 3x różnica przepustowości produkcyjnej |
| **TGI** | Ekosystem HF, regulowane branże | **Tryb utrzymania 11 grudnia 2025** |
| **vLLM** | Produkcja ogólnego przeznaczenia, 100+ użytkowników | Szeroki domyślny wybór produkcyjny; v0.15.1 luty 2026 |
| **SGLang** | Agentowe wieloobrotowe, obciążenia z dużym prefiksem | 400 000+ GPU w produkcji |

### Decyzja najpierw sprzęt

**Tylko CPU** → llama.cpp. Ollama też działa, ale jest wolniejsze. Żaden inny silnik nie jest konkurencyjny na CPU.

**AMD GPU** → vLLM (wsparcie AMD ROCm). SGLang też działa. TRT-LLM jest zablokowany dla NVIDIA, więc odpada.

**NVIDIA Hopper (H100 / H200)** → vLLM lub SGLang lub TRT-LLM. Wszystkie trzy najwyższej klasy.

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM liderem przepustowości (Phase 17 · 07). vLLM i SGLang podążają blisko.

**Apple Silicon (M-series)** → llama.cpp (Metal). Ollama to opakowuje.

### Decyzja potem skala

**1 użytkownik / lokalny dev** → Ollama. Jedno polecenie, pierwszy token w sekundach.

**10-100 użytkowników / mały zespół** → vLLM single-GPU.

**100-10k użytkowników / produkcja** → stos produkcyjny vLLM (Phase 17 · 18) lub SGLang.

**10k+ użytkowników / enterprise** → stos produkcyjny vLLM + rozdzielone (Phase 17 · 17) + LMCache (Phase 17 · 18).

### Decyzja potem obciążenie

**Ogólny czat / Q&A** → vLLM wygrywa na szerokim domyślnym.

**Agentowe wieloobrotowe (narzędzia, planowanie, pamięć)** → RadixAttention SGLang (Phase 17 · 06) dominuje.

**RAG z dużym ponownym użyciem prefiksu** → SGLang.

**Generowanie kodu** → vLLM w porządku; SGLang nieco lepszy na cache.

**Długi kontekst (128K+)** → vLLM + chunked prefill; SGLang + tiered KV.

### Pułapka utrzymania TGI

Hugging Face TGI weszło w tryb utrzymania 11 grudnia 2025 — tylko poprawki błędów idąc naprzód. Historycznie: obserwowalność najwyższej klasy, najlepsza w swojej klasie integracja z ekosystemem HF (karty modeli, narzędzia bezpieczeństwa), nieco za vLLM w surowej przepustowości.

Dla nowych projektów w 2026: domyślnie z dala od TGI. Istniejące wdrożenia TGI mogą kontynuować, ale powinny ostatecznie migrować. SGLang i vLLM są bezpieczniejszymi domyślnymi wyborami.

### Wzorzec potoku

Dev (Ollama) → staging (llama.cpp) → prod (vLLM). Te same wagi GGUF lub HF w całym potoku. Inżynierowie iterują szybko na laptopach; staging odzwierciedla kwantyzację produkcyjną; prod jest celem serwowania.

### Zastrzeżenie Ollama

Ollama jest świetne do dev. Nie jest świetne do współdzielonej produkcji: serializacja HTTP Go dodaje narzut, zarządzanie współbieżnością jest prostsze niż vLLM, wsparcie OpenTelemetry pozostaje w tyle. Używaj Ollama tam, gdzie błyszczy — jeden użytkownik, jedno polecenie — i przełącz się na vLLM dla współdzielonego.

### Samodzielne hostowanie vs zarządzane to osobna decyzja

Phase 17 · 01 (zarządzani hiperskalowcy), · 02 (platformy inferencyjne) pokrywają zarządzane. Ta lekcja zakłada, że już zdecydowałeś się na samodzielne hostowanie. Powody do samodzielnego hostowania: rezydencja danych, niestandardowe dostrojenie, całkowity koszt posiadania na dużą skalę, model domeny niedostępny u hostowanego.

### Liczby, które warto zapamiętać

- Tryb utrzymania TGI: 11 grudnia 2025.
- vLLM v0.15.1: luty 2026; PyTorch 2.10; wsparcie Blackwell SM120.
- Ślad produkcyjny SGLang: 400 000+ GPU.
- Różnica przepustowości Ollama vs llama.cpp: 15-30% wolniejsze; 3x pod obciążeniem produkcyjnym.

```figure
data-parallel
```

## Use It

`code/main.py` to przechodnik drzewa decyzyjnego: dla danego sprzętu + skali + obciążenia wybiera silnik i wyjaśnia dlaczego.

## Ship It

Ta lekcja produkuje `outputs/skill-engine-picker.md`. Na podstawie ograniczeń wybiera silnik i pisze plan migracji.

## Exercises

1. Uruchom `code/main.py` ze swoim sprzętem / skalą / obciążeniem. Czy wynik zgadza się z Twoją intuicją?
2. Twoja infrastruktura to 12 H100 i 8 MI300X AMD. Jaki silnik? Dlaczego TRT-LLM odpada?
3. Zespół chce użyć TGI w 2026, bo "to, co znamy". Uzasadnij przypadek migracji.
4. Ollama dev do vLLM prod: co zmienia się w kwantyzacji, konfiguracji i obserwowalności?
5. Produkt RAG z P99 długością prefiksu 8K i wysokim ponownym użyciem między tenantami. Wybierz silnik i połącz go z Phase 17 · 11 + 18.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| llama.cpp | "the CPU one" | Widest model support, fastest on CPU |
| Ollama | "the laptop one" | One-command install, dev-grade throughput |
| TGI | "HF's serving" | Maintenance mode since Dec 2025 |
| vLLM | "the default" | Broad production baseline 2026 |
| SGLang | "the agentic one" | Prefix-heavy, RadixAttention |
| TRT-LLM | "NVIDIA-locked" | Blackwell throughput leader, NVIDIA only |
| GGUF | "llama.cpp format" | Bundled K-quant variants |
| Production-stack | "vLLM K8s" | Phase 17 · 18 reference deployment |
| Pipeline pattern | "dev→stage→prod" | Ollama → llama.cpp → vLLM on same weights |

## Further Reading

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)