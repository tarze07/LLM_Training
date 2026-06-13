# Capstone 07 — Potok Dostrajania Od Końca do Końca (Dane do SFT do DPO do Serwowania)

> Model 8B wytrenowany na Twoich danych, DPO-dostosowany do Twoich preferencji, skwantyzowany, zdekodowany spekulatywnie i serwowany przy mierzalnych $/1M tokenów. Otwarty stos 2026 to Axolotl v0.8, TRL 0.15, Unsloth do iteracji, GPTQ/AWQ/GGUF do kwantyzacji, vLLM 0.7 z EAGLE-3 do serwowania. Capstone polega na uruchomieniu całego potoku w sposób powtarzalny — YAML na wejściu, serwowany endpoint na wyjściu — i opublikowaniu karty modelu zgodnie z Modelem Otwartości Framework 2026.

**Type:** Capstone
**Languages:** Python (pipeline), YAML (configs), Bash (scripts)
**Prerequisites:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 hours

## Problem

Każdy poważny zespół AI w 2026 roku utrzymuje gotowy potok dostrajania. Nie dlatego, że dostarczają graniczny model bazowy, ale dlatego, że adaptacja downstream — domenowa SFT, DPO na oznaczonych preferencjach, dystylowane szkice do dekodowania spekulatywnego, serwowanie z EAGLE-3 — to miejsce, gdzie żyją mierzalne zyski. Axolotl v0.8 obsługuje konfiguracje SFT na wielu GPU. TRL 0.15 obsługuje DPO i GRPO. Unsloth daje szybką iterację na pojedynczym GPU. vLLM 0.7 z EAGLE-3 zwiększa przepustowość dekodowania 2-3x bez utraty jakości. Narzędzia działają; rzemiosło tkwi w YAMLach, higienie danych i dyscyplinie ewaluacyjnej.

Uruchomisz model bazowy 8B (Llama 3.3, Qwen3 lub Gemma 3) przez SFT, a następnie DPO na danych specyficznych dla zadania, skwantyzujesz do serwowania i zmierzysz zyski względem lm-evaluation-harness, RewardBench-2, MT-Bench-v2 i MMLU-Pro. Wyprodukujesz kartę modelu zgodnie z Modelem Otwartości Framework 2026. Celem jest powtarzalność — jedno polecenie uruchamia cały potok od końca do końca.

## Koncepcja

Potok ma pięć etapów. **Dane**: deduplikacja (MinHash / Datatrove), filtr jakości (klasyfikator w stylu Nemotron-CC), czyszczenie PII, kontrola higieny podziału przeciwko skażeniu publicznymi benchmarkami. **SFT**: YAML Axolotl, ZeRO-3 na 8xH100, harmonogram cosinus, spakowane sekwencje, 2-3 epoki. **DPO lub GRPO**: konfiguracja TRL, 1 epoka, pary preferencji albo oznakowane przez człowieka, albo ocenione przez model, dostrajanie beta. **Kwantyzacja**: GPTQ + AWQ + GGUF dla elastyczności wdrożenia. **Serwowanie**: vLLM 0.7 z głowami spekulatywnymi EAGLE-3 (lub SGLang z SpecForge), wdrożenie K8s, HPA na queue-wait.

Ablacje są rezultatem: SFT-only vs SFT+DPO vs SFT+GRPO na trzech benchmarkach specyficznych dla zadania. Metryki serwowania: tokeny/s przy batch 1 / 8 / 32, współczynnik akceptacji EAGLE-3, $/1M tokenów. Ewaluacja bezpieczeństwa: wskaźnik przepustowości Llama Guard 4. Karta modelu: ewaluacje odchylenia, seedy powtarzalności, licencjonowanie danych.

## Architektura

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## Stack

- Dane: Datatrove do deduplikacji, klasyfikator Nemotron-CC do jakości, Presidio do PII
- Model bazowy: Llama 3.3 8B, Qwen3 14B lub Gemma 3 12B
- SFT: Axolotl v0.8 z ZeRO-3, Flash Attention 3, spakowane sekwencje
- Dostrajanie preferencji: TRL 0.15 dla DPO lub GRPO; Unsloth do iteracji na pojedynczym GPU
- Kwantyzacja: GPTQ (Marlin), AWQ, GGUF przez llama.cpp
- Serwowanie: vLLM 0.7 z EAGLE-3 speculative decoding (lub SGLang 0.4 + SpecForge)
- Ewaluacja: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- Ewaluacja bezpieczeństwa: Llama Guard 4, ShieldGemma-2
- Infrastruktura: Kubernetes + NVIDIA device plugin, HPA na metryce queue-wait
- Obserwowalność: W&B do trenowania, Langfuse do inferencji

## Build It

1. **Potok danych.** Uruchom deduplikację Datatrove na surowym korpusie. Zastosuj klasyfikator jakości w stylu Nemotron-CC. Presidio czyści PII. Zapisz podziały trening/walidacja z jawnym seedem.

2. **Kontrola skażenia.** Dla każdego podziału walidacyjnego, oblicz MinHash względem zestawów testowych MMLU-Pro, MT-Bench-v2, RewardBench-2. Odrzuć każde nakładanie się.

3. **Axolotl SFT.** YAML z ZeRO-3, FA3, pakowaniem sekwencji. 2-3 epoki na 8xH100. Loguj do W&B.

4. **TRL DPO / GRPO.** Weź checkpoint SFT, uruchom jedną epokę DPO na parach preferencji (lub GRPO z weryfikowalną nagrodą na matematyce/kodzie). Przeskanuj beta.

5. **Kwantyzacja.** Wyprodukuj trzy kwanty: GPTQ-INT4-Marlin, AWQ-INT4, GGUF-Q4_K_M dla llama.cpp. Zapisz rozmiar i nominalną przepustowość.

6. **Serwowanie z dekodowaniem spekulatywnym.** Konfiguracja vLLM 0.7 z głowami szkicu EAGLE-3 wytrenowanymi przez Red Hat Speculators. Zmierz współczynnik akceptacji i opóźnienie ogona przy batch 1 / 8 / 32. Raportuj $/1M tokenów vs Anthropic / OpenAI na tej samej ewaluacji.

7. **Macierz ewaluacyjna.** Uruchom lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro na modelu bazowym, SFT-only, SFT+DPO, SFT+GRPO. Wyprodukuj tabelę.

8. **Ewaluacja bezpieczeństwa.** Wskaźnik przepustowości Llama Guard 4 na zestawie deweloperskim. Filtr wyjścia ShieldGemma-2.

9. **Karta modelu.** Szablon MOF 2026: dane, trenowanie, ewaluacja, bezpieczeństwo, licencja, sekcja powtarzalności z YAMLami i SHA commitów.

## Use It

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Ship It

`outputs/skill-finetuning-pipeline.md` opisuje rezultat. Jedno polecenie uruchamia dane przez SFT przez DPO przez kwantyfikację przez serwowanie przez ewaluację i emituje kartę modelu oraz serwowany endpoint.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Różnica ewaluacyjna vs baseline | Zmierzony zysk na zadaniach docelowych (MMLU-Pro, MT-Bench-v2, zadanie-specyficzne) |
| 20 | Powtarzalność potoku | Jedno polecenie uruchamia od końca do końca z identycznymi seedami |
| 20 | Higiena danych | Wskaźnik deduplikacji, pokrycie czyszczenia PII, zielona kontrola skażenia |
| 20 | Wydajność serwowania | tokeny/s przy bs=1/8/32, współczynnik akceptacji EAGLE-3, $/1M tokenów |
| 15 | Karta modelu + ewaluacja bezpieczeństwa | Kompletność MOF 2026 + wskaźnik przepustowości Llama Guard 4 |
| **100** | | |

## Ćwiczenia

1. Uruchom SFT-only vs SFT+DPO vs SFT+GRPO na tym samym benchmarku specyficznym dla zadania. Raportuj, która metoda preferencji wygrywa i o ile.

2. Zamień Llama 3.3 8B na Qwen3 14B. Zmierz $/1M tokenów przy dopasowanej jakości.

3. Zmierz współczynnik akceptacji EAGLE-3 na danych domenowych vs generyczny ShareGPT. Raportuj różnicę i co oznacza dla budżetów opóźnień.

4. Wstrzyknij 1% skażenia (wycieknij odpowiedzi MMLU-Pro do danych treningowych) i uruchom ponownie ewaluację. Obserwuj, jak dokładność MMLU-Pro skacze nierealistycznie. Zbuduj bramkę CI do kontroli skażenia, która to łapie.

5. Dodaj LoRA SFT jako alternatywę dla pełnego dostrajania. Zmierz lukę jakościową przy 10x niższej pamięci.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Axolotl | "Trener SFT" | Zunifikowany trener sterowany YAML dla SFT, DPO i dystylacji |
| TRL | "Dostrajacz preferencji" | Biblioteka Hugging Face dla DPO, GRPO, PPO na LLM |
| GRPO | "Optymalizacja polityki względem grupy" | Przepis RL DeepSeek R1 z weryfikowalnymi nagrodami |
| EAGLE-3 | "Szczyt dekodowania spekulatywnego" | Głowy szkicu przewidujące N tokenów do przodu; vLLM weryfikuje z modelem docelowym |
| MOF | "Model Otwartości Framework" | Standard 2026 do oceniania wydań modeli pod względem danych, kodu, licencji |
| Contamination check | "Higiena podziału" | Wykrywanie oparte na MinHash wycieku zestawu testowego do treningu |
| Acceptance rate | "Metryka EAGLE / MTP" | Frakcja szkicowanych tokenów, które model docelowy akceptuje |

## Dalsza lektura

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — the reference SFT / DPO trainer
- [TRL documentation](https://huggingface.co/docs/trl) — DPO and GRPO reference implementations
- [Unsloth](https://github.com/unslothai/unsloth) — single-GPU iteration reference
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO methodology
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — reference serving stack
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — alternate speculative-decoding trainer
- [Model Openness Framework 2026](https://isocpp.org/) — the open-release grading standard
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — canonical eval runner