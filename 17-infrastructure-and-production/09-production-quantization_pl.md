# Kwantyzacja produkcyjna — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> Format kwantyzacji nie jest uniwersalnym wyborem — jest funkcją sprzętu, silnika serwującego i obciążenia. GGUF Q4_K_M lub Q5_K_M rządzi na CPU i edge, dostarczany przez llama.cpp i Ollama. GPTQ wygrywa w vLLM, gdy potrzebujesz multi-LoRA na tej samej bazie. AWQ z jądrami Marlin-AWQ dostarcza ~741 tok/s na modelu klasy 7B z najlepszym Pass@1 przy INT4 — domyślny wybór 2026 dla produkcji w centrum danych. FP8 pozostaje środkowym gruntem na Hopper, Ada i Blackwell — prawie bezstratny i szeroko wspierany. NVFP4 i MXFP4 (mikroskalowanie Blackwell) są agresywne i wymagają walidacji na blok. Dwie pułapki gryzą zespoły: zbiór kalibracyjny musi odpowiadać domenie wdrożenia, a pamięć podręczna KV jest oddzielona od kwantyzacji wag — lekcja AWQ „mój model ma teraz 4 GB" zapomina o 10-30 GB pamięci podręcznej KV przy produkcyjnych rozmiarach partii.

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## Learning Objectives

- Wymień sześć produkcyjnych formatów kwantyzacji i ich punkty słodkie w 2026.
- Wybierz format, mając sprzęt (CPU vs GPU, Hopper vs Blackwell), silnik (vLLM, TRT-LLM, llama.cpp) i obciążenie (rutynowy czat, rozumowanie, multi-LoRA).
- Oblicz zaoszczędzoną pamięć wag i nietkniętą pamięć podręczną KV dla wybranego formatu.
- Nazwij pułapkę zbioru kalibracyjnego, która degraduje skwantowane modele na ruchu domenowym.

## The Problem

Kwantyzacja zmniejsza pamięć i przepustowość HBM, czego dokładnie potrzebuje dekodowanie. Model FP16 70B to 140 GB wag. Skwantyzuj wagi do INT4 (AWQ lub GPTQ), a model ma 35 GB — mieści się w jednym H100 z miejscem na pamięć podręczną KV, co ma znaczenie, ponieważ przy 128 równoczesnych sekwencjach z kontekstem 2k, sama pamięć podręczna KV to 20-30 GB.

Ale kwantyzacja nie jest darmowa. Agresywna kwantyzacja degraduje jakość, szczególnie w zadaniach wymagających rozumowania. Różne formaty współpracują z różnymi silnikami. Różny sprzęt obsługuje natywnie różne precyzje. Zoo formatów 2026 jest realne i nie możesz skopiować cudzego wyboru — musisz wybrać na podstawie swojego stosu.

## The Concept

### Sześć formatów

| Format | Bity | Punkt słodki | Silniki |
|--------|------|-------------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, edge, laptopy | llama.cpp, Ollama |
| GPTQ | 4-8 | Multi-LoRA na vLLM | vLLM, TGI |
| AWQ | 4 | Produkcja GPU w centrum danych | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Centrum danych Hopper/Ada/Blackwell | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell wielu użytkowników | TRT-LLM |
| NVFP4 | 4 | Blackwell wielu użytkowników | TRT-LLM |

### GGUF — domyślny dla CPU/edge

GGUF to format pliku, a nie schemat kwantyzacji per se — łączy warianty K-quant (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) w jednym kontenerze. Q4_K_M i Q5_K_M to domyślne produkcyjne — jakość bliska BF16 przy 4-5 bitach. Najlepszy wybór dla serwowania na CPU lub edge, ponieważ llama.cpp jest zdecydowanie najszybszym silnikiem inferencji CPU.

Kara przepustowości w vLLM: ~93 tok/s na 7B — format nie jest zoptymalizowany dla jąder GPU. Używaj GGUF, gdy celem wdrożenia jest CPU/edge. Nie w innym przypadku.

### GPTQ — multi-LoRA w vLLM

GPTQ to algorytm kwantyzacji po treningu z przebiegiem kalibracyjnym. Jądra Marlin czynią go szybkim na GPU (2,6x przyspieszenie vs GPTQ bez Marlina). ~712 tok/s na 7B.

Unikalna zaleta: GPTQ-Int4 obsługuje adaptery LoRA w vLLM. Jeśli serwujesz model bazowy plus 10-50 dostrojonych wariantów (każdy jako LoRA), GPTQ jest twoją ścieżką. NVFP4 nie obsługuje jeszcze LoRA od początku 2026.

### AWQ — domyślny dla GPU w centrum danych

Aktywacyjna kwantyzacja wag (Activation-aware Weight Quantization). Chroni ~1% najbardziej znaczących wag podczas kwantyzacji. Jądra Marlin-AWQ: 10,9x przyspieszenie vs naiwny. ~741 tok/s na 7B, najlepszy Pass@1 wśród formatów INT4.

Wybierz AWQ dla nowego serwowania GPU, chyba że potrzebujesz multi-LoRA (GPTQ) lub agresywnego Blackwell FP4 (NVFP4).

### FP8 — niezawodny środek

8-bitowy zmiennoprzecinkowy. Prawie bezstratny. Szeroko wspierany. Tensor Cores Hopper przyspieszają FP8 natywnie. Blackwell dziedziczy. FP8 to bezpieczny domyślny 2026, gdy jakość nie podlega negocjacjom (rozumowanie, medycyna, generowanie kodu). Oszczędności pamięci są połową INT4, ale ryzyko jakościowe jest znacznie niższe.

### MXFP4 / NVFP4 — agresywny Blackwell

Mikroskalowanie FP4. Każdy blok wag ma własny współczynnik skali. Agresywny, ale sprzętowo przyspieszony na Blackwell Tensor Cores. Przecina liczbę bajtów na token na pół w porównaniu do FP8 — ekonomiczna przewaga z Fazy 17 · 07.

Zastrzeżenia:
- Brak wsparcia LoRA (wczesny 2026).
- Spadek jakości widoczny na obciążeniach wymagających rozumowania.
- Waliduj na swoim zbiorze ewaluacyjnym na model.

### Pułapka kalibracyjna

AWQ i GPTQ wymagają zbioru danych kalibracyjnych — typowo C4 lub WikiText. Dla modeli domenowych (kod, medyczny, prawny), kalibracja na ogólnym tekście internetowym pozwala algorytmowi podejmować złe decyzje, które wagi chronić. Pass@1 na HumanEval może spaść o kilka punktów.

Poprawka: kalibruj na danych z domeny. Kilkaset próbek domenowych zwykle wystarcza. Testuj na zbiorze ewaluacyjnym przed wdrożeniem.

### Pułapka pamięci podręcznej KV

AWQ zmniejsza wagi do 4 bitów. Pamięć podręczna KV jest oddzielna i pozostaje w FP16/FP8. Dla modelu 70B z AWQ:

- Wagi: ~35 GB (INT4 ze 140 GB).
- Pamięć podręczna KV przy 128 równoczesnych × 2k kontekstu: ~20 GB.
- Aktywacje: ~5 GB.
- Razem: ~60 GB — mieści się na H100 80GB.

Naiwne „skwantowałem mój model do 4 GB" zapomina o pozostałych 30-50 GB. Budżetuj HBM holistycznie.

Oddzielnie, kwantyzacja pamięci podręcznej KV (FP8 KV lub INT8 KV) to inny wybór z własnymi kompromisami — wpływa bezpośrednio na dokładność attention i nie jest darmową wygraną.

### AWQ INT4 jest ryzykowny dla rozumowania

Chain-of-thought, matematyka, generowanie kodu z długim kontekstem — te cierpią widocznie z powodu agresywnej kwantyzacji. AWQ INT4 traci ~3-5 punktów na MATH. Dla obciążeń wymagających rozumowania, dostarczaj FP8 lub BF16; zaakceptuj koszt pamięci.

### Przewodnik wyboru 2026

- Serwowanie CPU/edge: GGUF Q4_K_M. Gotowe.
- Serwowanie GPU, rutynowy czat, brak LoRA: AWQ.
- Serwowanie GPU, multi-LoRA: GPTQ z Marlin.
- Obciążenie rozumowania: FP8.
- Centrum danych Blackwell, zweryfikowana jakość: NVFP4 + FP8 KV.
- Niejednoznaczne: uruchom ewaluację 1000 próbek na każdym formacie kandydującym.

```figure
gpu-memory-breakdown
```

## Use It

`code/main.py` oblicza footprint pamięci (wagi + KV + aktywacje) i względną przepustowość dla sześciu formatów dla zakresu rozmiarów modeli. Pokazuje, gdzie dominuje pamięć podręczna KV, gdzie opłaca się kompresja wag i gdzie FP8 jest bezpiecznym wyborem.

## Ship It

Ta lekcja produkuje `outputs/skill-quantization-picker.md`. Biorąc sprzęt, rozmiar modelu, typ obciążenia i tolerancję jakościową, wybiera format i produkuje plan kalibracji/walidacji.

## Exercises

1. Uruchom `code/main.py`. Dla modelu 70B przy 128 równoczesnych z kontekstem 2k, oblicz całkowite HBM dla każdego formatu. Który format pozwala zmieścić się na jednym H100 80GB?
2. Masz model kodowania 7B. Wybierz format i uzasadnij. Jeśli myliłeś się co do tolerancji jakościowej, jaka jest ścieżka odzyskiwania?
3. Oblicz rozmiar zbioru danych kalibracyjnych potrzebny do kalibracji AWQ dla modelu domeny medycznej. Dlaczego więcej danych nie zawsze jest lepsze?
4. Przeczytaj artykuł o jądrach Marlin-AWQ lub notatki wydania. Wyjaśnij w trzech zdaniach, dlaczego AWQ osiąga 741 tok/s na 7B, podczas gdy surowy GPTQ osiąga ~712.
5. Kiedy ma sens łączenie wag AWQ z pamięcią podręczną KV FP8 vs utrzymywanie KV w BF16?

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| GGUF | „format llama.cpp" | Format pliku łączący warianty K-quant; domyślny dla CPU/edge |
| Q4_K_M | „Q4 K M" | 4-bitowy K-quant medium; produkcyjny domyślny GGUF |
| GPTQ | „gee pee tee q" | Po treningu INT4 z kalibracją; obsługuje LoRA w vLLM |
| AWQ | „a w q" | Aktywacyjny INT4; jądra Marlin; najlepszy Pass@1 przy INT4 |
| Marlin kernels | „szybkie jądra INT4" | Niestandardowe jądra CUDA dla INT4 na Hopper; 10x przyspieszenie |
| FP8 | „ośmiobitowy float" | Bezpieczna domyślna precyzja na Hopper/Ada/Blackwell |
| MXFP4 / NVFP4 | „mikroskalowanie cztery" | 4-bitowy FP Blackwell z czynnikami skali na blok |
| Calibration dataset | „dane kalibracyjne" | Tekst wejściowy używany do wyboru parametrów kwantyzacji; musi odpowiadać domenie |
| KV cache quantization | „KV INT8" | Oddzielny wybór od wag; wpływa na dokładność attention |

## Further Reading

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — benchmarki porównawcze.
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — liczby przepustowości według formatu.
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — wybór format po formacie.
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — obsługiwane formaty i flagi.
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — oryginalna formulacja AWQ.
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — oryginalna formulacja GPTQ.

(End of file - total 140 lines)