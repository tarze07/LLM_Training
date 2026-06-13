# TensorRT-LLM na Blackwell z FP8 i NVFP4

> TensorRT-LLM jest tylko dla NVIDIA, ale wygrywa na Blackwell. Na GB200 NVL72 z orkiestracją Dynamo, SemiAnalysis InferenceX zmierzył $0,012 za milion tokenów na modelu 120B w Q1-Q2 2026, wobec $0,09/M na H100 + vLLM — 7-krotna różnica ekonomiczna. Stos składa się z trzech skumulowanych reżimów zmiennoprzecinkowych: FP8 pozostaje krytyczny dla pamięci podręcznej KV i jąder attention, ponieważ ma zakres dynamiczny, którego potrzebują; NVFP4 (mikroskalowanie 4-bitowe) obsługuje wagi i aktywacje; wielotokenowa predykcja (MTP) i rozdzielony prefill/decode dodają kolejne 2-3x na wierzchu. Obsługa modeli w dniu 0 ładuje wagi FP4 bezpośrednio bez konwersji po treningu. Haczyk dla zespołów inżynieryjnych 2026: TRT-LLM to zamknięty stos NVIDIA, więc jego przyjęcie wymienia przenośność na przepustowość. Przeprowadź matematykę na swojej mieszance modeli i sprzętu przed zobowiązaniem się.

**Type:** Learn
**Languages:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 13 (Quantization)
**Time:** ~75 minutes

## Learning Objectives

- Wyjaśnij, dlaczego FP8 pozostaje krytyczny dla pamięci podręcznej KV i attention, nawet gdy wagi są w NVFP4.
- Oblicz footprint HBM modelu z pogranicza w BF16, FP8 i NVFP4 oraz uzasadnij, skąd pochodzą oszczędności.
- Wymień funkcje specyficzne dla Blackwell, które wykorzystuje TRT-LLM (FP4 dnia 0, MTP, rozdzielone serwowanie, prymitywy all-to-all).
- Zdecyduj, kiedy blokada NVIDIA przez TRT-LLM jest warta 7-krotnej różnicy kosztów vs vLLM na Hopper.

## The Problem

Front ekonomiki inferencji w 2026 to „ile tokenów za dolara". Odpowiedź zależy od czterech skumulowanych wyborów: generacja sprzętu (Hopper H100/H200 vs Blackwell B200/GB200), precyzja (BF16 → FP8 → NVFP4), silnik serwujący (vLLM vs SGLang vs TRT-LLM) i orkiestracja (zwykła vs rozdzielona vs Dynamo).

Na Hopper z vLLM, 120B MoE działa za ~$0,09 za milion tokenów. Na Blackwell z TRT-LLM + Dynamo, ten sam model działa za ~$0,012 — 7x taniej. Część tej różnicy to sprzęt (Blackwell ma 11-15x przepustowości LLM na GPU vs Hopper). Część to stos: wagi FP4, szkic MTP, rozdzielony prefill/decode i NVLink 5 all-to-all dla komunikacji ekspertów MoE.

Nie możesz tego odtworzyć poza stosem NVIDIA. To jest kompromis — przenośność za ekonomikę. Zrozumienie, które wybory stosu dają jaką część różnicy, jest celem tej lekcji.

## The Concept

### Dlaczego FP8 jest wciąż podłogą dla pamięci podręcznej KV

Częsty błąd w 2026: zakładanie, że NVFP4 stosuje się wszędzie. Nie stosuje. Pamięć podręczna KV potrzebuje FP8 (8-bitowy zmiennoprzecinkowy), ponieważ przechowuje klucze i wartości attention, które obejmują szeroki zakres dynamiczny. Kwantyzacja KV do FP4 powoduje katastrofalną utratę dokładności — ogon dystrybucji odpada, a wyniki attention załamują się. Bity wykładnika FP8 dają pamięci podręcznej KV zakres, którego potrzebuje.

NVFP4 (2025-2026) stosuje się do wag i aktywacji. Mikroskalowanie: każdy blok wag ma własny współczynnik skali, więc małe bloki mogą obejmować różne zakresy dynamiczne bez utraty skali na tensor. Dla aktywacji, FP4 się sprawdza, ponieważ aktywacje mają mały zakres w obrębie warstwy.

Typowa konfiguracja Blackwell:

- Wagi: NVFP4 (mikroskalowanie 4-bitowe).
- Aktywacje: NVFP4.
- Pamięć podręczna KV: FP8.
- Akumulator attention: FP32 (stabilność softmax).

### Prymitywy specyficzne dla Blackwell używane przez TRT-LLM

- **Wagi FP4 dnia 0**: dostawcy modeli dostarczają wagi FP4 bezpośrednio; TRT-LLM ładuje bez konwersji po treningu. Żaden krok AWQ / GPTQ dla FP4.
- **Wielotokenowa predykcja (MTP)**: ten sam pomysł co EAGLE (Faza 17 · 05), ale zintegrowany z kompilacją TRT-LLM.
- **Rozdzielone serwowanie**: prefill i decode na oddzielnych pulach GPU, pamięć podręczna KV przesyłana przez NVLink lub InfiniBand. Ten sam pomysł co Dynamo (Faza 17 · 20).
- **Prymitywy komunikacji all-to-all**: NVLink 5 obciął latencję komunikacji ekspertów MoE o 3x vs Hopper. Jądra MoE TRT-LLM są dostrojone do tego.
- **Mikroskalowanie NVFP4 + MXFP8**: sprzętowo przyspieszone obsługiwanie współczynników skali na Blackwell Tensor Cores.

### Liczby, które powinieneś zapamiętać

- HGX B200 przy $0,02/M tokenów na GPT-OSS-120B przez TRT-LLM.
- GB200 NVL72 przy $0,012/M tokenów przez Dynamo (orkiestrujący TRT-LLM).
- H100 + vLLM ≈ $0,09/M tokenów na porównywalnym obciążeniu.
- 2,8x wzrost przepustowości w ciągu trzech miesięcy aktualizacji TRT-LLM (2026).
- 11-15x przepustowości LLM na GPU, Blackwell vs Hopper.
- MLPerf Inference v6.0 (kwiecień 2026): Blackwell dominuje każde zgłoszone zadanie.

### Co FP4 faktycznie kosztuje w jakości

NVFP4 jest agresywny. Na obciążeniach wymagających rozumowania (chain-of-thought, matematyka, generowanie kodu z długim kontekstem), wagi FP4 widocznie degradują. Kalibracja na blok łagodzi, ale nie eliminuje. Zespoły dostarczające modele rozumowania często używają wag FP8 + aktywacji FP4 jako kompromisu, lub trzymają się H200 z FP8 wszędzie.

Zasada: zawsze waliduj jakość zadania na swoim zbiorze ewaluacyjnym przed zobowiązaniem się do wag NVFP4.

### Dlaczego to decyzja o blokadzie NVIDIA

TRT-LLM to C++ + CUDA + zamknięte jądra źródłowe. Modele muszą być skompilowane dla konkretnego SKU GPU. Brak AMD, brak Intel, brak ARM. Jeśli Twoja strategia infrastrukturalna jest wielodostawcza, TRT-LLM nie wchodzi w grę dla warstwy serwowanej przez TRT-LLM — nadal możesz serwować z vLLM na mieszanym sprzęcie. Jeśli jesteś tylko NVIDIA, 7-krotna różnica płaci za blokadę.

### Praktyczny przepis 2026

Dla rocznego rachunku za inferencję 100 mln USD+, działanie na Hopper + vLLM zostawia 7-10x na stole. Migruj obciążenia dominujące kosztowo do Blackwell + TRT-LLM + Dynamo. Utrzymaj poziom eksperymentalny na H100 + vLLM dla szybkości iteracji modelu. Waliduj jakość na każdym przekonwertowanym modelu NVFP4 przed produkcją.

### Bonus rozdzielenia

Rozdzielone serwowanie TRT-LLM (oddzielne pule prefill i decode) jest szczegółowo omówione w Fazie 17 · 20. Na Blackwell, mnożnik się kumuluje: wagi FP4 × przyspieszenie MTP × rozmieszczenie rozdzielone × routing świadomy pamięci podręcznej. Liczba 7x zakłada ten pełny stos.

```figure
pipeline-parallel
```

## Use It

`code/main.py` oblicza footprint HBM, przepustowość dekodowania (reżim ograniczony pamięcią) i $/M-tokenów dla modelu w trzech stosach: H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM. Uruchom, aby zobaczyć efekt kumulacyjny i udział każdej zmiany w różnicy.

## Ship It

Ta lekcja produkuje `outputs/skill-trtllm-blackwell-advisor.md`. Biorąc obciążenie, rozmiar modelu i roczny wolumen tokenów, decyduje, czy stos Blackwell + TRT-LLM jest wart blokady NVIDIA.

## Exercises

1. Uruchom `code/main.py`. Na 120B MoE z 30% aktywnych parametrów, oblicz przepustowość dekodowania ograniczoną przepustowością pamięci na H100 BF16, H100 FP8 i B200 NVFP4/FP8. Skąd pochodzi największy skok?
2. Klient wydaje 2 mln USD/rok na H100 + vLLM. Jaka jest liczba GPU Blackwell, którą muszą kupić, aby zamortyzować migrację do TRT-LLM w 12 miesięcy, przy 7-krotnej różnicy ekonomicznej?
3. Widzisz spadek dokładności o 3 punkty na MATH po konwersji wag NVFP4. Wymień dwie ścieżki odzyskiwania: jedna z priorytetem jakości (utrzymaj wagi FP8), jedna z priorytetem kosztu (kalibruj z danymi z domeny).
4. Przeczytaj wyniki inferencji MLPerf v6.0. Które zadanie ma najmniejszą lukę Blackwell-over-Hopper i dlaczego?
5. Oblicz HBM potrzebne dla modelu 405B przy wagach NVFP4 + pamięci podręcznej KV FP8 przy kontekście 128k. Czy mieści się na pojedynczym węźle GB200 NVL72?

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| FP8 | „ośmiobitowy float" | 8-bitowy zmiennoprzecinkowy; używany dla pamięci podręcznej KV i attention ze względu na zakres dynamiczny |
| NVFP4 | „czterobitowy mikro" | 4-bitowy format FP z mikroskalowaniem NVIDIA; wagi i aktywacje na Blackwell |
| MXFP8 | „MX osiem" | Wariant FP8 z mikroskalowaniem; sprzętowo przyspieszony na Blackwell Tensor Cores |
| Day-0 FP4 | „dostarcz wagi FP4" | Dostawcy modeli wydają wagi już w FP4; brak kroku konwersji po treningu |
| MTP | „wielotokenowa predykcja" | Zintegrowany szkic spekulacyjnego dekodowania TRT-LLM (Faza 17 · 05) |
| Disaggregated serving | „rozdziel prefill/decode" | Prefill i decode na oddzielnych pulach GPU; KV przesyłane przez NVLink/IB |
| All-to-all | „komunikacja ekspertów MoE" | Wzorzec komunikacji kierujący tokeny do GPU ekspertów; NVLink 5 tnie 3x |
| InferenceX | „benchmark inferencji SemiAnalysis" | Akceptowany w branży benchmark kosztu na token 2026 |

## Further Reading

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — wyniki MLPerf z kwietnia 2026.
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all i jądra MoE.
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — oficjalna dokumentacja silnika.
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — rozdzielona orkiestracja powyżej TRT-LLM.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — pakiet benchmarków publikujący liczby Blackwell.

(End of file - total 114 lines)