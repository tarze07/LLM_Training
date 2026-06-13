# Inferencja na krawędzi — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> Głównym ograniczeniem na krawędzi jest przepustowość pamięci, a nie moc obliczeniowa. Mobilna DRAM ma 50-90 GB/s; HBM3 w centrum danych osiąga 2-3 TB/s — 30-50x różnica. Dekodowanie jest ograniczone pamięcią, więc różnica jest decydująca. W 2026 krajobraz dzieli się na cztery kierunki. Apple M4/A18 Neural Engine osiąga szczyt 38 TOPS z pamięcią jednolitą (bez kopii CPU↔NPU). Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon osiąga 45 TOPS. WebGPU + WebLLM uruchamia Llama 3.1 8B (Q4) przy ~41 tok/s na M3 Max (zgrubnie 70-80% natywnego); 17,6k gwiazdek GitHub, API kompatybilne z OpenAI, ~70-75% pokrycia mobilnego. NVIDIA Jetson Orin Nano Super (8GB) mieści Llama 3.2 3B / Phi-3; AGX Orin uruchamia gpt-oss-20b przez vLLM przy ~40 tok/s; Jetson T4000 (JetPack 7.1) to 2x AGX Orin. TensorRT Edge-LLM obsługuje EAGLE-3, NVFP4, chunked prefill — pokazane na CES 2026 przez Bosch, ThunderSoft, MediaTek.

**Type:** Learn
**Languages:** Python (stdlib, toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60 minutes

## Learning Objectives

- Wyjaśnij, dlaczego mobilna inferencja LLM jest ograniczona przepustowością pamięci, a moc obliczeniowa jest drugorzędna.
- Wymień cztery cele krawędziowe (Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson) i dopasuj każdy do przypadku użycia.
- Nazwij lukę w pokryciu WebGPU 2026 (Firefox Android nadrabia zaległości) i lądowanie Safari iOS 26.
- Wybierz format kwantyzacji na cel (Core ML INT4 + FP16 dla ANE, QNN INT8/INT4 dla Hexagon, WebGPU Q4 dla przeglądarki, NVFP4 dla Jetson Thor).

## The Problem

Klient chce czatbota na urządzeniu: głosowy, prywatny domyślnie, działający offline. Na MacBook Pro M3 Max, Llama 3.1 8B Q4 działa przy ~55 tok/s — w porządku. Na iPhone 16 Pro, ten sam model działa przy 3 tok/s — nie w porządku. Na średnim Androidzie z Snapdragon 8 Gen 3, 7 tok/s. W przeglądarce przez WebGPU na Chrome Android v121+, 4-8 tok/s w zależności od urządzenia.

Wariancja przepustowości nie jest problemem portowania. To różnica przepustowości razy format kwantyzacji razy to, czy NPU jest dostępne z przestrzeni użytkownika. Inferencja na krawędzi w 2026 to cztery różne problemy z czterema różnymi rozwiązaniami.

## The Concept

### Przepustowość jest prawdziwym sufitem

Dekodowanie odczytuje pełny zestaw wag dla każdego tokena. Jeden model 7B w Q4 to 3,5 GB. Odczyt 3,5 GB przy 50 GB/s zajmuje 70 ms — teoretyczny sufit ~14 tok/s. Przy 90 GB/s (wysokiej klasy mobilna DRAM) sufit przesuwa się do ~25 tok/s. Żadna ilość mocy obliczeniowej nie pomoże poniżej tej liczby.

HBM3 w centrum danych przy 3 TB/s odczytuje te same 3,5 GB w 1,2 ms — sufit to 830 tok/s. Ten sam model, te same wagi. Inny podsystem pamięci.

### Apple Neural Engine (M4 / A18)

- Do 38 TOPS. Pamięć jednolita (CPU i ANE współdzielą tę samą pulę) — brak narzutu kopiowania.
- Dostęp przez Core ML + skompilowane modele `.mlmodel`, lub przez Metal Performance Shaders (MPS) przez PyTorch.
- Backend Metal llama.cpp używa MPS, a nie ANE bezpośrednio; natywny ANE wymaga konwersji Core ML.
- Najlepsza praktyczna ścieżka dla aplikacji iOS w 2026: Core ML z wagami INT4 + aktywacjami FP16.

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- Do 45 TOPS. Zintegrowany z CPU i GPU w SoC, ale oddzielna domena pamięci.
- SDK QNN (Qualcomm Neural Network) i AI Hub zapewniają konwersję z PyTorch/ONNX.
- Szablony czatu, Llama 3.2, Phi-3 wszystkie są dostarczane jako pierwszorzędne artefakty na AI Hub.

### Intel / AMD NPU (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS. Oprogramowanie pozostaje w tyle za Apple/Qualcomm; OpenVINO się poprawia, ale jest niszowe.
- Najlepsze dla aplikacji copilot Windows ARM; natywne na desktopach AMD/Intel dla lokalnego-first.

### WebGPU + WebLLM

- Uruchamiaj modele w przeglądarce przez shadery obliczeniowe WebGPU; bez instalacji.
- Llama 3.1 8B Q4 przy ~41 tok/s na M3 Max — zgrubnie 70-80% natywnego przez ten sam backend.
- 17,6k gwiazdek GitHub na WebLLM; API JS kompatybilne z OpenAI; Apache 2.0.
- Pokrycie 2026: Chrome Android v121+, Safari iOS 26 GA, Firefox Android wciąż nadrabia zaległości. Ogólnie ~70-75% pokrycia mobilnego.

### Rodzina NVIDIA Jetson

- Orin Nano Super (8GB): mieści Llama 3.2 3B, Phi-3 przy dobrych tok/s.
- AGX Orin: uruchamia gpt-oss-20b przez vLLM przy ~40 tok/s.
- Thor / T4000 (JetPack 7.1): 2x wydajność AGX Orin, obsługuje EAGLE-3 i NVFP4.
- TensorRT Edge-LLM (2026) obsługuje spekulacyjne dekodowanie EAGLE-3, wagi NVFP4, chunked prefill — optymalizacje z centrum danych przeniesione na krawędź.

### Wybór kwantyzacji na cel

| Cel | Format | Uwagi |
|--------|--------|-------|
| Apple ANE | INT4 wagi + FP16 aktywacje | Ścieżka konwersji Core ML |
| Qualcomm Hexagon | QNN INT8 / INT4 | Konwertery AI Hub |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | Użyj `mlc_llm convert_weight` + skompilowany `.wasm`; GGUF nie jest obsługiwane |
| Jetson Orin Nano | Q4 GGUF lub TRT-LLM INT4 | Ograniczone pamięcią |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Ścieżka Edge-LLM |

### Pułapka długiego kontekstu na krawędzi

128K kontekst Llama 3.1 to funkcja centrum danych. Na telefonie z 8 GB RAM, 4 GB model + 2 GB pamięć podręczna KV dla 32K tokenów + narzut systemu operacyjnego = OOM. Wdrożenia na krawędzi utrzymują kontekst na 4K-8K, chyba że zaakceptowana jest agresywna kwantyzacja KV (Q4 KV).

### Głos jest zabójczą aplikacją

Agenci głosowi są wrażliwi na latencję (pierwszy token < 500 ms). Lokalna inferencja całkowicie eliminuje opóźnienie sieciowe. Połącz z zamianą mowy na tekst (warianty Whisper Turbo działają na krawędzi), a inferencja na krawędzi staje się produkcyjną pętlą głosową.

### Liczby, które powinieneś zapamiętać

- Apple M4 / A18 ANE: 38 TOPS.
- Qualcomm Hexagon SD X Elite: 45 TOPS.
- WebLLM M3 Max: ~41 tok/s na Llama 3.1 8B Q4.
- AGX Orin: ~40 tok/s na gpt-oss-20b przez vLLM.
- Różnica przepustowości centrum danych-krawędź: 30-50x.
- Pokrycie mobilne WebGPU: ~70-75% (Firefox Android w tyle).

## Use It

`code/main.py` oblicza teoretyczne sufity przepustowości dekodowania z matematyki ograniczonej przepustowością dla celów krawędziowych. Porównuje z obserwowanymi benchmarkami i podkreśla, gdzie przepustowość, a nie moc obliczeniowa, jest wąskim gardłem.

## Ship It

Ta lekcja produkuje `outputs/skill-edge-target-picker.md`. Biorąc platformę (iOS/Android/przeglądarka/Jetson), model i budżet latencji/pamięci, wybiera format kwantyzacji i potok konwersji.

## Exercises

1. Uruchom `code/main.py`. Dla modelu 7B w Q4 na Snapdragon 8 Gen 3 (~77 GB/s przepustowości), oblicz sufit dekodowania. Porównaj z obserwowanym 6-8 tok/s — czy środowisko wykonawcze jest wydajne?
2. WebGPU na Androidzie wymaga Chrome v121+. Zaprojektuj rozwiązanie zastępcze dla starszych przeglądarek — po stronie serwera przez to samo API kompatybilne z OpenAI.
3. Twoja aplikacja iOS potrzebuje strumieniowania 4K kontekstu. Która kombinacja modelu/formatu pozwala pozostać poniżej 4 GB aktywnej pamięci na iPhone 16?
4. Jetson AGX Orin uruchamia gpt-oss-20b przy 40 tok/s. Jetson Nano mieści tylko 3B. Jeśli Twój produkt jest przeznaczony na obie platformy, jak ujednolicisz stos inferencyjny?
5. Przedstaw argument, czy „WebLLM jest gotowy do produkcji w 2026." Powołaj się na pokrycie, wydajność i lukę Firefox Android.

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| ANE | „silnik neuronowy Apple" | NPU na urządzeniu w seriach M i A; pamięć jednolita |
| Hexagon | „NPU Qualcomm" | NPU Snapdragon; SDK QNN do dostępu |
| WebGPU | „GPU przeglądarki" | API GPU przeglądarki standaryzowane przez W3C; Chrome/Safari 2026 |
| WebLLM | „środowisko wykonawcze LLM w przeglądarce" | Projekt MLC-LLM; Apache 2.0; JS kompatybilny z OpenAI |
| Jetson | „NVIDIA na krawędzi" | Rodzina Orin Nano / AGX / Thor / T4000 |
| TRT Edge-LLM | „TensorRT na krawędzi" | Port TensorRT-LLM na krawędź 2026; EAGLE-3 + NVFP4 |
| Unified memory | „współdzielona pula" | CPU i NPU widzą tę samą pamięć RAM; brak narzutu kopiowania |
| Bandwidth-bound | „ograniczone pamięcią" | Dekodowanie ograniczone bajtami/s odczytu wag |
| Core ML | „konwersja Apple" | Framework Apple dla modeli natywnych ANE |
| QNN | „stos Qualcomm" | SDK sieci neuronowych Qualcomm |

## Further Reading

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — krajobraz i benchmarki.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — ogłoszenie portu na krawędź 2026.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — projekt i benchmarki.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — konwersja natywna dla ANE.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — wstępnie przekonwertowane modele dla Hexagon.

(End of file - total 128 lines)