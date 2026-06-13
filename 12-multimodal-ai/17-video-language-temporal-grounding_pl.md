# Modele Wideo-Językowe: Tokeny Temporalne i Ugruntowanie

> Wideo to nie stos zdjęć. 5-sekundowy klip ma porządek przyczynowy, czasowniki akcji i timing zdarzeń, których model obrazu nie jest w stanie reprezentować. Video-LLaMA (Zhang i in., czerwiec 2023) dostarczył pierwszy otwarty wideo-LLM z ugruntowaniem audiowizualnym. VideoChat i Video-LLaVA skalowały ten wzorzec. Do 2025 roku TMRoPE Qwen2.5-VL zamknął lukę w stosunku do granicznych modeli własnościowych. Każdy system rozwiązywał tokeny temporalne inaczej — Q-former na klip, concat-pool na klatkę, TMRoPE na token. Ta lekcja opisuje wzorce, buduje próbnik klatek uniform-vs-dynamic i ocenia na zadaniach ugruntowania temporalnego.

**Type:** Build
**Languages:** Python (stdlib, próbnik klatek + ewaluator ugruntowania temporalnego)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## Learning Objectives

- Wyjaśnić, dlaczego temporalne kodowanie pozycyjne zmienia wydajność wideo VLM niezależnie od enkodera wizyjnego.
- Porównać uniform, dynamic-FPS i event-driven próbkowanie klatek na tokenach-na-sekundę vs dokładności ugruntowania.
- Opisać konstrukcje Q-former-na-klip (Video-LLaMA) vs pooled-na-klatkę (Video-LLaVA) vs M-RoPE-na-token (Qwen2.5-VL).
- Wymienić cztery benchmarki wideo: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## Problem

1-minutowe wideo przy 30 FPS to 1800 klatek. Przy 196 tokenach wizyjnych na klatkę (ViT-B przy 224) daje to 352k tokenów — więcej niż jakikolwiek kontekst LLM z 2024 roku.

Istnieją trzy strategie redukcji:

1. Subpróbkowanie klatek (1-8 FPS w zależności od treści).
2. Agresywne poolowanie tokenów łat każdej klatki (pooling bilinearny 3x3 lub 4x4).
3. Kompresja przez Q-former, który przyjmuje 16-klatkowy klip i zwraca 64 tokeny.

Każdy kompromis jest inny. Subpróbkowanie traci szczegóły temporalne. Poolowanie traci szczegóły przestrzenne. Q-former traci trochę jedno i drugie, ale oszczędza tokeny.

Temporalne kodowanie pozycyjne to druga oś: w jaki sposób model wie, że klatka 5 była przed klatką 6? Opcje obejmują prosty 1D temporalny RoPE (Video-LLaMA), wyuczone osadzenia temporalne (Video-LLaVA) i TMRoPE (Qwen2.5-VL, w pełni 3D).

## Koncepcja

### Video-LLaMA: Q-former na klip + gałąź audio

Video-LLaMA (2023) był pierwszym otwartym wideo-LLM. Architektura:

- 16-klatkowe klipy przy 2 FPS (czyli 8 sekund).
- Cechy ViT na klatkę -> Wideo Q-former, który cross-attenduje po wszystkich 16 klatkach -> 32 wyuczone zapytania -> LLM.
- Równoległa gałąź audio: przebieg -> enkoder audio ImageBind -> Audio Q-former -> 32 zapytania -> LLM.

Mocna strona: wspólne rozumowanie audiowizualne. Słabość: stała długość klipu, brak dowolnego ugruntowania czasowego.

### VideoChat i Video-LLaVA

VideoChat zachował ideę Video-LLaMA, ale usunął audio i uprościł. Video-LLaVA (Lin i in., 2023) trenował pojedynczy enkoder wizyjny zarówno na obrazach, jak i klatkach wideo ("alignment before projection"), dając ujednoliconą reprezentację. Oba to zamrożony enkoder CLIP + MLP + LLM.

Żaden nie radzi sobie z długim wideo. Oba to systemy 8-16 klatek.

### Qwen2.5-VL i TMRoPE

Qwen2.5-VL wprowadził TMRoPE — Temporal-Modality Rotary Position Embedding. Każdy token łatki niesie pozycję (t, h, w), gdzie t to rzeczywisty znacznik czasu (nie indeks klatki).

Kluczowe różnice w stosunku do prostego osadzenia temporalnego:

- Bezwzględny czas, nie indeks. Model widzi "przy 4.2 sekundzie", nie "przy klatce 15."
- Rotacja na token, nie na klip. Każdy token wizyjny obraca się niezależnie według swojego znacznika czasu.
- Zgodny z dynamicznym FPS. Jeśli próbkujesz przy 2 FPS tu i 4 FPS tam, TMRoPE obsługuje nierówne odstępy naturalnie.

TMRoPE umożliwia zapytania "przy której sekundzie skacze kot?". Model może odpowiedzieć "przy 4.2 sekundzie." Video-LLaMA mógł tylko powiedzieć "wcześnie w klipie."

### Strategie próbkowania klatek

Uniform: próbkuj N klatek równomiernie w czasie. Proste, traci szczyty ruchu.

Dynamic FPS: próbkuj adaptacyjnie na podstawie intensywności ruchu. Przepływ optyczny lub różnicowanie klatek wybiera segmenty o wysokim ruchu do gęstszego próbkowania. Qwen2.5-VL trenuje na tym.

Event-driven: uruchom lekki detektor, próbkuj więcej tam, gdzie dzieje się akcja. Używany przez VideoAgent.

Klatka kluczowa + kontekst: próbkuj na granicach ujęć + kilka sąsiednich klatek. Używane dla treści filmowych.

### Poolowanie na klatkę

Przy 1 FPS i 576 tokenach na klatkę, 5-minutowy klip to 172 800 tokenów. Wykonalne przy 128k kontekście Qwen2.5-VL-72B, ale kosztowne.

Pooling bilinearny 3x3 redukuje do 64 tokenów na klatkę -> 19 200 tokenów na 5 minut. Optymalny punkt dla większości zadań.

Pooluj bardziej agresywnie (6x6 -> 16 tokenów na klatkę) dla przepływów pracy agenta, gdzie szczegóły przestrzenne mają mniejsze znaczenie.

### Cztery benchmarki wideo

- VideoMME: kompleksowe rozumienie wideo, krótkie + średnie + długie.
- TempCompass: szczegółowe rozumowanie temporalne, pytania "przed" / "po".
- EgoSchema: długohoryzontowe wideo z perspektywy pierwszej osoby.
- Video-MMMU: multimodalne wielodyscyplinarne pytania wideo.

Pełna ewaluacja wideo-VLM obejmuje wszystkie cztery. Testują różne osie — TempCompass dotyczy kolejności, EgoSchema rozumowania 3+ minut, VideoMME obejmuje różne czasy trwania.

### Formaty wyjściowe ugruntowania

Formaty wyjściowe dla ugruntowania temporalnego:

- Wolny tekst: "Kot skacze około 4 sekundy." Łatwy do parsowania, ale nieprecyzyjny.
- Strukturalny JSON: `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL trenuje to.
- Oparte na tokenach: specjalne tokeny `<time>4.1</time>` przeplatane z odpowiedzią. Wewnętrzny format Qwen2.5-VL.

Format oparty na tokenach jest najdokładniejszy dla zastosowań downstream. Format wyjściowy JSON Qwen2.5-VL parsuje się bezpośrednio.

### Najlepsza praktyka 2026

Dla wideo VLM w 2026:

- Enkoder: SigLIP 2 z M-RoPE lub TMRoPE (Qwen2.5-VL).
- Próbkowanie klatek: dynamiczny FPS (1-4 w zależności od ruchu) z limitem maksymalnej liczby klatek.
- Poolowanie na klatkę: bilinear 3x3.
- Wyjście: strukturalny JSON z polami czasu i zdarzenia.
- Benchmarki: VideoMME + TempCompass dla ogólnych; EgoSchema dla długohoryzontowych.

## Use It

`code/main.py` zawiera:

- Próbnik klatek uniform i dynamic-FPS.
- Zabawkowy ewaluator ugruntowania temporalnego: dla "ground truth" zdarzenia w czasie T i wyjścia modelu, oceń dokładność z tolerancją.
- Porównanie między Video-LLaMA (16 klatek, Q-former), Video-LLaVA (8 klatek, MLP), Qwen2.5-VL (dynamiczny FPS + TMRoPE).

## Ship It

Ta lekcja produkuje `outputs/skill-video-vlm-frame-planner.md`. Na podstawie zadania wideo (monitorowanie, rozpoznawanie akcji, ugruntowanie temporalne, podsumowywanie) wybiera próbnik klatek, współczynnik poolowania, format wyjściowy i oczekiwany poziom dokładności.

## Exercises

1. Dla 3-minutowego pokazu gotowania, wybierz uniform vs dynamiczny FPS. Uzasadnij liczbą tokenów.

2. Co konkretnie dodaje TMRoPE, czego prosta tabela osadzeń temporalnych nie może zrobić?

3. Napisz schemat JSON dla ugruntowania temporalnego, którego VLM może się nauczyć emitować. Uwzględnij przypadki błędów.

4. Przeczytaj sekcję 3 Video-LLaVA na temat "Alignment Before Projection." Dlaczego jest to lepsze niż trenowanie oddzielnych enkoderów obrazu i wideo?

5. Biorąc pod uwagę tablicę liderów VideoMME, jaka jest luka między najlepszym otwartym modelem a najlepszym modelem własnościowym według stanu na 2026? Jaka część tej luki wynika z kodowania temporalnego, a jaka ze skali bazowego LLM?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Temporal grounding | "Odpowiedzi zlokalizowane w czasie" | VLM wypisuje konkretny zakres znaczników czasu dla momentu wystąpienia zdarzenia |
| TMRoPE | "Czasowo-Multimodalny RoPE" | 3D rotary position z bezwzględnymi znacznikami czasu, używany przez Qwen2.5-VL |
| Dynamic FPS | "Próbkowanie świadome ruchu" | Próbkuj więcej klatek w segmentach o wysokim ruchu, mniej w statycznych |
| Frame pooling | "Kompresja przestrzenna na klatkę" | Redukuj łatki na klatkę przez interpolację bilinearną przed LLM |
| Video Q-former | "Kompresor klipów" | Wąskie gardło cross-atencyjne mapujące N klatek na K wyuczonych zapytań |
| VideoMME | "Benchmark wideo" | Kompleksowy benchmark krótkich/średnich/długich wideo, 2500+ próbek |

## Further Reading

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)