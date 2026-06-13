# Generowanie Wideo

> Obraz to tensor 2-W. Wideo to tensor 3-W. Teoria jest taka sama; obliczenia są 10-100× trudniejsze. Sora od OpenAI (luty 2024) udowodniła, że to możliwe. Do 2026 roku Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0 i WAN 2.2 dostarczają produkcyjne wideo z tekstu w 1080p — a stack z otwartymi wagami (CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2) jest 12 miesięcy za nimi.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## Problem

10-sekundowe wideo 1080p przy 24 fps to 240 klatek 1920×1080×3 pikseli. To ~1.5 GB surowych danych na klip. Dyfuzja w przestrzeni pikseli jest niewykonalna. Potrzebujesz:

1. **Kompresji czasoprzestrzennej.** VAE, który koduje wideo, a nie klatki, w sekwencję łat czasoprzestrzennych.
2. **Spójności czasowej.** Klatki muszą dzielić zawartość, oświetlenie i tożsamość obiektów przez sekundy. Sieć musi modelować ruch.
3. **Budżetu obliczeniowego.** Trening wideo jest 10-100× droższy niż obrazu dla tego samego rozmiaru modelu.
4. **Warunkowania.** Tekst, obraz (pierwsza klatka), audio lub inne wideo. Większość modeli produkcyjnych akceptuje wszystkie cztery.

Architekturą, która to rozwiązała, jest **Diffusion Transformer (DiT)** zastosowany do łat czasoprzestrzennych, trenowany na ogromnych zbiorach danych (prompt, podpis, wideo). Ta sama strata dyfuzyjna co w Lekcji 06.

## Koncepcja

![Dyfuzja wideo: patchyfikacja, DiT, dekodowanie](../assets/video-generation.svg)

### Patchyfikacja

Zakoduj wideo za pomocą 3D VAE (uczona kompresja czasoprzestrzenna). Latent ma kształt `[T_latent, H_latent, W_latent, C_latent]`. Podziel na łaty o rozmiarze `[t_p, h_p, w_p]`. Dla modeli w stylu Sory, `t_p = 1` (łaty na klatkę) lub `t_p = 2` (co dwie klatki). 10-sekundowe wideo 1080p kompresuje się do ~20,000-100,000 łat.

### Spatiotemporalny DiT

Transformer przetwarza płaską sekwencję łat. Każda łata ma 3-W osadzenie pozycyjne (czas + y + x). Attention jest zwykle faktoryzowany:

- **Attention przestrzenny** wewnątrz łat każdej klatki.
- **Attention czasowy** w poprzek klatek w tej samej lokalizacji przestrzennej.
- **Pełny attention 3-W** jest 16-100× droższy; używany tylko przy niskiej rozdzielczości lub w badaniach.

### Warunkowanie tekstem

Cross-attention z dużym enkoderem tekstu (T5-XXL dla Sory, CogVideoX-5B używa T5-XXL). Długie prompty mają znaczenie — zbiór treningowy Sory miał gęste przepisy wygenerowane przez GPT, średnio 200 tokenów na klip.

### Trening

Standardowa strata dyfuzyjna (predykcja ε lub v) na latentach czasoprzestrzennych. Dane: wideo z sieci + ~100M wyselekcjonowanych klipów + syntetyczne podpisy tekstowe. Obliczenia: 10,000+ godzin GPU nawet dla małego badania naukowego; skala Sory to 100,000+.

## Krajobraz produkcyjny 2026

| Model | Data | Maks. czas | Maks. rozdz. | Otwarte wagi? | Uwagi |
|-------|------|------------|--------------|---------------|-------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | Nie | Pierwszy model pokazujący właściwości symulatora świata w skali |
| Sora Turbo | 2024-12 | 20s | 1080p | Nie | Produkcyjna Sora przy 5× szybszym wnioskowaniu |
| Veo 2 (Google) | 2024-12 | 8s | 4K | Nie | Najwyższa jakość + fizyka w 2025 |
| Veo 3 | 2025 Q3 | 15s | 4K | Nie | Natywne audio i silniejsza kontrola kamery |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | Nie | Najlepszy ruch ludzki w 2025 Q1 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | Nie | Profesjonalne narzędzia wideo na wierzchu |
| Pika 2.0 | 2024-10 | 5s | 1080p | Nie | Najsilniejsza spójność postaci |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Tak (2B, 5B) | Pierwszy otwarty 5B-skala wideo |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Tak (13B) | Otwarty SOTA koniec 2024 |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Tak (10B) | Najbardziej permisywnie licencjonowany |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Tak | Najsilniejszy otwarty model połowa 2025 |

Otwarte wagi zamykają lukę szybciej niż w przestrzeni obrazu: HunyuanVideo + LoRA WAN 2.2 już zasilają większość open-source'owych przepływów pracy w połowie 2026.

## Zbuduj To

`code/main.py` symuluje główną ideę spatiotemporalnego DiT: podziel na łaty małe syntetyczne wideo, dodaj osadzenie pozycyjne na łatę i odszum całą sekwencję za pomocą attention w stylu transformerowym na łatach. Bez numpy; czysty Python. Pokazujemy, że spójność czasowa wyłania się nawet w 1-W, gdy sąsiednie klatki dzielą odszumiacz i osadzenia pozycyjne.

### Krok 1: podziel na łaty syntetyczne 1-W "wideo"

```python
def make_video(T_frames=8, rng=None):
    # "wideo" to sekwencja wartości 1-W podążająca za gładką trajektorią
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Krok 2: osadzenie pozycyjne na klatkę

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Krok 3: odszumiacz widzi całą sekwencję

Zamiast odszumiać każdą klatkę niezależnie, nasza malutka sieć konkatenuje wszystkie wartości klatek + ich osadzenia pozycyjne i przewiduje szum dla wszystkich klatek wspólnie.

### Krok 4: test spójności czasowej

Po treningu próbkuj wideo. Zmierz różnicę między klatkami. Jeśli model nauczył się struktury czasowej, różnice pozostaną mniejsze niż przy próbkowaniu każdej klatki niezależnie.

## Pułapki

- **Niezależne próbkowanie na klatkę = migotanie.** Jeśli uruchomisz dyfuzję obrazu na każdej klatce osobno, wynik migocze, ponieważ szum każdej klatki jest niezależny. Dyfuzja wideo naprawia to, łącząc klatki przez attention lub wspólny szum.
- **Naiwny attention 3-W = OOM.** Pełny attention 3-W na 10-sekundowym latent 1080p to setki miliardów operacji. Sfaktoryzuj na przestrzenny + czasowy.
- **Opisywanie danych ma większe znaczenie niż rozmiar.** Głównym ulepszeniem Sory w stosunku do wcześniejszych prac był trening na ~10× bardziej szczegółowych podpisach (klipy ponownie oznaczone przez GPT-4). Raport techniczny OpenAI jest wyraźny w tej kwestii.
- **Warunkowanie pierwszą klatką.** Większość modeli produkcyjnych akceptuje również obraz jako pierwszą klatkę. To jest tryb "image-to-video"; trening uwzględnia ten wariant.
- **Dryf fizyki.** Długie klipy (>10s) kumulują subtelne niespójności. Generowanie z przesuwnym oknem + kotwiczenie klatek kluczowych pomaga.

## Użyj Tego

| Zastosowanie | Wybór 2026 |
|-------------|------------|
| Najwyższa jakość text-to-video, hostowane | Veo 3 lub Sora |
| Kinematografia sterowana kamerą | Runway Gen-3 z pędzlami ruchu |
| Spójność postaci w poprzek klipów | Pika 2.0 lub Kling 2.1 |
| Otwarte wagi, szybki fine-tune | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V, Kling 2.1 I2V lub Runway |
| Audio-to-video synchronizacja ust | Veo 3 (natywne audio) lub dedykowany model synchronizacji ust |
| Edycja wideo | Runway Act-Two, Kling Motion Brush, Flux-Kontext (klatka nieruchoma) |

Koszt na sekundę wideo przy równej jakości spadł 20× między 2024 a 2026.

## Wyślij To

Zapisz `outputs/skill-video-brief.md`. Umiejętność przyjmuje brief wideo (czas trwania, proporcje, styl, plan kamery, spójność podmiotu, audio) i zwraca: model + hosting, scaffolding promptu (język kamery, opis podmiotu, deskryptory ruchu), protokół ziarna + odtwarzalności oraz checklistę QA na poziomie klatki.

## Ćwiczenia

1. **Łatwe.** W `code/main.py` porównaj różnicę między klatkami dla (a) niezależnego próbkowania na klatkę, (b) wspólnego próbkowania sekwencji. Podaj średnią i wariancję różnic.
2. **Średnie.** Dodaj warunek pierwszej klatki: przypnij klatkę 0 do danej wartości i próbkuj resztę. Zmierz, jak przypięta wartość propaguje się.
3. **Trudne.** Użyj HuggingFace diffusers do uruchomienia CogVideoX-2B na lokalnym GPU. Zmierz czas 20 kroków wnioskowania przy 720p dla 6-sekundowego klipu. Profiluj attention czasoprzestrzenny, aby zidentyfikować wąskie gardło.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|--------|-----------------|-----------------------|
| VAE wideo | "3-W VAE" | Enkoder, który kompresuje `(T, H, W, C)` → latent czasoprzestrzenny. |
| Łaty | "Tokeny" | Bloki 3-W o stałym rozmiarze latentu; wejście do DiT. |
| Sfaktoryzowany attention | "Przestrzenny + czasowy" | Uruchom attention nad przestrzenią, potem nad czasem; pomiń pełny attention 3-W. |
| Image-to-video (I2V) | "Animuj to zdjęcie" | Model przyjmuje obraz + tekst, zwraca wideo, które od niego zaczyna. |
| Kotwiczenie klatek kluczowych | "Klatki kotwiczne" | Przypnij konkretne klatki, aby kontrolować łuk wideo. |
| Pędzel ruchu | "Wskazówka kierunkowa" | Wejście UI, gdzie użytkownik maluje wektory ruchu na obrazie. |
| Ponowne opisywanie | "Gęste podpisy" | Użycie LLM do ponownego oznaczania klipów treningowych szczegółowymi promptami. |
| Migotanie | "Artefakt czasowy" | Niespójność między klatkami; naprawiane przez sprzężone odszumianie. |

## Uwaga produkcyjna: latenty wideo to problem przepustowości pamięci

10-sekundowy klip 1080p przy 24 fps to 240 klatek × 1920 × 1080 × 3 ≈ 1.5 GB surowych pikseli. Po 4× kompresji VAE wideo (`2 × przestrzenna × 2 × czasowa`) latent to ~100 MB na żądanie. Przepuść to przez spatiotemporalny DiT przez 30 kroków z batchem 1, a przemieszczasz ~3 GB/krok przez HBM — przepustowość pamięci, a nie FLOPy, jest wąskim gardłem.

Trzy produkcyjne gałki, wszystkie wprost z literatury produkcyjnej wnioskowania:

- **TP w poprzek DiT.** Modele text-to-video są rutynowo ≥10B parametrów. TP=4 na 4 H100 to standard; PP=2 × TP=2 dla modeli klasy 405B. Opóźnienie na krok spada mniej więcej liniowo z TP aż do ściany all-reduce.
- **Grupowanie klatek = ciągłe grupowanie.** W czasie generowania wideo jest koncepcyjnie batch'em klatek połączonych attention. Ciągłe grupowanie (planowanie w locie) ma zastosowanie: zacznij renderować klatkę `t+1`, podczas gdy klatka `t-1` jest zwracana, jeśli architektura modelu pozwala na generowanie z przesuwnym oknem.
- **Cache prefiksu na poziomie klipu.** Dla image-to-video, warunkowanie pierwszą klatką jest analogiczne do prefiksu promptu w LLM: oblicz raz, użyj ponownie w przejściach dekodera czasowego. To efektywnie KV-cache dla wideo.

## Dalsza Literatura

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — raport techniczny Sory.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — otwarty SOTA połowa 2025.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — pionierska praca o dyfuzji wideo.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — przodek Stable Video Diffusion.