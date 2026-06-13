# Latentna Dyfuzja i Stable Diffusion

> Dyfuzja w przestrzeni pikseli na obrazach 512×512 to zbrodnia wojenna w obliczeniach. Rombach et al. (2022) zauważyli, że nie potrzebujesz wszystkich 786k wymiarów, aby wygenerować obraz — potrzebujesz wystarczająco dużo, by uchwycić strukturę semantyczną, oraz osobnego dekodera dla reszty. Przeprowadź dyfuzję w przestrzeni latentnej VAE. Ten jeden pomysł to Stable Diffusion.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75 minutes

## Problem

Dyfuzja w przestrzeni pikseli przy 512² oznacza, że U-Net operuje na tensorach kształtu `[B, 3, 512, 512]`. Każdy krok próbkowania to ~100 GFLOPS dla U-Netu z 500M parametrami. Pięćdziesiąt kroków to 5 TFLOPS na obraz. Trenuj na miliardzie obrazów, a rachunek za obliczenia jest absurdalny.

Większość tych FLOPów idzie na przepychanie percepcyjnie nieistotnych szczegółów przez sieć — wysokoczęstotliwościowej tekstury, którą stratny VAE mógłby skompresować. Pomysł Rombacha: wytrenuj VAE raz (pierwszy *etap*), zamroź go i przeprowadź dyfuzję w całości w 4-kanałowej przestrzeni latentnej 64×64 (drugi *etap*). Ten sam U-Net. 1/16 pikseli. ~64× mniej FLOPów przy porównywalnej jakości.

To jest przepis Stable Diffusion. SD 1.x / 2.x używały U-Netu 860M na latentach `64×64×4`, SDXL używał U-Netu 2.6B na latentach `128×128×4`, SD3 zamieniło U-Net na Diffusion Transformer (DiT) z flow matching. Flux.1-dev (Black Forest Labs, 2024) dostarcza 12B-parametrowy DiT-MMDiT. Wszystkie działają na tym samym dwuetapowym podłożu.

## Koncepcja

![Latentna dyfuzja: kompresja VAE + dyfuzja w przestrzeni latentnej](../assets/latent-diffusion.svg)

**Dwa etapy, trenowane osobno.**

1. **Etap 1 — VAE.** Enkoder `E(x) → z`, dekoder `D(z) → x`. Docelowa kompresja: 8× zmniejszenie w każdej osi przestrzennej + dostosowanie kanałów, tak aby całkowity rozmiar latentny wynosił ~1/16 rozmiaru pikseli. Strata = rekonstrukcja (L1 + LPIPS percepcyjna) + KL (mała waga, aby `z` nie było zbyt gaussowskie, bo nie potrzebujemy dokładnego próbkowania z `z`). Często trenowana ze stratą adversarialną, aby zdekodowane obrazy były ostre.

2. **Etap 2 — dyfuzja na `z`.** Traktuj `z = E(x_real)` jako dane. Trenuj U-Net (lub DiT) do odszumiania `z_t`. Podczas wnioskowania: próbkuj `z_0` przez dyfuzję, a następnie `x = D(z_0)`.

**Warunek tekstowy.** Dwa dodatkowe komponenty. Zamrożony enkoder tekstu (CLIP-L dla SD 1.x, CLIP-L+OpenCLIP-G dla SD 2/XL, T5-XXL dla SD3 i Flux). Wstrzyknięcie cross-attention: każdy blok U-Netu przyjmuje `[Q = cechy obrazu, K = V = tokeny tekstu]` i miesza je. Tokeny są jedynym sposobem, w jaki tekst wpływa na obraz.

**Funkcja straty jest identyczna jak w Lekcji 06.** To samo DDPM / flow matching MSE na szumie. Po prostu zamieniasz dziedzinę danych.

## Warianty architektury

| Model | Rok | Kręgosłup | Kształt latentny | Enkoder tekstu | Parametry |
|-------|-----|-----------|------------------|----------------|-----------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokenów) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Destylowany | 128×128×4 | ten sam | 1-4 kroki próbkowania |
| SD3 | 2024 | MMDiT (wielomodalny DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT destylowany | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 kroki |

Trend: zastąp U-Net DiT (transformer na łatkach latentnych), skalaj enkoder tekstu (T5 bije CLIP w dokładności podążania za promptem), zwiększaj kanały latentne (4 → 16 daje więcej przestrzeni na szczegóły).

```figure
noise-schedule
```

## Zbuduj To

`code/main.py` łączy zabawkowy 1-W "VAE" (enkoder tożsamościowy + dekoder, dla demonstracji; prawdziwy VAE byłby siecią konwolucyjną) na szczycie DDPM z Lekcji 06 i dodaje warunkowanie klasą z klasyfikator-bezprzewodowym sterowaniem. Pokazuje, że ta sama strata dyfuzyjna działa niezależnie od tego, czy operujesz na surowych wartościach 1-W, czy na zakodowanych — to kluczowy wgląd.

### Krok 1: enkoder/dekoder

```python
def encode(x):    return x * 0.5          # zabawkowa "kompresja" do mniejszej skali
def decode(z):    return z * 2.0
```

Prawdziwy VAE ma wytrenowane wagi. Dla pedagogiki ta liniowa mapa wystarczy, aby pokazać, że dyfuzja operuje na `z` bez przejmowania się oryginalną przestrzenią danych.

### Krok 2: dyfuzja w przestrzeni `z`

Ten sam DDPM co w Lekcji 06. Dane, które widzi sieć, to `z = E(x)`. Po próbkowaniu `z_0`, dekoduj przez `D(z_0)`.

### Krok 3: klasyfikator-bezprzewodowe sterowanie

Podczas treningu pomiń etykietę klasy w 10% przypadków (zastąp tokenem zerowym). Podczas wnioskowania oblicz zarówno `ε_cond`, jak i `ε_uncond`, a następnie:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = brak sterowania (pełna różnorodność), `w = 3` = domyślne, `w = 7+` = nasycone / zbyt ostre.

### Krok 4: warunkowanie tekstem (koncepcja, nie kod)

Zastąp etykietę klasy wynikiem zamrożonego enkodera tekstu. Wprowadź osadzenie tekstu do U-Netu przez cross-attention:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

To jest jedyna istotna różnica między modelem dyfuzyjnym warunkowanym klasą a Stable Diffusion.

## Pułapki

- **Niedopasowanie skali VAE.** VAE SD 1.x mają stałą skalowania (`scaling_factor ≈ 0.18215`) stosowaną po kodowaniu. Zapomnienie o tym powoduje, że U-Net trenuje na latentach o drastycznie błędnej wariancji. Każdy checkpoint ją dostarcza.
- **Cichy błąd enkodera tekstu.** SD3 potrzebuje T5-XXL z >=128 tokenami, a przejście na sam CLIP jest stratne. Zawsze sprawdzaj `use_t5=True`, bo wierność promptu spada.
- **Mieszanie przestrzeni latentnych.** SDXL, SD3 i Flux używają różnych VAE. LoRA wytrenowana na latentach SDXL nie będzie działać na SD3. Hugging Face diffusers 0.30+ odmawia ładowania niedopasowanych checkpointów.
- **Zbyt wysokie CFG.** `w > 10` produkuje nasycone, oleiste obrazy i przydopasowuje się do promptu kosztem różnorodności. Optymalny zakres to `w = 3-7`.
- **Wyciekające negatywne prompty.** Pusty negatywny prompt staje się tokenem zerowym; wypełniony negatywny prompt staje się `ε_uncond`. To nie to samo; niektóre potoki domyślnie używają zerowego.

## Użyj Tego

Produkcyjne stacki w 2026:

| Cel | Zalecany kręgosłup |
|-----|--------------------|
| Wąska dziedzina, sparowane dane, trenowanie modelu od zera | SDXL fine-tune (LoRA / pełny) — najszybsze do wdrożenia |
| Otwarta domena text-to-image, otwarte wagi | Flux.1-dev (12B, Apache / niekomercyjny) lub SD3.5-Large |
| Najszybsze wnioskowanie, otwarte wagi | Flux.1-schnell (1-4 kroki, Apache) lub SDXL-Lightning |
| Najlepsza wierność promptu, hostowane | GPT-Image / DALL-E 3 (nadal), Midjourney v7, Imagen 4 |
| Przepływy edycyjne | Flux.1-Kontext (grudzień 2024) — natywnie akceptuje obraz + tekst |
| Badania, punkt odniesienia | SD 1.5 — starożytny, ale dobrze zbadany |

## Wyślij To

Zapisz `outputs/skill-sd-prompter.md`. Umiejętność przyjmuje prompt tekstowy + docelowy styl i zwraca: model + checkpoint, skalę CFG, próbnik, negatywny prompt, rozdzielczość, opcjonalną kombinację ControlNet/IP-Adapter oraz checklistę QA na krok.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` ze sterowaniem `w ∈ {0, 1, 3, 7, 15}`. Zapisz średnią próbkę według klasy. Przy jakim `w` średnie klas odbiegają dalej niż rzeczywiste średnie danych?
2. **Średnie.** Zamień zabawkowy liniowy enkoder na parę tanh-MLP enkoder/dekoder ze stratą rekonstrukcji. Przeprowadź ponownie trening dyfuzji na nowych latentach. Czy jakość próbek się zmienia?
3. **Trudne.** Skonfiguruj prawdziwe wnioskowanie Stable Diffusion z diffusers: załaduj `sdxl-base`, uruchom 30 kroków Eulera z CFG=7, zmierz czas. Teraz przełącz na `sdxl-turbo` z 4 krokami i CFG=0. Ten sam temat, inna jakość — opisz, co się zmieniło i dlaczego.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|--------|-----------------|-----------------------|
| Pierwszy etap | "VAE" | Wytrenowana para enkoder/dekoder; kompresuje 512² do 64². |
| Drugi etap | "U-Net" | Model dyfuzyjny w przestrzeni latentnej. |
| CFG | "Skala sterowania" | `(1+w)·ε_cond - w·ε_uncond`; reguluje siłę warunku. |
| Token zerowy | "Osadzenie pustego promptu" | Osadzenie bezwarunkowe używane dla `ε_uncond`. |
| Cross-attention | "Jak tekst wchodzi" | Każdy blok U-Netu uwzględnia tokeny tekstu jako K i V. |
| DiT | "Dyfuzyjny Transformer" | Zastąpienie U-Netu transformerem na łatkach latentnych; lepiej się skaluje. |
| MMDiT | "Wielomodalny DiT" | Architektura SD3: strumienie tekstu i obrazu ze wspólnym attention. |
| Współczynnik skalowania VAE | "Magiczna liczba" | Dzieli latenty przez ~5.4, aby dyfuzja działała w przestrzeni o jednostkowej wariancji. |

## Uwaga produkcyjna: uruchamianie Flux-12B na 8GB konsumenckim GPU

Referencyjna integracja Flux to kanoniczny przepis "Mam konsumenckie GPU, czy mogę to wdrożyć?". Sztuczka to ten sam trzygałkowy przepis z literatury produkcyjnej wnioskowania, zastosowany do dyfuzyjnego DiT:

1. **Przesunięte ładowanie.** Flux ma trzy sieci, które nigdy nie muszą współistnieć w VRAM: enkoder tekstu T5-XXL (~10 GB w fp32), CLIP-L (mały), 12B MMDiT oraz VAE. Zakoduj prompt najpierw, *usuń* enkodery, załaduj DiT, odszum, *usuń* DiT, załaduj VAE, dekoduj. Konsumenckie 8GB GPU mieszczą tylko jeden etap naraz.
2. **Kwantyzacja 4-bitowa przez bitsandbytes.** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` zarówno na enkoderze T5, jak i DiT. Zmniejsza pamięć 8×, spadek jakości jest niezauważalny dla text-to-image według benchmarków Aritry (link w notatniku).
3. **CPU offload.** `pipe.enable_model_cpu_offload()` automatycznie przełącza moduły między CPU a GPU w miarę postępu każdego przejścia w przód. Dodaje 10-20% opóźnienia, ale sprawia, że potok w ogóle działa.

Rachunek pamięci to: `10 GB T5 / 8 = 1.25 GB` skwantowane, `12 B parametrów × 0.5 bajta = ~6 GB` skwantowany DiT, plus aktywacje. W terminologii stas00 jest to skrajny koniec wnioskowania TP=1 — bez równoległości modelu, maksymalna kwantyzacja. W produkcji uruchomiłbyś TP=2 lub TP=4 na H100; na pojedynczym laptopie deweloperskim jest to przepis.

## Dalsza Literatura

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — rodzina Flux.1.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — referencyjna implementacja dla każdego checkpointu powyżej.