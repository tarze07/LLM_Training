# Modele generatywne — Taksonomia i historia

> Każdy model obrazu, tekstu, wideo i model 3D mieści się w jednym z pięciu koszyków. Wybierz zły koszyk, a będziesz walczyć z matematyką tygodniami. Wybierz właściwy, a ostatnie dwanaście lat postępu w tej dziedzinie ułoży się w twojej głowie w czystą całość.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45 minutes

## Problem

Model generatywny robi jedno: mając próbki treningowe pochodzące z nieznanego rozkładu `p_data(x)`, generuje nowe próbki, które wyglądają, jakby pochodziły z tego samego rozkładu. Twarze, zdania, pliki MIDI, struktury białek — to wszystko ten sam problem, jeśli spojrzeć z odpowiedniej perspektywy.

Haczyk polega na tym, że `p_data` żyje w przestrzeni o milionach wymiarów (obraz RGB 512×512 to ~786k wymiarów), próbki znajdują się na cienkiej rozmaitości wewnątrz tej przestrzeni, a ty masz może 10M przykładów. Brutalne wyznaczanie gęstości jest bezcelowe. Każdy model generatywny to kompromis, który zamienia jeden trudny problem na nieco mniej trudny.

Pięć rodzin przetrwało ostatnie dwanaście lat. Znajomość kompromisu, na jaki idzie każda z nich, mówi ci, dlaczego w niektórych zadaniach wygrywa, a w innych upada.

## Koncepcja

![Pięć rodzin modeli generatywnych — taksonomia według tego, co modelują](../assets/taxonomy.svg)

**1. Gęstość jawna, obliczalna.** Zapisz `log p(x)` jako sumę, którą można rzeczywiście wyliczyć. Modele autoregresyjne (PixelCNN, WaveNet, GPT) faktoryzują `p(x) = ∏ p(x_i | x_<i)`. Przepływy normalizujące (RealNVP, Glow) budują `p(x)` jako odwracalną transformację prostej bazy. Zalety: dokładne prawdopodobieństwo, czysta funkcja straty treningowej. Wady: wnioskowanie autoregresyjne jest sekwencyjne (wolne dla długich sekwencji), przepływy wymagają odwracalnych architektur (architektonicznie restrykcyjne).

**2. Gęstość jawna, aproksymacyjna.** Ogranicz `log p(x)` od dołu (ELBO) i optymalizuj to ograniczenie. VAE (Kingma 2013) używają enkodera-dekodera z wariacyjnym rozkładem a posteriori. Modele dyfuzyjne (DDPM, Ho 2020) trenują model usuwający szum, który implicite optymalizuje ważone ELBO. Dyfuzja jest dominującym backbone'em obrazu, wideo i 3D w 2026 roku.

**3. Gęstość niejawna.** Pomiń gęstość całkowicie; naucz się generatora `G(z)`, który produkuje próbki, i dyskryminatora `D(x)`, który odróżnia prawdziwe od fałszywych. GANy (Goodfellow 2014). Szybkie we wnioskowaniu (jedno przejście w przód), ale notorycznie niestabilne podczas trenowania. StyleGAN 1/2/3 pozostają state-of-the-art w fotorealizmie w wąskich domenach (twarze, sypialnie) nawet w 2026 roku.

**4. Oparte na score / ciągłe w czasie.** Naucz się gradientu log-gęstości `∇_x log p(x)` (score) bezpośrednio. Song i Ermon (2019) pokazali, że dopasowywanie score'u uogólnia dyfuzję do SDE. Flow matching (Lipman 2023) to gorący temat lat 2024-2026: trenowanie bez symulacji, prostsze ścieżki, 4-10x szybsze próbkowanie niż DDPM. Stable Diffusion 3, Flux, AudioCraft 2 — wszystkie używają flow matchingu.

**5. Autoregresja na tokenach dyskretnych.** Skompresuj dane wysokowymiarowe za pomocą VQ-VAE lub kwantyzatora resztowego do krótkiej sekwencji dyskretnych tokenów, a następnie użyj Transformera do modelowania sekwencji tokenów. Parti, MuseNet, AudioLM, VALL-E, tokenizer patchy Sory — wszystkie to wykorzystują. To koszyk 1 plus wyuczony tokenizer.

## Krótka historia

| Rok | Model | Dlaczego miał znaczenie |
|------|-------|-------------------------|
| 2013 | VAE (Kingma) | Pierwszy głęboki model generatywny z użyteczną funkcją straty treningowej. |
| 2014 | GAN (Goodfellow) | Gęstość niejawna, brak prawdopodobieństwa — zaskakująco ostre próbki. |
| 2015 | DRAW, PixelCNN | Sekwencyjne generowanie obrazu. |
| 2017 | Glow, RealNVP | Odwracalne przepływy; dokładne prawdopodobieństwo z głębią. |
| 2017 | Progressive GAN | Pierwsze megapikselowe twarze. |
| 2019 | StyleGAN / StyleGAN2 | Fotorealistyczne twarze wciąż trudne do pobicia w tej jednej domenie. |
| 2020 | DDPM (Ho) | Dyfuzja staje się praktyczna. |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image wchodzi do głównego nurtu. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Dyfuzja latentna + conditioning tekstem = towar masowy. |
| 2022 | ControlNet, LoRA | Precyzyjna kontrola nad wytrenowaną dyfuzją. |
| 2023 | SDXL, Midjourney v5, Flow matching | Skala + lepsza dynamika trenowania. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Dyfuzja wideo; flow matching wygrywa. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Wideo w jakości produkcyjnej. |
| 2026 | Consistency + Rectified Flow | Próbkowanie jednoetapowe z backbone'ów dyfuzyjnych. |

## Pięciopytaniowa triaż

Kiedy pojawia się nowy artykuł o modelach generatywnych, odpowiedz na te pięć pytań przed przeczytaniem sekcji z metodą.

1. **Co jest modelowane?** Piksele, latenty, dyskretne tokeny, 3D Gaussianty, meshe, przebiegi?
2. **Czy gęstość jest jawna czy niejawna?** Czy zapisują `log p(x)`?
3. **Próbkowanie: jednoetapowe czy iteracyjne?** Iteracyjne oznacza wolniejsze wnioskowanie; jednoetapowe zwykle oznacza adversarialne lub dystylowane.
4. **Conditioning: bezwarunkowy, klasowy, tekstowy, obrazowy, pozy?** To determinuje funkcję straty i rusztowanie architektury.
5. **Ewaluacja: FID, CLIP score, IS, preferencje ludzkie, dokładność zadania?** Każda ma znane tryby awarii (zobacz Lekcję 14).

Będziesz odpowiadać na te pięć pytań dla każdej lekcji w tej fazie. Pod koniec staną się odruchem.

## Build It

Kod tej lekcji to lekka wizualizacja: dopasowanie 1-wymiarowej mieszanki Gaussowskiej z próbek przy użyciu trzech podejść (gęstość jądrowa, histogram dyskretny i generator "GAN-opodobny" najbliższego sąsiada), aby zobaczyć różnicę między gęstością jawną a niejawną na problemie, który zmieści się na jednym ekranie.

Uruchom `code/main.py`. Rysuje 2000 próbek z dwumodalnej mieszanki Gaussa, a następnie wyświetla:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

Zauważ: dwa pierwsze pozwalają zapytać "jakie jest prawdopodobieństwo tego punktu?". Trzeci nie może. To jest rozróżnienie *jawne vs niejawne*, które będzie miało znaczenie dla każdej przyszłej lekcji.

## Use It

Która rodzina, do którego zadania, w 2026 roku?

| Zadanie | Najlepsza rodzina | Dlaczego |
|---------|-------------------|----------|
| Fotorealistyczne twarze, wąska domena | StyleGAN 2/3 | Wciąż najostrzejsze, najszybsze wnioskowanie. |
| Ogólne text-to-image | Dyfuzja latentna + flow matching | SD3, Flux.1, DALL-E 3. |
| Szybkie text-to-image | Rectified flow + dystylacja | SDXL-Turbo, SD3-Turbo, LCM. |
| Text-to-video | Dyfuzyjny Transformer + flow matching | Sora, Veo 2, Kling. |
| Mowa + muzyka | Tokenowa AR (AudioLM, VALL-E, MusicGen) lub flow matching (AudioCraft 2) | Dyskretne tokeny skalują się tanio. |
| Sceny 3D | Dopasowanie Gaussian Splatting, prior dyfuzyjny | 3D-GS do rekonstrukcji, dyfuzja do nowego widoku. |
| Estymacja gęstości (bez próbkowania) | Przepływy | Jedyna rodzina z dokładnym `log p(x)`. |
| Symulacja / fizyka | Flow matching, score SDE | Proste ścieżki, gładkie pola wektorowe. |

## Ship It

Zapisz jako `outputs/skill-model-chooser.md`.

Umiejętność przyjmuje opis zadania i zwraca: (1) którą rodzinę użyć, (2) rankingowaną listę trzech opcji open-source i trzech hostowanych, (3) prawdopodobny tryb awarii, na który należy uważać, oraz (4) budżet obliczeniowo-czasowy.

## Ćwiczenia

1. **Łatwe.** Dla każdego z tych pięciu produktów zidentyfikuj rodzinę i backbone: ChatGPT image, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. Dowody powinny pochodzić z publicznych raportów technicznych.
2. **Średnie.** Artykuł, który przeczytasz jutro, twierdzi, że ma 100x szybsze próbkowanie niż dyfuzja. Zapisz trzy pytania, aby sprawdzić, czy przyspieszenie utrzymuje się przy conditioning'u i wysokiej rozdzielczości.
3. **Trudne.** Weź jedną domenę, na której ci zależy (np. struktura białek, CAD, molekuły, trajektorie). Odpowiedz na pięciopytaniową triaż dla obecnego SOTA w tej domenie i naszkicuj, co zmieniłby lepszy model.

## Kluczowe pojęcia

| Pojęcie | Co ludzie mówią | Co naprawdę znaczy |
|---------|-----------------|--------------------|
| Model generatywny | "Tworzy nowe rzeczy" | Uczy się próbkować z `p_data(x)`, opcjonalnie udostępnia `log p(x)`. |
| Gęstość jawna | "Można ją wyliczyć" | Model dostarcza zamkniętą lub obliczalną postać `log p(x)`. |
| Gęstość niejawna | "W stylu GAN" | Tylko próbkowanie — brak możliwości wyliczenia `p(x)` dla danego punktu. |
| ELBO | "Evidence lower bound" | Obliczalne dolne ograniczenie na `log p(x)`; VAE i dyfuzja je optymalizują. |
| Score | "Gradient log-gęstości" | `∇_x log p(x)`; modele dyfuzyjne i SDE uczą się tego pola. |
| HIpoteza rozmaitości | "Dane żyją na powierzchni" | Dane wysokowymiarowe koncentrują się na niskowymiarowej rozmaitości; dlatego redukcja wymiarowości działa. |
| Autoregresja | "Przewiduj następny kawałek" | Faktoryzuj łączny rozkład jako iloczyn rozkładów warunkowych. |
| Latent | "Skompresowany kod" | Niskowymiarowa reprezentacja, z której dekoder może zrekonstruować wejście. |

## Uwaga produkcyjna: pięć rodzin, pięć kształtów wnioskowania

Każda rodzina mapuje się na inny profil kosztów serwera wnioskowania. Literatura produkcyjnego wnioskowania opisuje wnioskowanie LLM jako prefill + dekodowanie; ten sam rozkład ma zastosowanie tutaj:

- **Autoregresja (koszyk 1 i 5).** Sekwencyjne dekodowanie dominuje nad opóźnieniem; KV-cache, ciągłe batching i spekulatywne dekodowanie mają bezpośrednie zastosowanie.
- **VAE / dyfuzja / flow matching (koszyki 2 i 4).** Nie ma dekodowania w sensie LLM. Koszt = `num_steps × step_cost`, a `step_cost` to przejście w przód przez transformer lub U-Net w pełnej rozdzielczości latentnej. Gałki produkcyjne to liczba kroków (DDIM / DPM-Solver / dystylacja), rozmiar batcha i precyzja (bf16 / fp8 / int4).
- **GAN (koszyk 3).** Jedno przejście w przód. Żadnego harmonogramu, żadnego KV-cache. TTFT ≈ całkowite opóźnienie. Dlatego StyleGAN wciąż wygrywa w UX wąskich domen.

Kiedy w abstrakcie artykułu widzisz "szybszy niż dyfuzja", przetłumacz to na "mniej kroków × ten sam koszt kroku" lub "te same kroki × tańszy koszt kroku". Wszystko inne to marketing.

## Dalsza lektura

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — artykuł o GAN.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — artykuł o VAE.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — artykuł o DDPM.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — dyfuzja jako SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — artykuł o flow matchingu.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.