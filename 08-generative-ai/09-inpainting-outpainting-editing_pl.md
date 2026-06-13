# Inpainting, Outpainting i Edycja Obrazu

> Text-to-image tworzy nowe rzeczy. Inpainting naprawia stare. W produkcji 70% płatnej pracy z obrazem to edycja — zamień tło, usuń logo, rozszerz płótno, wygeneruj ponownie dłoń. Inpainting to miejsce, gdzie dyfuzja zarabia na swoje utrzymanie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75 minutes

## Problem

Klient wysyła idealne zdjęcie produktu z rozpraszającym znakiem w tle. Chcesz usunąć znak i pozostawić wszystko inne piksel-w-piksel identyczne. Nie możesz uruchomić text-to-image od zera — wynik będzie miał inny kolor, inne oświetlenie, inny kąt produktu. Chcesz wygenerować ponownie *tylko* zamaskowany region, i chcesz, aby regeneracja respektowała otaczający kontekst.

To jest inpainting. Warianty:

- **Inpainting.** Wygeneruj ponownie wewnątrz maski, zachowaj piksele na zewnątrz.
- **Outpainting.** Wygeneruj ponownie na zewnątrz maski (lub poza płótnem), zachowaj wewnątrz.
- **Edycja obrazu.** Wygeneruj ponownie cały obraz, ale zachowaj wierność semantyczną lub strukturalną wobec oryginału (SDEdit, InstructPix2Pix).

Każdy potok dyfuzyjny w 2026 roku ma tryb inpaintingu. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. Działają na tej samej zasadzie.

## Koncepcja

![Inpainting: maska-świadome odszumianie z reiniekcją zachowującą kontekst](../assets/inpainting.svg)

### Naiwne podejście (i dlaczego jest złe)

Uruchom standardowy text-to-image z maską. Na każdym kroku próbkowania zastąp nie zamaskowany region zaszumionego latentu obrazem czystym poddanym dyfuzji w przód. Działa... źle. Artefakty na granicach przebijają się, ponieważ model nie ma informacji o tym, co jest w zamaskowanym regionie.

### Właściwy model inpaintingu

Wytrenuj zmodyfikowany U-Net, który przyjmuje 9 kanałów wejściowych zamiast 4:

```
input = concat([ nośny_latent (4ch), zakodowany_obraz (4ch), maska (1ch) ], dim=kanał)
```

Dodatkowe kanały to kopia źródłowego obrazu zakodowanego przez VAE plus jednokanałowa maska. Podczas treningu losowo maskujesz regiony obrazu i trenujesz model do odszumiania tylko zamaskowanego regionu, podczas gdy nie zamaskowany region jest podawany jako czysty sygnał warunkujący. Podczas wnioskowania model może "zobaczyć", co otacza zamaskowany region, i produkuje spójne uzupełnienia.

SD-Inpaint, SDXL-Inpaint, Flux-Fill wszystkie używają tego 9-kanałowego (lub analogicznego) wejścia. Diffusers `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — darmowa edycja

Dodaj szum do źródłowego obrazu do pewnego pośredniego `t`, a następnie uruchom odwrotny łańcuch od `t` w dół do 0 z nowym promptem. Bez ponownego treningu. Wybór początkowego `t` wymienia wierność na swobodę twórczą:

- `t/T = 0.3` → prawie identyczny ze źródłem, małe zmiany stylistyczne
- `t/T = 0.6` → umiarkowane edycje, zachowuje grubą strukturę
- `t/T = 0.9` → wygenerowane z blisko szumu, minimalne zachowanie źródła

### InstructPix2Pix (Brooks et al., 2023)

Dostrój model dyfuzyjny na trójkach `(obraz_wejściowy, instrukcja, obraz_wyjściowy)`. Podczas wnioskowania warunkuj zarówno na obrazie wejściowym, jak i na instrukcji tekstowej ("zrób zachód słońca", "dodaj smoka"). Dwie skale CFG: skala obrazu i skala tekstu.

### RePaint (Lugmayr et al., 2022)

Zachowaj standardowy bezwarunkowy model dyfuzyjny. Na każdym odwrotnym kroku próbkuj ponownie — skocz z powrotem do bardziej zaszumionego stanu od czasu do czasu i regeneruj. Unika artefaktów granicznych. Używane, gdy nie masz wytrenowanego modelu inpaintingu.

## Zbuduj To

`code/main.py` implementuje zabawkowy 1-W schemat inpaintingu na danych 5-wymiarowych. Trenujemy DDPM na 5-W danych mieszaniny, gdzie każda próbka to 5 liczb zmiennoprzecinkowych z jednego z dwóch klastrów. Podczas wnioskowania "maskujemy" 2 z 5 wymiarów, wstrzykujemy zaszumioną-w-przód wersję trzech nie zamaskowanych w każdym kroku i regenerujemy tylko zamaskowane wymiary.

### Krok 1: 5-W dane DDPM

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Krok 2: trenuj odszumiacz na wszystkich 5 wymiarach

Standardowy DDPM. Sieć zwraca 5-W predykcję szumu dla 5-W zaszumionego wejścia.

### Krok 3: podczas wnioskowania, maska-świadome odwrotne działanie

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # zastąp nie zamaskowane wymiary świeżo zaszumioną wersją czystego źródła
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...następnie wykonaj normalny odwrotny krok na x_t
```

To jest naiwne podejście i działa na zabawkowych 1-W danych. Prawdziwy inpainting obrazów używa 9-kanałowego wejścia, ponieważ spójność tekstury ma większe znaczenie.

### Krok 4: outpainting

Outpainting to inpainting z odwróconą maską: zamaskuj nowe (wcześniej nieistniejące) płótno, wypełnij resztę oryginałem. Identyczny cel treningowy.

## Pułapki

- **Szwy.** Naiwne podejście pozostawia widoczne granice, ponieważ informacja gradientowa nie przepływa przez maskę. Rozwiązanie: rozszerz maskę o 8-16 pikseli lub użyj właściwego modelu inpaintingu.
- **Wyciek maski.** Jeśli nie zamaskowany region obrazu warunkującego jest niskiej jakości lub zaszumiony, zanieczyszcza generację wewnątrz maski. Odszum lub delikatnie rozmyj.
- **CFG oddziałuje z rozmiarem maski.** Wysokie CFG na małej masce = nasycona łatka. Zmniejsz CFG dla małych edycji.
- **Klif wierności SDEdit.** Przejście od `t/T = 0.5` do `t/T = 0.6` może stracić tożsamość podmiotu. Przeskanuj i robić checkpointy.
- **Niedopasowanie promptu.** Prompt powinien opisywać *cały* obraz, nie tylko nową zawartość. "Kot siedzący na krześle", nie "kot".

## Użyj Tego

| Zadanie | Potok |
|---------|-------|
| Usunięcie obiektu, mała maska | SD-Inpaint lub Flux-Fill, standardowy prompt |
| Zastąpienie nieba | SD-Inpaint + "błękitne niebo o zachodzie słońca" |
| Rozszerzenie płótna | SDXL tryb outpaint (8px wtapianie) lub Flux-Fill z maską outpaint |
| Regeneracja dłoni / twarzy | SD-Inpaint z promptem ponownie opisującym podmiot + ControlNet-Openpose |
| Zmiana stylu jednego regionu | SDEdit przy `t/T=0.5` na zamaskowanym regionie |
| "Zrób zachód słońca" | InstructPix2Pix lub Flux-Kontext |
| Wymiana tła | SAM maska → SD-Inpaint |
| Ultra-wysoka wierność | Flux-Fill lub GPT-Image (hostowane) dla najtrudniejszych przypadków |

SAM (Meta's Segment Anything, 2023) + dyfuzyjny inpaint to potok usuwania tła w 2026 roku. SAM 2 (2024) działa na wideo.

## Wyślij To

Zapisz `outputs/skill-editing-pipeline.md`. Umiejętność przyjmuje oryginalny obraz + opis edycji + opcjonalną maskę (lub prompt SAM) i zwraca: podejście do generowania maski, model bazowy, skale CFG (obraz + tekst), SDEdit-t lub tryb inpaintingu oraz checklistę QA.

## Ćwiczenia

1. **Łatwe.** W `code/main.py` zmieniaj ułamek maskowanych wymiarów od 0.2 do 0.8. Przy jakim ułamku jakość inpaintu (reszta w maskowanych wymiarach) równa się bezwarunkowemu generowaniu?
2. **Średnie.** Zaimplementuj RePaint: co 10. odwrotny krok, skocz z powrotem 5 kroków (dodaj szum) i ponownie odszum. Zmierz, czy zmniejsza to resztę graniczną na krawędzi maski.
3. **Trudne.** Użyj Hugging Face diffusers do porównania: SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill na 20 zadaniach regeneracji twarzy. Oceń osobno wierność pozy i zachowanie tożsamości.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|--------|-----------------|-----------------------|
| Inpainting | "Wypełnij dziurę" | Wygeneruj ponownie wewnątrz maski; zachowaj piksele na zewnątrz. |
| Outpainting | "Rozszerz płótno" | Wygeneruj ponownie poza płótnem; zachowaj wewnątrz. |
| 9-kanałowy U-Net | "Właściwy model inpaintingu" | U-Net z `nośny \| zakodowane-źródło \| maska` jako wejściem. |
| SDEdit | "Img2img z poziomem szumu" | Szum do czasu `t`, odszum z nowym promptem. |
| InstructPix2Pix | "Edycje tylko tekstem" | Dostrojona dyfuzja na trójkach (obraz, instrukcja, wynik). |
| RePaint | "Bez ponownego treningu" | Okresowe ponowne szumienie podczas odwrotnego procesu w celu zmniejszenia szwów. |
| SAM | "Segmentuj wszystko" | Generator maski przez kliknięcia lub ramki; łączy się z inpaintem. |
| Flux-Kontext | "Edytuj z kontekstem" | Wariant Flux, który przyjmuje obraz referencyjny + instrukcję do edycji. |

## Uwaga produkcyjna: potoki edycyjne są wrażliwe na opóźnienie

Użytkownicy edytujący obraz oczekują czasów odpowiedzi poniżej 5 sekund. SDXL-Inpaint z 30 krokami przy 1024² to 3-4 s na L4, plus generowanie maski SAM (~200 ms) i VAE encode/decode (~500 ms łącznie). W ujęciu produkcyjnym jest to ograniczone przez TTFT, a nie przepustowość — batch 1, niska współbieżność, minimalizuj każdy etap:

- **SAM-H jest tym wolnym.** SAM-H przy 1024² to ~200 ms; SAM-ViT-B to ~40 ms z niewielką utratą jakości. SAM 2 (wideo) dodaje narzut czasowy; nie używaj go do edycji pojedynczych obrazów.
- **Pomiń kodowanie, gdy to możliwe.** `pipe.image_processor.preprocess(img)` koduje do latentów. Jeśli masz latenty z poprzedniego generowania (typowe w iteracyjnych UI edycyjnych), przekaż je bezpośrednio przez `latents=...`, aby pominąć jedno kodowanie VAE.
- **Dylatacja maski ma znaczenie również dla przepustowości.** Mała maska oznacza, że większość przejścia w przód U-Netu jest marnowana (niemaskowane piksele są i tak zaciskane). `diffusers`' `StableDiffusionInpaintPipeline` uruchamia pełny U-Net niezależnie; tylko 9-kanałowe warianty właściwego inpaintingu wykorzystują maskowane obliczenia.
- **Flux-Kontext to odpowiedź z 2025 roku.** Pojedyncze przejście w przód nad `(obraz_źródłowy, instrukcja)` — bez osobnej maski, bez przeszukiwania szumu SDEdit. Na H100 dostarcza edycję w ~1.5 s. Lekcja architektoniczna: skróć etapy.

## Dalsza Literatura

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — inpainting bez treningu.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — edycja przez instrukcję tekstową.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, źródło maski.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — SAM wideo.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — edycja na poziomie attention.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — narzędzia 2024.