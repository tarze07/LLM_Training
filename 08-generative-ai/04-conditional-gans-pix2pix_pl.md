# Warunkowe GANy i Pix2Pix

> Pierwszym wielkim odblokowaniem lat 2014-2017 było kontrolowanie tego, co GAN produkuje. Dołącz etykietę, obraz lub zdanie. Pix2Pix zrobił wersję obrazową i wciąż bije każdy ogólny model text-to-image w wąskich zadaniach image-to-image.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75 minutes

## Problem

Bezwarunkowy GAN próbkuje losowe twarze. Przydatne do demo, bezużyteczne w produkcji. Chcesz: *mapować szkic na zdjęcie*, *mapować mapę na zdjęcie lotnicze*, *mapować scenę dzienną na nocną*, *koloryzować obraz w skali szarości*. We wszystkich tych przypadkach dostajesz obraz wejściowy `x` i musisz wyjścić `y` z pewną semantyczną odpowiedniością. Dla każdego `x` istnieje wiele wiarygodnych `y`. Średniokwadratowy błąd spłaszcza je w papkę. Adversarialna strata nie, ponieważ "wygląda prawdziwie" jest ostre.

Warunkowy GAN (Mirza i Osindero, 2014) dodaje warunek `c` jako wejście do zarówno `G`, jak i `D`. Pix2Pix (Isola i in., 2017) wyspecjalizował to: warunkiem jest pełny obraz wejściowy, generatorem jest U-Net, dyskryminatorem jest *patchesny* klasyfikator (PatchGAN), a stratą jest adversarialna + L1. Ta recepta wciąż przewyższa modele text-to-image pisane od zera w wąskich domenach image-to-image nawet w 2026 roku, ponieważ jest trenowana na *sparowanych danych* — masz dokładnie taki sygnał, jakiego potrzebujesz.

## Koncepcja

![Pix2Pix: generator U-Net, dyskryminator PatchGAN](../assets/pix2pix.svg)

**Warunkowy G.** `G(x, z) → y`. W Pix2Pix, `z` to dropout wewnątrz G (bez szumu wejściowego — Isola odkrył, że jawny szum był ignorowany).

**Warunkowy D.** `D(x, y) → [0, 1]`. Wejściem jest *para* (warunek, wyjście). To jest kluczowa różnica: D musi ocenić, czy `y` jest zgodne z `x`, a nie tylko czy `y` wygląda prawdziwie.

**Generator U-Net.** Enkoder-dekoder z połączeniami skipowymi przez wąskie gardło. Kluczowe dla zadań, gdzie wejście i wyjście dzielą niskopoziomową strukturę (krawędzie, sylwetka). Bez skipów szczegóły wysokiej częstotliwości znikają.

**Dyskryminator PatchGAN.** Zamiast zwracać pojedynczy wynik prawdziwe/fałszywe, D zwraca siatkę `N×N`, gdzie każda komórka ocenia pole recepcyjne o rozmiarze ~70×70 pikseli. Uśrednione. Jest to założenie markowskiego pola losowego: realizm jest lokalny. Znacznie szybszy w trenowaniu, mniej parametrów, ostrzejsze wyjście.

**Strata.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

Składnik L1 stabilizuje trenowanie i popycha G w kierunku znanego celu. L1 daje ostrzejsze krawędzie niż L2 (mediany, nie średnie). `λ = 100` było domyślną wartością Pix2Pix.

## CycleGAN — gdy nie masz par

Pix2Pix potrzebuje sparowanych danych `(x, y)`. CycleGAN (Zhu i in., 2017) usuwa ten wymóg kosztem dodatkowej straty: straty *cyklicznej zgodności* (cycle consistency). Dwa generatory `G: X → Y` i `F: Y → X`. Trenuj je tak, aby `F(G(x)) ≈ x` i `G(F(y)) ≈ y`. To pozwala tłumaczyć konie na zebry, lato na zimę, bez sparowanych przykładów.

W 2026 roku niesparowane image-to-image jest głównie robione przez dyfuzję (ControlNet, IP-Adapter) a nie CycleGAN, ale idea cyklicznej zgodności przetrwała w prawie każdym artykule o niesparowanej adaptacji domen.

## Build It

`code/main.py` implementuje mały warunkowy GAN na danych 1-wymiarowych. Warunek `c` to etykieta klasy (0 lub 1). Zadanie: wyprodukuj próbkę z warunkowego rozkładu dla danej klasy.

### Krok 1: dołącz warunek do wejść G i D

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

Kodowanie one-hot to najprostszy sposób. Większe modele używają wyuczonych osadzeń, modulacji FiLM lub cross-attention.

### Krok 2: trenuj warunkowo

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Generator musi dopasować prawdziwy rozkład *dla danego warunku*, a nie brzegowy.

### Krok 3: zweryfikuj wyjście dla każdej klasy

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Pułapki

- **Warunek ignorowany.** G uczy się marginalizować, D nigdy nie karze, ponieważ sygnał warunku jest słaby. Naprawa: conditionuj D bardziej agresywnie (wczesna warstwa, nie tylko późna), użyj dyskryminatora projekcyjnego (Miyato i Koyama 2018).
- **Waga L1 zbyt niska.** G dryfuje w kierunku arbitralnych, realistycznie wyglądających wyjść, a nie wiernych. Zacznij od λ≈100 dla zadań w stylu Pix2Pix.
- **Waga L1 zbyt wysoka.** G produkuje rozmyte wyjścia, ponieważ L1 to wciąż norma L_p. Zmniejszaj, gdy trenowanie się ustabilizuje.
- **Wyciek ground-truth w D.** Konkatenuj `(x, y)` jako wejście D, a nie tylko `y`. Bez tego D nie może sprawdzić zgodności.
- **Zapadnięcie się modów na klasę.** Każda klasa może się zapaść niezależnie. Przeprowadzaj warunkowe kontrole różnorodności.

## Use It

Stan zadań image-to-image w 2026 roku:

| Zadanie | Najlepsze podejście |
|---------|---------------------|
| Szkic → zdjęcie, ta sama domena, dane sparowane | Pix2Pix / Pix2PixHD (wciąż szybkie, wciąż ostre) |
| Szkic → zdjęcie, niesparowane | ControlNet z modelem conditioning'u Scribble |
| Mapa semantyczna → zdjęcie | SPADE / GauGAN2 lub SD + ControlNet-Seg |
| Transfer stylu | Dyfuzja z IP-Adapter lub LoRA; metody GAN są przestarzałe |
| Głębokość → zdjęcie | ControlNet-Depth na Stable Diffusion |
| Super-rozdzielczość | Real-ESRGAN (GAN), ESRGAN-Plus, lub SD-Upscale (dyfuzja) |
| Koloryzacja | ColTran, koloryzatory oparte na dyfuzji, lub Pix2Pix-color |
| Dzień → noc, pory roku, pogoda | CycleGAN lub oparty na ControlNet |

Pix2Pix pozostaje odpowiednim narzędziem, gdy (a) masz tysiące sparowanych przykładów, (b) zadanie jest wąskie i powtarzalne, oraz (c) potrzebujesz szybkiego wnioskowania. W ogólnych zadaniach otwartej domeny wygrywa dyfuzja.

## Ship It

Zapisz `outputs/skill-img2img-chooser.md`. Umiejętność przyjmuje opis zadania, dostępność danych (sparowane vs niesparowane, N próbek) i budżet opóźnienia/jakości, a następnie zwraca: podejście (Pix2Pix, CycleGAN, wariant ControlNet, SDXL + IP-Adapter), wymagania dotyczące danych treningowych, koszt wnioskowania i protokół ewaluacji (LPIPS, FID, specyficzny dla zadania).

## Ćwiczenia

1. **Łatwe.** Zmodyfikuj `code/main.py`, aby dodać trzecią klasę. Potwierdź, że G wciąż mapuje szum każdej klasy do właściwego modu.
2. **Średnie.** Zastąp L1 stratą w stylu percepcyjnym w ustawieniu 1-D (np. mały zamrożony D działający jako ekstraktor cech). Czy zmienia to ostrość warunkowego rozkładu?
3. **Trudne.** Naszkicuj CycleGAN w ustawieniu 1-D: dwa rozkłady, dwa generatory, strata cyklu. Pokaż, że uczy się mapować między nimi bez sparowanych danych.

## Kluczowe pojęcia

| Pojęcie | Co ludzie mówią | Co naprawdę znaczy |
|---------|-----------------|--------------------|
| Warunkowy GAN | "GAN z etykietami" | G(z, c), D(x, c). Obie sieci widzą warunek. |
| Pix2Pix | "Image-to-image GAN" | Sparowany cGAN z U-Net G i PatchGAN D + strata L1. |
| U-Net | "Enkoder-dekoder z skipami" | Symetryczna sieć konwolucyjna; skipy zachowują wysokie częstotliwości. |
| PatchGAN | "Klasyfikator lokalnego realizmu" | D zwraca wynik na patch zamiast globalnego wyniku. |
| CycleGAN | "Niesparowana translacja obrazu" | Dwa G + strata cyklicznej zgodności; brak sparowanych danych. |
| SPADE | "GauGAN" | Normalizuje aktywacje pośrednie mapą semantyczną; segmentacja-na-obraz. |
| FiLM | "Feature-wise linear modulation" | Afiniczna transformacja na cechę z warunku; tanie conditioning. |

## Uwaga produkcyjna: Pix2Pix jako wzorzec ograniczenia opóźnienia

Gdy masz sparowane dane i wąskie zadanie (szkic → render, mapa semantyczna → zdjęcie, dzień → noc), jednoetapowe wnioskowanie Pix2Pix bije dyfuzję o rząd wielkości pod względem opóźnienia. Porównanie produkcyjne wygląda zwykle tak:

| Ścieżka | Kroki | Typowe opóźnienie przy 512² na pojedynczym L4 |
|---------|-------|-----------------------------------------------|
| Pix2Pix (U-Net w przód) | 1 | ~30 ms |
| SD-Inpaint lub SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix wygrywa na przepustowości w statycznych batchach (każde żądanie ma takie same FLOPy). Dyfuzja wygrywa na jakości i generalizacji. Współczesnym podejściem jest często wysłanie modelu dystylowanego w stylu Pix2Pix dla wąskiego zadania i dyfuzyjnego fallbacku dla wejść brzegowych.

## Dalsza lektura

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — artykuł o cGAN.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — projekcyjny D.