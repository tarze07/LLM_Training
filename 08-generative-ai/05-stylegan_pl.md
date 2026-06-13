# StyleGAN

> Większość generatorów miesza `z` z każdą warstwą w tym samym czasie. StyleGAN rozdzielił to: najpierw zmapuj `z` do pośredniego `w`, a następnie *wstrzyknij* `w` na każdym poziomie rozdzielczości przez AdaIN. Ta jedna zmiana rozplątała przestrzeń latentną i uczyniła fotorealistyczne twarze rozwiązany problemem na siedem lat.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time:** ~45 minutes

## Problem

DCGAN mapuje `z` na obraz przez stos transponowanych konwolucji. Problem: `z` kontroluje wszystko — pozę, oświetlenie, tożsamość, tło — splątane razem. Przesuń się wzdłuż jednej osi `z`, a zmienią się wszystkie cztery. Nie możesz zapytać modelu "ta sama osoba, inna poza", ponieważ reprezentacja nie jest tak sfaktoryzowana.

Karras i in. (2019, NVIDIA) zaproponowali: przestań podawać `z` bezpośrednio do warstw konwolucyjnych. Podawaj stały tensor `4×4×512` jako wejście sieci. Naucz się 8-warstwowego MLP, który mapuje `z ∈ Z → w ∈ W`. Wstrzykuj `w` na każdej rozdzielczości przez *adaptacyjną normalizację instancyjną* (AdaIN): normalizuj każdą mapę cech konwolucyjnych, a następnie skaluj i przesuwaj przez afiniczne projekcje `w`. Dodaj szum na warstwę dla stochastycznych detali (pory skóry, pasma włosów).

Rezultat: `W` ma w przybliżeniu ortogonalne osie dla "stylu wysokiego poziomu" (poza, tożsamość) vs "stylu drobnego" (oświetlenie, kolor). Możesz zamieniać style między dwoma obrazami, używając `w` obrazu A dla niskich poziomów rozdzielczości i `w` obrazu B dla wysokich. To odblokowało edycję, stylizację między domenami i cały nurt badań "StyleGAN-inversion".

## Koncepcja

![StyleGAN: sieć mapująca + AdaIN + szum na warstwę](../assets/stylegan.svg)

**Sieć mapująca.** `f: Z → W`, 8-warstwowy MLP. `Z = N(0, I)^512`. `W` nie musi być Gaussowskie — uczy się kształtu dostosowanego do danych.

**Sieć syntezy.** Zaczyna od wyuczonej stałej `4×4×512`. Każdy blok rozdzielczości: `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`. Rozdzielczości podwajają się: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

gdzie `y_scale` i `y_bias` pochodzą z afinicznych projekcji `w`. Normalizuj na mapę cech, a następnie ponownie stylizuj. "Styl" to tutaj statystyki pierwszego i drugiego rzędu mapy cech.

**Szum na warstwę.** Jednokanałowy szum Gaussa dodawany do każdej mapy cech, skalowany przez wyuczony współczynnik na kanał. Kontroluje stochastyczne detale bez wpływu na globalną strukturę.

**Triki obcięcia (truncation).** We wnioskowaniu, próbkuj `z`, oblicz `w = mapping(z)`, a następnie `w' = ŵ + ψ·(w - ŵ)`, gdzie `ŵ` to średnie `w` z wielu próbek. `ψ < 1` wymienia różnorodność na jakość. Prawie każde demo StyleGAN używa `ψ ≈ 0.7`.

## StyleGAN 1 → 2 → 3

| Wersja | Rok | Innowacja |
|--------|-----|-----------|
| StyleGAN | 2019 | Sieć mapująca + AdaIN + szum + progresywny wzrost. |
| StyleGAN2 | 2020 | Demodulacja wag zastępuje AdaIN (naprawia artefakty kroplowe); architektura skip/residual; regularyzacja długości ścieżki. |
| StyleGAN3 | 2021 | Konwolucja bez aliasingu + jądra ekwiwariantne; eliminuje przyklejanie tekstury do siatki pikseli. |
| StyleGAN-XL | 2022 | Warunkowy klasowo, 1024², ImageNet. |
| R3GAN | 2024 | Rebranding z silniejszą reg.; zamyka lukę do dyfuzji na FFHQ-1024 z 20x mniejszą liczbą parametrów. |

W 2026 roku StyleGAN3 pozostaje domyślnym wyborem dla (a) wąsko-domenowego fotorealizmu przy wysokim FPS, (b) adaptacji domeny z małą liczbą przykładów (trenuj na nowym zbiorze danych ze 100 obrazami, zamroź mapowanie), (c) edycji opartej na inwersji (znajdź `w`, które rekonstruuje prawdziwe zdjęcie, a następnie edytuj to `w`). Do otwartego text-to-image nie jest to narzędzie — dyfuzja jest.

## Build It

`code/main.py` implementuje zabawkowy "style-GAN lite" w 1-D: mapujący MLP, funkcję syntezy, która przyjmuje wyuczony stały wektor i moduluje go skalowaniem/przesunięciem pochodzącym z `w`, oraz szum na warstwę. Pokazuje, że wstrzykiwanie `w` przez modulację afinicznie dorównuje lub przewyższa konkatenację `z` z wejściem generatora.

### Krok 1: sieć mapująca

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Krok 2: adaptacyjna normalizacja instancyjna

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Skalowanie i przesunięcie na mapę cech pochodzą z `w` przez liniową projekcję.

### Krok 3: szum na warstwę

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma na kanał jest uczona.

## Pułapki

- **Artefakty kroplowe.** StyleGAN 1 generował grudkowate krople w mapach cech, ponieważ AdaIN zerował średnią. Demodulacja wag w StyleGAN 2 naprawia to, skalując wagi konwolucyjne zamiast tego.
- **Przyklejanie tekstury.** StyleGAN 1 i 2 podążały za współrzędnymi pikseli, a nie obiektów (widoczne podczas interpolacji). Konwolucje bez aliasingu w StyleGAN 3 naprawiają to za pomocą okienkowych filtrów sinc.
- **Pokrycie modów.** Obcięcie `ψ < 0.7` wygląda czysto, ale próbkuje z wąskiego stożka; użyj `ψ = 1.0`, jeśli potrzebujesz różnorodności.
- **Inwersja jest stratna.** Odwracanie prawdziwego zdjęcia do `W` zwykle odbywa się przez optymalizację lub enkoder (e4e, ReStyle, HyperStyle). Wyniki dryfują po wielu iteracjach.

## Use It

| Przypadek użycia | Podejście |
|----------------|-----------|
| Fotorealistyczne ludzkie twarze (anime, produkt, wąskie) | StyleGAN3 FFHQ / niestandardowe fine-tune |
| Edycja twarzy ze zdjęcia | Inwersja e4e + StyleSpace / kierunki InterFaceGAN |
| Zamiana twarzy / reenactment | StyleGAN + enkoder + blending |
| Potoki awatarów | StyleGAN3 z ADA dla fine-tune na małej ilości danych |
| Adaptacja domeny z kilku obrazów | Zamroź sieć mapującą, fine-tune syntezę |
| Generacja multimodalna lub warunkowana tekstem | Nie — użyj dyfuzji |

W produkcyjnych demonstracjach, gdzie odpowiedzią jest "zdjęcie twarzy osoby", StyleGAN bije dyfuzję na koszcie wnioskowania (pojedyncze przejście w przód, <10ms na 4090) i ostrości dla tego samego progu jakości.

## Ship It

Zapisz `outputs/skill-stylegan-inversion.md`. Umiejętność przyjmuje prawdziwe zdjęcie i zwraca: metodę inwersji (e4e / ReStyle / HyperStyle), oczekiwaną stratę latentną, budżet edycji (jak daleko w `W` można się przesunąć przed pojawieniem się artefaktów) oraz listę znanych dobrych kierunków edycji (wiek, ekspresja, poza).

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` z `adain_on=True` i `adain_on=False`. Porównaj rozrzut wyjść dla stałego latentu vs zaburzonego latentu.
2. **Średnie.** Zaimplementuj regularyzację mieszania: dla batcha treningowego oblicz `w_a`, `w_b` i zastosuj `w_a` dla pierwszej połowy syntezy i `w_b` dla drugiej połowy. Czy dekoder uczy się rozplątanych stylów?
3. **Trudne.** Weź wytrenowany model StyleGAN3 FFHQ (ffhq-1024.pkl). Znajdź kierunek `w`, który kontroluje "uśmiech", trenując SVM na oznaczonych próbkach; zgłoś, jak daleko można przesunąć, zanim tożsamość dryfuje.

## Kluczowe pojęcia

| Pojęcie | Co ludzie mówią | Co naprawdę znaczy |
|---------|-----------------|--------------------|
| Sieć mapująca | "MLP" | `f: Z → W`, 8 warstw, odsprzęga geometrię latentu od statystyk danych. |
| Przestrzeń W | "Przestrzeń stylów" | Wyjście sieci mapującej; w przybliżeniu rozplątane. |
| AdaIN | "Adaptacyjna norma instancyjna" | Normalizuj mapę cech, następnie skaluj + przesuń przez projekcję `w`. |
| Triki obcięcia | "Psi" | `w = mean + ψ·(w - mean)`, ψ<1 wymienia różnorodność na jakość. |
| Regularyzacja długości ścieżki | "PL reg" | Karze duże zmiany obrazu na jednostkę zmiany `w`; wygładza `W`. |
| Demodulacja wag | "Naprawa StyleGAN2" | Normalizuj wagi konwolucyjne zamiast aktywacji; zabija artefakty kroplowe. |
| Bez aliasingu | "Triki StyleGAN3" | Okienkowe filtry sinc; eliminuje przyklejanie tekstury do siatki pikseli. |
| Inwersja | "Znajdź w dla prawdziwego obrazu" | Optymalizuj lub enkoduj `x → w` tak, by `G(w) ≈ x`. |

## Uwaga produkcyjna: dlaczego StyleGAN wciąż jest wysyłany w 2026 roku

StyleGAN3 na 4090 generuje twarz FFHQ 1024² w poniżej 10 ms — `num_steps = 1`, bez dekodowania VAE, bez przejścia cross-attention. W kategoriach produkcyjnych jest to podłogowe opóźnienie dla dowolnego generatora obrazu. 50-krokowy potok SDXL + VAE-decode przy tej samej rozdzielczości to ~3 sekundy. To **300× różnica**, a dla produktów wąskiej domeny (usługi awatarów, potoki dokumentów tożsamości, generowanie stockowych twarzy) wygrywa na TCO.

Dwie operacyjne konsekwencje:

- **Brak harmonogramu, brak batchera.** Statyczny batch przy docelowym obciążeniu jest optymalny. Ciągłe batching (niezbędne dla LLM i dyfuzji) nie przynosi korzyści, ponieważ każde żądanie ma te same FLOPy.
- **Obcięcie `ψ` to pokrętło bezpieczeństwa.** `ψ < 0.7` próbkuje z wąskiego stożka zakresu sieci mapującej. To jedyna dźwignia, jaką warstwa serwująca ma nad wariancją próbki. Niższe `ψ` przy szczytowym obciążeniu, podnieś dla użytkowników premium.

## Dalsza lektura

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — inwersja e4e.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — nowoczesna minimalistyczna recepta GAN.