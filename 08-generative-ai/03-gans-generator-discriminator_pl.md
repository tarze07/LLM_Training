# GANy — Generator vs Dyskryminator

> Trik Goodfellowa z 2014 roku polegał na całkowitym pominięciu gęstości. Dwie sieci. Jedna tworzy fałszywki. Druga je łapie. Walczą, aż fałszywki staną się nieodróżnialne od prawdziwych. Nie powinno działać. Często nie działa. Kiedy działa, próbki są wciąż najostrzejsze w literaturze dla wąskich domen.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## Problem

VAE produkują rozmyte próbki, ponieważ ich strata MSE dekodera jest Bayesowsko optymalna dla *średniego* obrazu — a średnia wielu wiarygodnych cyfr to rozmyta cyfra. Potrzebujesz straty, która nagradza *wiarygodność*, a nie pikselową bliskość do jakiegokolwiek pojedynczego celu. Nie ma zamkniętej formy dla wiarygodności. Musisz się jej nauczyć.

Pomysł Goodfellowa: trenuj klasyfikator `D(x)`, aby odróżniać prawdziwe obrazy od fałszywych. Trenuj generator `G(z)`, aby oszukiwał `D`. Sygnał straty dla `G` to to, co `D` aktualnie uważa za sprawiające, że coś wygląda prawdziwie. Sygnał ten aktualizuje się w miarę poprawy `G`, goniąc ruchomy cel. Jeśli obie sieci zbiegną się, `G` nauczył się rozkładu danych, nigdy nie zapisując `log p(x)`.

To jest trenowanie adversarialne. Matematyka to gra minimax:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

W 2026 roku GANy nie są już generatorem SOTA (dyfuzja i flow matching przejęły tę koronę). Ale StyleGAN 2/3 pozostają najostrzejszymi modelami twarzy, jakie kiedykolwiek wydano, dyskryminatory GAN są używane jako *straty percepcyjne* w trenowaniu dyfuzji, a trenowanie adversarialne napędza szybkie 1-krokowe dystylacje (SDXL-Turbo, SD3-Turbo, LCM), które umożliwiają wysyłkę dyfuzji w czasie rzeczywistym.

## Koncepcja

![Trenowanie GAN: generator i dyskryminator w minimax](../assets/gan.svg)

**Generator `G(z)`.** Mapuje wektor szumu `z ~ N(0, I)` do próbki `x̂`. Sieć w kształcie dekodera (gęsta lub transponowana konwolucyjna).

**Dyskryminator `D(x)`.** Mapuje próbkę do skalarnego prawdopodobieństwa (lub score'u). Prawdziwe → 1, fałszywe → 0.

**Strata.** Dwie naprzemienne aktualizacje:

- **Trenuj `D`:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. Binary cross-entropy na real=1, fake=0.
- **Trenuj `G`:** `loss_G = -log D(G(z))`. To jest forma *nienasycająca*, której użył Goodfellow (oryginalne `log(1 - D(G(z)))` nasyca się i zabija gradienty, gdy `D` jest pewne).

**Pętla treningowa.** Jeden krok `D`, jeden krok `G`. Powtórz.

**Dlaczego działa.** Jeśli `G` idealnie dopasuje `p_data`, wtedy `D` nie może działać lepiej niż losowo i zwraca 0.5 wszędzie; `G` nie dostaje już gradientu. Równowaga.

**Dlaczego się psuje.** Zapadnięcie się modów (`G` znajduje jeden mod, którego `D` nie potrafi sklasyfikować, i produkuje go w nieskończoność), zanikający gradient (`D` uczy się zbyt szybko i `log D` się nasyca), niestabilność treningu (współczynniki uczenia, rozmiary batchy, wszystko).

## Warianty, które sprawiły, że GANy działają

| Rok | Innowacja | Naprawa |
|-----|-----------|---------|
| 2015 | DCGAN | Conv/deconv, batch norm, LeakyReLU — pierwsza stabilna architektura. |
| 2017 | WGAN, WGAN-GP | Zastąp BCE odległością Wassersteina + karą gradientową. Naprawia zanikający gradient. |
| 2017 | Spectral normalization | Lipschitz-ogranicz dyskryminator. Wciąż używane w dyskryminatorach w 2026. |
| 2018 | Progressive GAN | Trenuj najpierw niską rozdzielczość, dodawaj warstwy. Pierwsze megapikselowe wyniki. |
| 2019 | StyleGAN / StyleGAN2 | Sieć mapująca + adaptacyjna norma instancyjna. State-of-the-art dla fotorealizmu w wąskiej domenie. |
| 2021 | StyleGAN3 | Bez aliasingu, ekwiwariantny translacyjnie — wciąż złoty standard twarzy w 2026. |
| 2022 | StyleGAN-XL | Warunkowy, świadomy klas, większa skala. |
| 2024 | R3GAN | Rebranding z silniejszą regularyzacją; działa na 1024² bez trików. |

```figure
gan-minimax
```

## Build It

`code/main.py` trenuje malutki GAN na 1-wymiarowych danych: mieszance dwóch Gaussów. Generator i dyskryminator to MLP z jedną warstwą ukrytą. Implementujemy ręcznie przejście w przód, wstecz i pętlę minimax. Celem jest zobaczenie dwóch kluczowych trybów awarii (zapadnięcie się modów + zanikający gradient) w akcji.

### Krok 1: nienasycająca strata

Waniliowa strata Goodfellowa `log(1 - D(G(z)))` spada do 0, gdy D klasyfikuje fałszywkę G jako fałszywą z wysoką pewnością. W tym momencie gradient dla G jest zasadniczo zerowy — G nie może się poprawić. Nienasycająca forma `-log D(G(z))` ma przeciwną asymptotę: eksploduje, gdy D jest pewne, dając G silny sygnał.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Krok 2: jeden krok dyskryminatora na jeden krok generatora

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

Świeże fałszywki dla G, w przeciwnym razie gradienty są nieaktualne.

### Krok 3: obserwuj zapadnięcie się modów

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

Kanoniczny symptom: jeden z dwóch rzeczywistych modów przestaje być generowany. Dyskryminator przestaje go korygować, ponieważ nigdy nie widzi go jako fałszywki.

## Pułapki

- **Dyskryminator zbyt silny.** Zmniejsz współczynnik uczenia D 2-5x, lub dodaj szum instancyjny/warstwowy. Jeśli D osiąga >95% dokładności, G jest martwy.
- **Generator zapamiętuje jeden mod.** Dodaj szum do wejść D, użyj warstwy minibatch-discriminator, lub przełącz się na WGAN-GP.
- **Batch norm wyciekający statystyki.** Rzeczywisty batch + fałszywy batch przepływające przez tę samą warstwę BN mieszają ich statystyki. Użyj instance norm lub spectral norm.
- **Granie na Inception Score.** FID i IS są zaszumione przy małych liczebnościach próbek. Używaj ≥10k próbek przy ewaluacji.
- **Jednoetapowe próbkowanie to kłamstwo dla zadań warunkowych.** Wciąż potrzebujesz skal CFG, trików obcięcia i ponownego próbkowania, aby uzyskać użyteczne wyniki.

## Use It

Stos GAN w 2026 roku:

| Sytuacja | Wybór |
|----------|-------|
| Fotorealistyczne ludzkie twarze, ustalona poza | StyleGAN3 (najostrzejszy, najmniejszy) |
| Twarze anime / stylizowane | StyleGAN-XL lub Stable Diffusion LoRA |
| Translacja obraz-na-obraz | Pix2Pix / CycleGAN (Faza 8 · 04) lub ControlNet (Faza 8 · 08) |
| Szybkie 1-krokowe text-to-image | Adversarialna dystylacja dyfuzji (SDXL-Turbo, SD3-Turbo) |
| Strata percepcyjna wewnątrz trenera dyfuzji | Mały dyskryminator GAN na wycinkach obrazu |
| Cokolwiek multimodalnego, otwartego | Nie — użyj dyfuzji lub flow matchingu |

GANy są ostre, ale wąskie. Gdy twoja domena się otwiera — zdjęcia, dowolne podpowiedzi tekstowe, wideo — przełącz się na dyfuzję. Adversarialny trik żyje dalej jako komponent (straty percepcyjne, dystylacja), a nie samodzielny generator.

## Ship It

Zapisz `outputs/skill-gan-debugger.md`. Umiejętność przyjmuje nieudane uruchomienie GAN (krzywe straty, siatka próbek, rozmiar zbioru danych) i zwraca rankingową listę prawdopodobnych przyczyn, jedno-linijkowe naprawy i protokół ponownego uruchomienia.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` z domyślnymi ustawieniami. Następnie ustaw `D_LR = 5 * G_LR` i uruchom ponownie. Jak szybko strata G zapada się do stałej?
2. **Średnie.** Zastąp stratę BCE Goodfellowa stratą WGAN: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]` i obetnij wagi D do `[-0.01, 0.01]`. Czy trenowanie jest bardziej stabilne? Porównaj czas zbieżności ścienny.
3. **Trudne.** Rozszerz przykład 1-D do danych 2-D (mieszanka 8 Gaussów na pierścieniu). Śledź, ile z 8 modów generator przechwytuje w krokach 1k, 5k, 10k. Zaimplementuj dyskryminację minibatchową i zmierz ponownie.

## Kluczowe pojęcia

| Pojęcie | Co ludzie mówią | Co naprawdę znaczy |
|---------|-----------------|--------------------|
| Generator | "G" | Sieć od szumu do próbki, `G: z → x̂`. |
| Dyskryminator | "D" | Klasyfikator `D: x → [0, 1]`, prawdziwe vs fałszywe. |
| Minimax | "Gra" | `min_G max_D` wspólnego celu. |
| Nienasycająca strata | "Naprawa" | Użyj `-log D(G(z))` dla G zamiast `log(1 - D(G(z)))`. |
| Zapadnięcie się modów | "G zapamiętał jedną rzecz" | Generator produkuje mało różnych wyjść mimo różnorodnych danych. |
| WGAN | "Wasserstein" | Zastąp BCE odległością Earth-Mover + karą gradientową; gładszy gradient. |
| Spectral norm | "Triki Lipschitza" | Ogranicz normy wag D, aby ograniczyć jego nachylenie; stabilizuje trenowanie. |
| StyleGAN | "Ten, który działa" | Sieć mapująca + AdaIN; najlepszy w swojej klasie dla twarzy, wciąż w 2026. |

## Uwaga produkcyjna: jednoetapowe wnioskowanie to trwała przewaga GAN

GANy nie wygrywają już pod względem jakości próbek dla otwartej domeny, ale wciąż wygrywają pod względem kosztu wnioskowania. W słowniku literatury produkcyjnego wnioskowania GAN ma:

- **Brak faz prefill i decode.** Pojedyncze przejście w przód `G(z)`. TTFT ≈ całkowite opóźnienie.
- **Brak presji na KV-cache.** Jedynym stanem są wagi. Rozmiar batcha jest ograniczony przez pamięć aktywacji, nie cache.
- **Trywialne ciągłe batching.** Ponieważ każde żądanie ma te same stałe FLOPy, statyczny batch przy docelowym obciążeniu serwera jest zwykle optymalny. Nie potrzeba harmonogramu w locie.

Dlatego dystylacja GAN (SDXL-Turbo, SD3-Turbo, ADD, LCM) jest dominującą techniką szybkiego text-to-image w 2026 roku: zwija 20-50-krokowy potok dyfuzyjny w 1-4 przejścia w przód w stylu GAN, zachowując rozkład bazy dyfuzyjnej. Adversarialna strata przetrwała jako pokrętło w czasie trenowania do zamiany wolnych generatorów w szybkie.

## Dalsza lektura

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — oryginalny artykuł o GAN.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — pierwsza stabilna architektura.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.