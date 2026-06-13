# Autoenkodery i wariacyjne autoenkodery (VAE)

> Zwykły autoenkoder kompresuje, a potem rekonstruuje. Zapamiętuje. Nie generuje. Dodaj jeden trik — wymuś, by kod wyglądał jak Gaussowski — a dostaniesz próbnik. Ten jeden trik, reparametryzacja `z = μ + σ·ε`, jest powodem, dla którego każdy model obrazu oparty na latentnej dyfuzji i flow matchingu, którego używasz w 2026 roku, ma VAE na wejściu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 minutes

## Problem

Skompresuj cyfrę MNIST o 784 pikselach do 16-liczbowego kodu, a następnie zrekonstruuj. Zwykły autoenkoder osiągnie doskonały MSE rekonstrukcji, ale przestrzeń kodów będzie grudkowatym bałaganem. Wybierz losowy punkt w przestrzeni kodów, zdekoduj go, a dostaniesz szum. Nie ma próbnika. To model kompresji przebrany za model generatywny.

To, czego naprawdę chcesz, to: (a) przestrzeń kodów jest czystym, gładkim rozkładem, z którego można próbkować — powiedzmy izotropowym Gaussowskim `N(0, I)`, (b) dekodowanie dowolnej próbki daje wiarygodną cyfrę, oraz (c) enkoder i dekoder wciąż dobrze kompresują. Trzy cele, jedna architektura, jedna funkcja straty.

VAE Kingmy z 2013 roku rozwiązuje to, trenując enkoder do zwracania *rozkładu* `q(z|x) = N(μ(x), σ(x)²)`, przyciągając ten rozkład do priora `N(0, I)` poprzez karę KL, a następnie próbkując `z` z `q(z|x)` przed dekodowaniem. W czasie wnioskowania porzuć enkoder, próbkuj `z ~ N(0, I)`, dekoduj. Kara KL jest tym, co wymusza strukturę przestrzeni kodów.

W 2026 roku VAE rzadko występują samodzielnie — zostały prześcignięte przez dyfuzję pod względem jakości surowego obrazu — ale są enkoderem z wyboru dla każdego modelu latentnej dyfuzji (SD 1/2/XL/3, Flux, AudioCraft). Naucz się VAE, a poznasz niewidzialną pierwszą warstwę każdego potoku obrazu, którego używasz.

## Koncepcja

![Autoenkoder vs VAE: trik reparametryzacji](../assets/vae.svg)

**Autoenkoder.** `z = encoder(x)`, `x̂ = decoder(z)`, strata = `||x - x̂||²`. Przestrzeń kodów nieustrukturyzowana.

**Enkoder VAE.** Zwraca dwa wektory: `μ(x)` i `log σ²(x)`. Definiują one `q(z|x) = N(μ, diag(σ²))`.

**Trik reparametryzacji.** Próbkowanie z `q(z|x)` nie jest różniczkowalne. Przepisz próbkę jako `z = μ + σ·ε`, gdzie `ε ~ N(0, I)`. Teraz `z` jest deterministyczną funkcją `(μ, σ)` plus nieparametryczny szum — gradienty przepływają przez `μ` i `σ`.

**Strata.** Evidence Lower BOund (ELBO), dwa składniki:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Rekonstrukcja popycha `x̂` w kierunku `x`. KL popycha `q(z|x)` w kierunku priora. Są w relacji wymiany. Małe β (<1) = ostrzejsze próbki, przestrzeń kodów mniej Gaussowska. Duże β (>1) = czystsza przestrzeń kodów, rozmyte próbki. β-VAE (Higgins 2017) rozsławiło to pokrętło i zapoczątkowało badania nad rozplątywaniem reprezentacji.

**Próbkowanie.** We wnioskowaniu: wyciągnij `z ~ N(0, I)`, przepuść przez dekoder. Jedno przejście w przód — żadnego iteracyjnego próbkowania jak w dyfuzji.

```figure
vae-latent-grid
```

## Build It

`code/main.py` implementuje malutki VAE bez numpy ani torch. Wejście to 8-wymiarowe syntetyczne dane pochodzące z 2-składnikowej mieszanki Gaussowskiej w 8-D. Enkoder i dekoder to MLP z jedną warstwą ukrytą. Implementujemy aktywację tanh, przejście w przód, stratę i ręcznie napisane przejście wsteczne. Nie produkcyjne — pedagogiczne.

### Krok 1: enkoder w przód

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²` zamiast `σ`, aby wyjście sieci było nieograniczone (softplus σ to pułapka — gradienty umierają przy σ ≈ 0).

### Krok 2: reparametryzacja i dekodowanie

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Krok 3: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

Dokładne KL w postaci zamkniętej, ponieważ oba rozkłady są Gaussowskie. Nie całkuj numerycznie. Ludzie wciąż w 2026 roku wysyłają kod z Monte Carlo estymatami KL — to 3x wolniejsze bez powodu.

### Krok 4: generowanie

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

To jest model generatywny. Pięć linijek.

## Pułapki

- **Zapadnięcie się rozkładu a posteriori.** Składnik KL tak agresywnie popycha `q(z|x) → N(0, I)`, że `z` nie niesie żadnej informacji o `x`. Naprawa: wyżarzanie β (zacznij β=0, zwiększaj do 1), free bits, lub pomiń KL na nieaktywnych wymiarach.
- **Rozmyte próbki.** Gaussowskie prawdopodobieństwo dekodera implikuje MSE rekonstrukcji, które jest Bayesowsko optymalne dla L2 (średniej) — średnia zbioru wiarygodnych cyfr to rozmyta cyfra. Naprawa: dyskretny dekoder (VQ-VAE, NVAE) lub użyj VAE tylko jako enkodera i nałóż dyfuzję na latenty (tak robi Stable Diffusion).
- **β zbyt duże, zbyt wcześnie.** Patrz: zapadnięcie się rozkładu a posteriori. Zacznij od β≈0.01 i zwiększaj.
- **Zbyt mały wymiar latentny.** 16-D działa dla MNIST, 256-D dla ImageNet 256², 2048-D dla ImageNet 1024². VAE Stable Diffusion kompresuje 512×512×3 → 64×64×4 (32x czynnik zmniejszania w obszarze przestrzennym, 32x w kanałach).

## Use It

Stos VAE w 2026 roku:

| Sytuacja | Wybór |
|----------|-------|
| Enkoder latentny obrazu dla dyfuzji | Stable Diffusion VAE (`sd-vae-ft-ema`) lub Flux VAE |
| Enkoder latentny audio | Encodec (Meta), SoundStream, lub DAC (Descript) |
| Latenty wideo | Spatiotemporalne patche Sory, Latte VAE, WAN VAE |
| Uczenie rozplątanych reprezentacji | β-VAE, FactorVAE, TCVAE |
| Dyskretne latenty (do modelowania transformerem) | VQ-VAE, RVQ (ResidualVQ) |
| Ciągłe latenty do generacji | Zwykły VAE, następnie conditionuj model flow/dyfuzji w tej przestrzeni latentnej |

Model latentnej dyfuzji to VAE z modelem dyfuzyjnym żyjącym między enkoderem a dekoderem. VAE robi grubą kompresję, model dyfuzyjny wykonuje ciężką pracę. Ten sam wzorzec dla wideo (VAE + wideo-dyfuzyjny DiT) i audio (Encodec + transformer MusicGen).

## Ship It

Zapisz `outputs/skill-vae-trainer.md`.

Umiejętność przyjmuje: profil zbioru danych + docelowy wymiar latentny + zastosowanie końcowe (rekonstrukcja, próbkowanie lub wejście do latentnej dyfuzji) i zwraca: wybór architektury (zwykły/β/VQ/RVQ), harmonogram β, wymiar latentny, prawdopodobieństwo dekodera (Gaussowskie vs kategoryczne) i plan ewaluacji (MSE rekonstrukcji, KL na wymiar, odległość Frécheta między `q(z|x)` a `N(0, I)`).

## Ćwiczenia

1. **Łatwe.** Zmień `β` w `code/main.py` na `0.01`, `0.1`, `1.0`, `5.0`. Zapisz końcowy MSE rekonstrukcji i KL. Które β jest Pareto-najlepsze dla twoich syntetycznych danych?
2. **Średnie.** Zastąp Gaussowskie prawdopodobieństwo dekodera Bernoullim (strata cross-entropy). Porównaj jakość próbek na binaryzowanej wersji tych samych syntetycznych danych.
3. **Trudne.** Rozszerz `code/main.py` do mini VQ-VAE: zastąp ciągły `z` wyszukiwaniem najbliższego sąsiada w książce kodów o K=32 wpisach. Porównaj MSE rekonstrukcji i zgłoś, ile wpisów książki kodów jest używanych (zapadnięcie się książki kodów jest realne).

## Kluczowe pojęcia

| Pojęcie | Co ludzie mówią | Co naprawdę znaczy |
|---------|-----------------|--------------------|
| Autoenkoder | Sieć enkodująco-dekodująca | `x → z → x̂`, uczy MSE. Niegeneratywny. |
| VAE | AE z próbnikiem | Enkoder zwraca rozkład, kara KL kształtuje przestrzeń kodów. |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; ścisłe, gdy `q = p(z\|x)`. |
| Reparametryzacja | `z = μ + σ·ε` | Przepisuje węzeł stochastyczny jako deterministyczny + czysty szum. Umożliwia backprop przez próbkowanie. |
| Prior | `p(z)` | Docelowy rozkład dla latentu, zazwyczaj `N(0, I)`. |
| Zapadnięcie się rozkładu a posteriori | "Składnik KL wygrywa" | Enkoder ignoruje `x`, zwraca prior; dekoder musi halucynować. |
| β-VAE | Regulowana waga KL | `loss = recon + β·KL`. Wyższe β = bardziej rozplątane, ale rozmyte. |
| VQ-VAE | Dyskretny latent | Zastąp ciągłe `z` najbliższym wektorem z książki kodów; umożliwia modelowanie transformerem. |

## Uwaga produkcyjna: VAE to najgorętsza ścieżka w serwerze dyfuzyjnym

W potoku Stable Diffusion / Flux / SD3 VAE jest wywoływany dwa razy na żądanie — raz do enkodowania (jeśli img2img / inpainting) i raz do dekodowania. Przy 1024² przejście dekodera jest często pojedynczym największym szczytem pamięci aktywacji w całym potoku, ponieważ upsampluje latenty `128×128×16` z powrotem do `1024×1024×3`. Dwie praktyczne konsekwencje:

- **Pokrój lub podziel dekodowanie na kafelki.** `diffusers` udostępnia `pipe.vae.enable_slicing()` i `pipe.vae.enable_tiling()`. Kafelkowanie zamienia mały artefakt na szwie na pamięć `O(tile²)` zamiast `O(H·W)`. Niezbędne dla 1024²+ na konsumenckich GPU.
- **Dekoder w bf16, numeryka fp32 dla końcowego skalowania.** VAE SD 1.x było wydane w fp32 i *cicho produkuje NaN*y przy rzutowaniu na fp16 przy 1024²+. SDXL zawiera `madebyollin/sdxl-vae-fp16-fix` — zawsze preferuj wariant fp16-fix lub użyj bf16.

## Dalsza lektura

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — artykuł o VAE.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — rozplątane β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — obrazowy VAE state-of-the-art.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; VAE jako enkoder.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, standard audio VAE.