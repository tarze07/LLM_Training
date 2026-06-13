# ControlNet, LoRA i Warunkowanie

> Sam tekst to niezgrabny sygnał sterujący. ControlNet pozwala sklonować wytrenowany model dyfuzyjny i sterować go mapą głębi, skeletonem pozy, bazgrołem lub obrazem krawędzi. LoRA pozwala dostroić model z 2B parametrów, trenując 10 milionów parametrów. Razem zamieniły Stable Diffusion z zabawki w produkcyjny potok obrazowy, który w 2026 roku wdraża każda agencja.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — dla podstaw LoRA)
**Time:** ~75 minutes

## Problem

Prompt taki jak "kobieta w czerwonej sukience wyprowadzająca psa na ruchliwej ulicy" nie daje modelowi żadnej informacji o tym, *gdzie* jest pies, *w jakiej pozie* jest kobieta ani *z jakiej perspektywy* jest ulica. Tekst określa około 10% tego, co potrzebujesz do określenia obrazu. Reszta jest wizualna i nie może być skutecznie opisana słowami.

Trenowanie nowego modelu warunkowego od zera dla każdego sygnału (pozy, głębi, canny'ego, segmentacji) jest zbyt kosztowne. Chcesz utrzymać zamrożony kręgosłup SDXL z 2.6B parametrów, dołączyć małą sieć boczną, która odczytuje warunek, i sprawić, by dostosowywała pośrednie cechy kręgosłupa. To jest ControlNet.

Chcesz także nauczyć model nowych koncepcji (swojej twarzy, swojego produktu, swojego stylu) bez ponownego trenowania całego modelu. Chcesz delty 100× mniejszej. To jest LoRA — adaptery niskiego rzędu, które podłączają się do istniejących wag attention.

ControlNet + LoRA + tekst = zestaw narzędzi praktyka w 2026 roku. Większość produkcyjnych potoków obrazowych nakłada 2-5 LoRA, 1-3 ControlNety i IP-Adapter na bazę SDXL / SD3 / Flux.

## Koncepcja

![ControlNet klonuje enkoder; LoRA dodaje delty niskiego rzędu](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

Weź wytrenowane SD. *Sklonuj* enkoderową połowę U-Netu. Zamroź oryginał. Trenuj klona, aby akceptował dodatkowe wejście warunkujące (krawędzie, głębię, pozę). Podłącz klona z powrotem do dekoderowej połowy oryginału za pomocą połączeń pomijających z *zero-konwolucją* (konwolucje 1×1 zainicjowane zerami — zacznij jako no-op, naucz się delty).

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Inicjalizacja zero-konwolucji oznacza, że ControlNet zaczyna jako tożsamość — brak szkody nawet przed treningiem. Trenuj na 1M trójek (prompt, warunek, obraz) ze standardową stratą dyfuzyjną.

ControlNety per-modalność są dostarczane jako małe modele boczne (~360M dla SDXL, ~70M dla SD 1.5). Możesz je komponować podczas wnioskowania:

```
features += weight_a * control_a(głębia) + weight_b * control_b(poza)
```

### LoRA (Hu et al., 2021)

Dla dowolnej warstwy liniowej `W ∈ R^{d×d}` w modelu, zamroź `W` i dodaj deltę niskiego rzędu:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

z `r << d`. Rząd 4-16 jest standardem dla attention, rząd 64-128 dla dużych dostrojeń. Liczba nowych parametrów: `2 · d · r` zamiast `d²`. Dla attention w SDXL z `d=640`, `r=16`: 20k parametrów na adapter zamiast 410k — redukcja 20×. W całym modelu: LoRA ma zwykle 20-200 MB wobec bazowych 5 GB.

Podczas wnioskowania możesz skalować LoRA: `W' = W + α · B @ A`. `α = 0.5-1.5` jest normalne. Wiele LoRA nakłada się addytywnie (z zastrzeżeniem, że oddziałują w nieliniowy sposób).

### IP-Adapter (Ye et al., 2023)

Mały adapter, który przyjmuje *obraz* jako warunek (obok tekstu). Używa enkodera obrazu CLIP do produkcji tokenów obrazu, wstrzykuje je do cross-attention obok tokenów tekstu. ~20MB na model bazowy. Pozwala na "wygeneruj obraz w stylu tego odniesienia" bez LoRA.

## Macierz kompozycyjności

| Narzędzie | Co kontroluje | Rozmiar | Kiedy używać |
|-----------|---------------|---------|--------------|
| ControlNet | Struktura przestrzenna (poza, głębia, krawędzie) | 70-360MB | Dokładny układ, kompozycja |
| LoRA | Styl, podmiot, koncepcja | 20-200MB | Personalizacja, styl |
| IP-Adapter | Styl lub podmiot z obrazu referencyjnego | 20MB | Żaden tekst nie opisze wyglądu |
| Textual Inversion | Pojedyncza koncepcja jako nowy token | 10KB | Przestarzałe, głównie zastąpione przez LoRA |
| DreamBooth | Pełne dostrojenie na podmiocie | 2-5GB | Silna tożsamość, wysokie zasoby |
| T2I-Adapter | Lżejsza alternatywa ControlNet | 70MB | Urządzenia brzegowe, ograniczony budżet |

ControlNet ≈ przestrzenne. LoRA ≈ semantyczne. Używaj obu.

## Zbuduj To

`code/main.py` symuluje dwa mechanizmy na 1-W:

1. **LoRA.** Wytrenowana warstwa liniowa `W`. Zamroź ją. Trenuj niskiego rzędu `B @ A`, tak aby `W + BA` odpowiadała docelowej warstwie liniowej. Pokaż, że `r = 1` wystarczy, aby idealnie nauczyć się korekty rzędu 1.

2. **ControlNet-lite.** "Zamrożony bazowy" predyktor i "sieć boczna", która odczytuje dodatkowy sygnał. Wynik sieci bocznej jest bramkowany przez uczony skalar zainicjowany zerem (nasza wersja zero-konwolucji). Trenuj i obserwuj, jak bramka narasta.

### Krok 1: matematyka LoRA

```python
def lora(W, A, B, x, alpha=1.0):
    # W jest zamrożone; A, B to trenowalne czynniki niskiego rzędu.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Krok 2: sieć boczna z inicjacją zerową

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate zainicjowane na 0
h = base(x) + gated
```

W kroku 0 wynik jest identyczny z bazą. Wczesny trening aktualizuje `gate` powoli — brak katastrofalnego dryfu.

## Pułapki

- **Przeskalowane LoRA.** `α = 2` lub `α = 3` to powszechny hack "zrób to silniejsze", który produkuje przestylizowane / zepsute wyniki. Utrzymuj `α ≤ 1.5`.
- **Konflikt wag ControlNet.** Użycie ControlNetu Pozy z wagą 1.0 i ControlNetu Głębi z wagą 1.0 zwykle przestrzeliwuje. Suma wag ≈ 1.0 to bezpieczna domyślna wartość.
- **LoRA na niewłaściwej bazie.** LoRA SDXL działają po cichu jako no-op na SD 1.5, ponieważ wymiary attention nie są zgodne. Diffusers ostrzega w wersji 0.30+.
- **Dryf Textual Inversion.** Tokeny wytrenowane na jednym checkpointie dryfują poważnie na innym. LoRA jest bardziej przenośna.
- **Łączenie i przechowywanie wag LoRA.** Możesz wtopić LoRA w wagi modelu bazowego dla szybszego wnioskowania (brak dodawania w czasie wykonywania), ale tracisz możliwość skalowania `α` w czasie wykonania. Przechowuj obie wersje.

## Użyj Tego

| Cel | Potok w 2026 |
|-----|--------------|
| Odtworzenie stylu artystycznego marki | LoRA wytrenowana na ~30 wyselekcjonowanych obrazach z rzędem 32 |
| Umieszczenie mojej twarzy w wygenerowanym obrazie | DreamBooth lub LoRA + IP-Adapter-FaceID |
| Określona poza + prompt | ControlNet-Openpose + SDXL + tekst |
| Kompozycja świadoma głębi | ControlNet-Depth + SD3 |
| Odniesienie + prompt | IP-Adapter + tekst |
| Dokładny układ | ControlNet-Scribble lub ControlNet-Canny |
| Zamiana tła | ControlNet-Seg + Inpainting (Lekcja 09) |
| Szybki 1-krokowy styl | LCM-LoRA na SDXL-Turbo |

## Wyślij To

Zapisz `outputs/skill-sd-toolkit-composer.md`. Umiejętność przyjmuje zadanie (zasoby wejściowe: prompt, opcjonalny obraz referencyjny, opcjonalna poza, opcjonalna głębia, opcjonalny bazgroł) i zwraca stos narzędzi, wagi oraz protokół odtwarzalności z ziarnem.

## Ćwiczenia

1. **Łatwe.** W `code/main.py` zmieniaj rząd LoRA `r` od 1 do 4. Przy jakim rzędzie LoRA dokładnie dopasowuje docelową deltę rzędu 2?
2. **Średnie.** Wytrenuj dwie oddzielne LoRA na dwóch docelowych transformacjach. Załaduj je razem i pokaż ich addytywną interakcję. Kiedy interakcja załamuje liniowość?
3. **Trudne.** Użyj diffusers do zestawienia: SDXL-base + Canny-ControlNet (waga 0.8) + LoRA stylu (α 0.8) + IP-Adapter (waga 0.6). Zmierz kompromis FID-wierność-promptu przy zmiennych wagach stosu.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|--------|-----------------|-----------------------|
| ControlNet | "Kontrola przestrzenna" | Sklonowany enkoder + skoki zero-konwolucji; odczytuje obraz warunkujący. |
| Zero-konwolucja | "Zaczyna jako tożsamość" | Konwolucja 1×1 zainicjowana zerami; ControlNet zaczyna jako no-op. |
| LoRA | "Adapter niskiego rzędu" | `W + B @ A`, `r << d`; 100× mniej parametrów niż pełne dostrojenie. |
| rząd r | "Gałka" | Kompresja LoRA; 4-16 typowe, 64+ dla dużej personalizacji. |
| α | "Siła LoRA" | Skalowanie delty LoRA w czasie wykonania. |
| IP-Adapter | "Obraz referencyjny" | Mały adapter warunkowania obrazem przez tokeny CLIP-image. |
| DreamBooth | "Pełne dostrojenie podmiotu" | Trenuj cały model na ~30 obrazach podmiotu. |
| Textual Inversion | "Nowy token" | Naucz tylko nowego osadzenia słowa; przestarzałe, głównie zastąpione. |

## Uwaga produkcyjna: wymiana LoRA, pasma ControlNet, serwowanie wielodostępne

Prawdziwy SaaS tekst-obraz obsługuje setki LoRA i tuziny ControlNetów na tym samym bazowym checkpointie. Problem serwowania wygląda bardzo podobnie do wielodostępności LLM (literatura produkcyjna opisuje przypadek LLM pod hasłem continuous batching i LoRAX / S-LoRA):

- **Gorąca wymiana LoRA, bez scalania.** Scalanie `W' = W + α·B·A` z bazą daje ~3-5% szybsze wnioskowanie na krok, ale zamraża `α` i bazę. Trzymaj LoRA w VRAM jako delty rzędu r; diffusers udostępnia `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` do aktywacji na żądanie. Koszt wymiany to wagi `2 · d · r · num_layers` — skala MB, poniżej sekundy.
- **ControlNet jako drugie pasmo attention.** Sklonowany enkoder działa równolegle z bazą. Dwa ControlNety z wagą 1.0 każdy = dwa dodatkowe przejścia w przód na krok, nie jedno scalone. Zapas rozmiaru batcha spada kwadratowo. Budżetuj ~1.5× kosztu kroku na aktywny ControlNet.
- **Kwantyzowane LoRA też.** Jeśli skwantowałeś bazę (patrz Lekcja 07, Flux na 8GB), delta LoRA również kwantyzuje się czysto do 8-bit lub 4-bit. Ładowanie w stylu QLoRA pozwala na ułożenie 5-10 LoRA na szczycie 4-bitowej bazy Flux bez przekraczania pamięci.

Flux-specific: Notatnik Niels'a Flux-on-8GB kwantyzuje bazę do 4-bitów; nałożenie LoRA stylu (`pipe.load_lora_weights("user/style-lora")`) na tę skwantowaną bazę z `weight_name="pytorch_lora_weights.safetensors"` nadal działa. To jest przepis, który w 2026 roku wdraża większość agencji SaaS.

## Dalsza Literatura

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (pierwotnie dla LLM; przeniesione do dyfuzji).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — lżejsza alternatywa dla ControlNet.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — referencyjne potoki.