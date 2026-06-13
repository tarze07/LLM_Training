# Ewaluacja — FID, CLIP Score, Preferencja Ludzka

> Każda tabela liderów modeli generatywnych podaje FID, CLIP score i współczynnik wygranych z areny preferencji ludzkich. Każda z tych liczb ma tryb awarii, który zdeterminowany badacz może wykorzystać. Jeśli nie znasz trybów awarii, nie odróżnisz prawdziwej poprawy od oszukiwania.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**Time:** ~45 minutes

## Problem

Model generatywny oceniany jest na podstawie *jakości próbek* i *zgodności z warunkowaniem*. Żadna z tych rzeczy nie ma miary w formie zamkniętej. Twój model musi wyrenderować 10 000 obrazów; coś musi przypisać im liczby; musisz ufać tym liczbom między rodzinami modeli, rozdzielczościami i architekturami. Trzy metryki przetrwały okres 2014-2026:

- **FID (Fréchet Inception Distance).** Odległość między dwoma rozkładami — rzeczywistym i wygenerowanym — w przestrzeni cech sieci Inception. Niższe = lepiej.
- **CLIP score.** Podobieństwo cosinusowe między embeddingiem CLIP-image wygenerowanego obrazu a embeddingiem CLIP-text promptu. Wyższe = lepiej. Mierzy zgodność z promptem.
- **Preferencja ludzka.** Porównaj dwa modele na tym samym prompcie, niech ludzie (lub model klasy GPT-4) wybiorą lepszy, zagreguj do wyniku Elo.

Zobaczysz również: IS (inception score, w dużej mierze wycofany), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. Każda koryguje jeden błąd poprzedniczki.

## Koncepcja

![FID, CLIP i preferencja: trzy osie, różne tryby awarii](../assets/evaluation.svg)

### FID — jakość próbek

Heusel et al. (2017). Kroki:

1. Wyodrębnij cechy Inception-v3 (2048-D) dla N rzeczywistych i N wygenerowanych obrazów.
2. Dopasuj Gaussianę do każdej puli: oblicz średnią `μ_r, μ_g` i kowariancję `Σ_r, Σ_g`.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

Interpretacja: Odległość Frécheta między dwiema wielowymiarowymi Gaussianami w przestrzeni cech. Niższe = bardziej podobne rozkłady.

Tryby awarii:
- **Obciążenie przy małym N.** FID to średnia kwadratowa z rozkładu cech — małe N niedoszacowuje kowariancję, dając fałszywie niski FID. Zawsze używaj N ≥ 10 000.
- **Zależność od Inception.** Inception-v3 zostało wytrenowane na ImageNet. Domeny odległe od ImageNet (twarze, sztuka, obrazy tekstowe) dają bezsensowny FID. Użyj ekstraktora cech specyficznego dla domeny.
- **Oszukiwanie.** Przeuczenie na priorytecie Inception daje niski FID bez poprawy jakości wizualnej. Pokonaj to za pomocą CMMD (poniżej).

### CLIP score — zgodność z promptem

Radford et al. (2021). Dla wygenerowanego obrazu + promptu:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

Średnia z 30k wygenerowanych obrazów → skalarna wartość porównywalna między modelami.

Tryby awarii:
- **Własne martwe punkty CLIP.** CLIP ma słabe rozumowanie kompozycyjne ("czerwony sześcian na niebieskiej kuli" często zawodzi). Modele mogą osiągać wysokie wyniki CLIP bez rzeczywistego podążania za złożonymi promptami.
- **Obciążenie krótkimi promptami.** Krótkie prompty mają więcej dopasowań CLIP-image w dzikim świecie. Dłuższe prompty mają mechanicznie niższe wyniki CLIP.
- **Oszukiwanie promptem.** Dodanie "high quality, 4k, masterpiece" do promptu zawyża CLIP score bez poprawy wiązania obraz-tekst.

CMMD (Jayasumana et al., 2024) naprawia niektóre z tych problemów: używa cech CLIP zamiast Inception, maksymalną średnią rozbieżność (MMD) zamiast Frécheta. Lepiej wykrywa subtelne różnice jakości.

### Preferencja ludzka — złoty standard

Wybierz pulę promptów. Generuj modelem A i modelem B. Pokaż pary ludziom (lub silnemu sędziemu LLM). Zagreguj wygrane do wyniku Elo lub Bradley-Terry. Benchmarki:

- **PartiPrompts (Google)**: 1 600 różnorodnych promptów, 12 kategorii.
- **HPSv2**: 107k ludzkich adnotacji, szeroko używany jako automatyczny proxy.
- **ImageReward**: 137k par preferencji prompt-obraz, licencja MIT.
- **PickScore**: trenowany na Pick-a-Pic 2.6M preferencji.
- **Areny obrazów w stylu Chatbot-Arena**: https://imagearena.ai/ i inne.

Tryby awarii:
- **Zmienność sędziego.** Niespecjaliści mają inne preferencje niż eksperci. Używaj obu.
- **Dystrybucja promptów.** Starannie wybrane prompty faworyzują jedną rodzinę. Zawsze dokumentuj.
- **Nagradzanie przez sędziego LLM.** GPT-4-sędzia daje się oszukać ładnym, ale błędnym wynikom. Trianguluj z człowiekiem.

## Używaj razem

Produkcyjny raport ewaluacji powinien zawierać:

1. FID na 10-30k próbek względem odłożonego rzeczywistego rozkładu (jakość próbek).
2. CLIP score / CMMD na tych samych próbkach względem ich promptów (zgodność).
3. Współczynnik wygranych w zaślepionej arenie względem poprzedniego modelu (ogólna preferencja).
4. Analiza trybów awarii: 50 losowo wybranych wyników, oznaczonych pod kątem znanych problemów (anatomia dłoni, renderowanie tekstu, spójna liczba obiektów).

Każda pojedyncza metryka to kłamstwo. Trzy potwierdzające się metryki + przegląd jakościowy to twierdzenie.

## Zbuduj to

`code/main.py` implementuje FID, CLIP-score-podobny i agregację Elo na syntetycznych "wektorach cech" (używamy wektorów 4-D jako substytutów cech Inception). Zobaczysz:

- Obliczenia FID na małym N i na dużym N — obciążenie.
- "CLIP score" jako podobieństwo cosinusowe między pulami cech.
- Regułę aktualizacji Elo z syntetycznego strumienia preferencji.

### Krok 1: FID w czterech liniach

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Krok 2: Podobieństwo cosinusowe w stylu CLIP

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Krok 3: Agregacja Elo

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Pułapki

- **FID przy N=1000.** Heurystyka jest niewiarygodna poniżej N=10k. Artykuły raportujące niski FID przy małym N oszukują.
- **Porównywanie FID między rozdzielczościami.** Zmiana rozmiaru Inception 299×299 zmienia rozkład cech. Porównuj tylko przy dopasowanej rozdzielczości.
- **Raportowanie jednego seeda.** Uruchom minimum 3 seedy. Raportuj odchylenie standardowe.
- **Inflacja CLIP score przez negatywne prompty.** Niektóre potoki zawyżają CLIP przez przeuczenie promptu. Sprawdź nasycenie wizualne.
- **Obciążenie Elo przez nakładanie się promptów.** Jeśli oba modele widziały benchmarkowy prompt podczas treningu, Elo jest bez znaczenia. Używaj odłożonych zestawów promptów.
- **Obciążenie płatnych tłumaczy w ewaluacji ludzkiej.** Annotatorzy z Prolific, MTurk są młodsi / bardziej technologiczni. Mieszaj z rekrutowanymi ekspertami sztuki/projektowania.

## Użyj tego

Produkcyjny protokół ewaluacji w 2026:

| Filar | Minimum | Zalecane |
|-------|---------|----------|
| Jakość próbek | FID na 10k względem odłożonych rzeczywistych | + CMMD na 5k + FID na podzbiorze per kategoria |
| Zgodność z promptem | CLIP score na 30k | + HPSv2 + ImageReward + odpowiadanie na pytania w stylu VQA |
| Preferencja | 200 zaślepionych par vs baseline | + 2000 par ludzkich + LLM-sędzia + Chatbot Arena |
| Analiza błędów | 50 ręcznie oznaczonych | 500 ręcznie oznaczonych + automatyczny klasyfikator bezpieczeństwa |

Wszystkie cztery filary w jednym raporcie = twierdzenie. Każdy z osobna = marketing.

## Dostarcz to

Zapisz `outputs/skill-eval-report.md`. Skill przyjmuje nowy checkpoint modelu + baseline i zwraca pełny plan ewaluacji: wielkości próbek, metryki, sondy trybów awarii, kryteria zatwierdzenia.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Porównaj FID przy N=100 vs N=1000 na tych samych syntetycznych rozkładach. Raportuj wielkość obciążenia.
2. **Średnie.** Zaimplementuj CMMD z syntetycznych cech w stylu CLIP (zobacz Jayasumana et al., 2024 po wzór). Porównaj czułość na różnice jakości vs FID.
3. **Trudne.** Odtwórz konfigurację HPSv2: weź 1000 par obraz-prompt z podzbioru Pick-a-Pic, dostrój mały scorer oparty na CLIP na preferencjach i zmierz jego zgodność z odłożonym zbiorem.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| FID | "Fréchet Inception Distance" | Odległość Frécheta dopasowań Gaussa do rzeczywistych vs generowanych cech Inception. |
| CLIP score | "Podobieństwo tekst-obraz" | Podobieństwo cosinusowe między embeddingami CLIP obrazu i tekstu. |
| CMMD | "Zamiennik FID" | MMD cech CLIP; mniej obciążone, bez założenia Gaussa. |
| IS | "Inception score" | Exp KL(p(y|x) || p(y)); słabo koreluje na nowoczesnych modelach, wycofany. |
| HPSv2 / ImageReward / PickScore | "Nauczone proxy preferencji" | Małe modele trenowane na ludzkich preferencjach; używane jako automatyczni sędziowie. |
| Elo | "Ranking szachowy" | Agregacja Bradley-Terry parzystych wygranych. |
| PartiPrompts | "Benchmarkowy zestaw promptów" | 1 600 promptów wyselekcjonowanych przez Google w 12 kategoriach. |
| FD-DINO | "Samonadzorowany zamiennik" | FD z użyciem cech DINOv2; lepszy dla domen poza ImageNet. |

## Uwaga produkcyjna: ewaluacja to również obciążenie inferencyjne

Uruchomienie FID na 10k próbkach oznacza wygenerowanie 10k obrazów. Dla 50-krokowej bazy SDXL przy 1024² na pojedynczym L4 to ~11 godzin inferencji pojedynczego żądania. Budżety ewaluacyjne są realne, a ramy są dokładnie scenariuszem inferencji offline (maksymalizuj przepustowość, ignoruj TTFT):

- **Grupuj w batch, zapomnij o opóźnieniu.** Ewaluacja offline = statyczne batchowanie przy największym rozmiarze mieszczącym się w pamięci. `pipe(...).images` z `num_images_per_prompt=8` na 80GB H100 działa 4-6× szybciej w czasie rzeczywistym niż pojedyncze żądanie.
- **Zachowaj w cache rzeczywiste cechy.** Ekstrakcja cech Inception (FID) lub CLIP (CLIP-score, CMMD) na rzeczywistym zbiorze referencyjnym jest uruchamiana *raz*, przechowywana jako `.npz`. Nie przeliczaj per ewaluacja.

Dla bramek CI / regresji: uruchom FID + CLIP score na podzbiorze 500 próbek na PR (~30 min); uruchom pełny FID 10k + HPSv2 + Elo nocą.

## Dalsza lektura

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — artykuł FID.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — przegląd trybów awarii.