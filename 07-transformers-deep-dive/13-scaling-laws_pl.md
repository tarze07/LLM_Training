# Prawa skalowania

> Artykuł Kaplana z 2020 mówił: większy model, niższa strata. Artykuł Hoffmann z 2022 mówił: trenowałeś za mało. Moc obliczeniowa idzie do dwóch wiader — parametrów i tokenów — a podział nie jest oczywisty.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## Problem

Gdy dysponujesz C FLOPs mocy obliczeniowej do trenowania i chcesz najlepszy model, stoisz przed dwoma pokrętłami:

1. **Ile parametrów (N)?** Większy model, większa pojemność.
2. **Ile tokenów treningowych (D)?** Więcej danych, lepsze wykorzystanie pojemności.

FLOPy skalują się w przybliżeniu jak `6 × N × D`. Możesz zwiększać N kosztem D, albo D kosztem N. Co jest lepsze?

Przed 2022 rokiem odpowiedź brzmiała "zwiększaj N." GPT-3 (2020) miał 175B parametrów trenowanych na ~300B tokenów. Stosunek około 1.7 tokena na parametr. Prawa skalowania Kaplana to potwierdzały.

Hoffmann i in. (2022), trenując małą rodzinę modeli o nazwie Chinchilla, odkryli coś innego: optymalny stosunek jest bliższy **20 tokenów na parametr**. GPT-3 był 10× niedotrenowany. Chinchilla (70B parametrów, 1.4T tokenów) pobiła GPT-3 (175B, 300B tokenów) na każdym benchmarku przy 2.5× niższym koszcie inferencji.

Rok 2026 to świat Chinchilli — z jednym ważnym zastrzeżeniem. Llama 3 8B była trenowana na 15 bilionach tokenów, co daje stosunek 1,875 tokenów na parametr. Dziewięćdziesiąt cztery razy więcej niż optimum Chinchilli. Koszt inferencji ma większe znaczenie niż koszt trenowania dla modeli używanych na dużą skalę, więc przetrenowywanie (ponad Chinchillę) dla mniejszego wdrożeniowego footprintu jest domyślnym podejściem w 2026.

## Koncepcja

![Krzywe Chinchilli: strata vs moc obliczeniowa przy różnych stosunkach N/D](../assets/scaling-laws.svg)

### Prawo Hoffmann

Z artykułu o Chinchilli, strata wyraża się wzorem:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parametry (bez embeddingów).
- `D` = tokeny treningowe.
- `α ≈ 0.34`, `β ≈ 0.28` (w przybliżeniu symetryczne).
- `E ≈ 1.69`, nieprzekraczalny sufit straty.
- `A ≈ 406`, `B ≈ 411`.

Dwa składniki równoważą się przy skalowaniu. Weź pochodną względem `N` przy ustalonej mocy obliczeniowej (C = 6ND) i rozwiąż:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Optymalne obliczeniowo: 20 tokenów na parametr.

### Dlaczego jednak przetrenowywanie

Optimum Chinchilli minimalizuje stratę treningową na FLOP treningowy. Ale koszt trenowania płacisz raz; koszt inferencji płacisz zawsze.

Dla czatbota obsługującego bilion tokenów miesięcznie, inferencja dominuje nad całkowitym kosztem. Podejście Llamy: trenuj mniejszy, dłużej. 8B przy 15T tokenów jest głęboko zoptymalizowane pod kątem inferencji:

- Mieści się na konsumenckich GPU.
- Opóźnienie jest ułamkiem modelu 70B z optimum Chinchilli.
- Jakość jest wystarczająco dobra dla większości zadań.

Artykuł DeepMind z 2024 ("Przetrenowywanie to nowe optimum") sformalizował to. Dla obciążeń z dominującą inferencją, właściwy stosunek wynosi bliżej 100–500 tokenów na parametr, w zależności od wolumenu serwowania.

### Emergencja a gładkość

Twierdzenie: pewne zdolności (arytmetyka, wieloetapowe rozumowanie, podążanie za łańcuchem myśli) "wyłaniają się" nagle przy pewnej skali.

Schaeffer i in. (2023) argumentowali, że jest to artefakt pomiarowy: metryki emergencji używają nieciągłego oceniania (dokładne dopasowanie, dokładność przy progu), które ukrywa płynną poprawę w bazowych logitach. Ciągłe metryki (cross-entropia) pokazują gładkie krzywe.

W 2026 roku konsensus jest taki: przewidywania poprzez ciągłą stratę są wiarygodne. Skoki w benchmarkach są często artefaktami oceniania. Planuj budżety względem ciągłych metryk.

### Obraz w 2026 roku

Prawa skalowania wciąż działają, ale:

| Czynnik | Jak się zmieniło |
|---------|------------------|
| Jakość danych | Kuratorowanie "dobrych" tokenów (styl Phi) przesuwa krzywe o >2× efektywnej mocy obliczeniowej |
| MoE | Całkowita liczba parametrów odłącza się od aktywnych FLOPów; prawa skalowania na aktywny FLOP |
| Post-trenowanie | Niektóre zdolności (podążanie za instrukcjami, kod) zmieniają się z SFT+RLHF bardziej niż z pretrenowaniem |
| Multimodalność | Tokeny obrazu + tekstu skalują się razem; osobne krzywe dla każdej modalności |
| Dane syntetyczne | Modele generują dane treningowe; efektywna moc obliczeniowa może się kumulować |

Optymalizator Muon (Kimi Moonlight, 2024) pokazał ~2× wzrost efektywnej mocy obliczeniowej w porównaniu do AdamW przy tych samych danych. Niektóre trenowania w 2026 używają Muona domyślnie. Zmienia to stałą absolutną w prawie skalowania, ale nie jego kształt.

```figure
scaling-laws
```

## Zbuduj To

Zobacz `code/main.py`. Implementujemy równanie straty Chinchilli i rozwiązujemy dla optymalnego obliczeniowo `(N, D)` przy każdym z kilku budżetów obliczeniowych.

### Krok 1: Strata Chinchilli

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

Narysuj `L` jako kontur nad `(N, D)` przy ustalonym `C = 6ND`. Znajdź minimum.

### Krok 2: Granica optymalności obliczeniowej

Dla budżetów obliczeniowych od `1e17` do `1e25` FLOPów, znajdź `(N, D)` minimalizujące stratę przy ograniczeniu `6ND = C`. Zweryfikuj stosunek `D/N ≈ 20`.

### Krok 3: Koszt przetrenowywania

Oblicz dodatkową stratę, jaką płacisz za trenowanie 10× mniejszego modelu (1/10 optymalnego N, 10× optymalnego D). Raportuje oszczędności FLOPów w inferencji (proporcjonalne do N) w zamian.

### Krok 4: Porównanie z rzeczywistymi modelami

Wprowadź znane pary `(N, D)` dla GPT-3, Chinchilli, Llamy 3 8B, DeepSeek-V3 (aktywne parametry) i porównaj przewidywaną z raportowaną stratą.

## Użyj Tego

Prawdopodobnie nie będziesz sam trenować modelu na miarę granicy możliwości. Ale prawa skalowania mówią ci:

1. **Czy twój fine-tune ma wystarczająco dużo danych.** Jeśli twoje dane zadaniowe są poniżej 20 tokenów na parametr modelu bazowego, spodziewaj się nasycenia na pewnym poziomie straty.
2. **Czy wybrać większy model bazowy.** Jeśli wydajesz cały budżet na inferencję, preferuj mniejszy, dłużej trenowany model.
3. **Gdzie zwroty maleją.** Poza 1000× optimum Chinchilli, zmiany log-loss stają się szumem.

**Kierunek badań w 2026 roku:**

- **Reżim ograniczonej ilości danych.** W sieci jest skończona liczba wysokiej jakości tokenów (~5–10 bilionów angielskich po filtrowaniu). Pretrenowanie na granicy możliwości zbliża się do tego sufitu. Dane syntetyczne, wielojęzyczność, multimodalność i skalujący się fine-tuning z RLHF to kolejne dźwignie.
- **Sztuczki z mnożnikiem mocy obliczeniowej.** Optymalizator Muon, MoE, lepsza kuratoracja danych — każde przesuwa stałe absolutne, ale nie asymptotę.
- **Prawa skalowania dla RL.** Otwarte pytanie. Wczesne dowody sugerują prawo potęgowe w próbkach RL, ale z bardzo różnymi wykładnikami niż w pretrenowaniu.

## Dostarcz To

Zobacz `outputs/skill-training-budget-estimator.md`. Umiejętność dobiera `(N, D, godziny, GPU)` dla nowego trenowania, uwzględniając budżet obliczeniowy, ograniczenia wdrożeniowe i docelową stratę.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Wypisz optymalne `(N, D)` według Chinchilli dla budżetów obliczeniowych `1e20`, `1e22`, `1e24`. Porównaj z tabelą rzeczywistych modeli.
2. **Średnie.** Zaimplementuj krzywą straty Hoffmanna w funkcji mocy obliczeniowej. Narysuj stratę vs `log10(C)` dla granicy optymalności obliczeniowej. Zidentyfikuj, kiedy prawo przewiduje, że potrzebujemy `>10^28` FLOPów na kolejne 0.1 redukcji cross-entropii.
3. **Trudne.** Dopasuj własne prawo skalowania na 5 małych modelach (100K do 10M parametrów) trenowanych na tym samym zbiorze danych. Oszacuj `α` i `E`. Jak dobrze twoje wykładniki pasują do opublikowanych?

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|--------|-----------------|-----------------------|
| Parametry (N) | "Rozmiar modelu" | Liczba wag poza embeddingami; określa pojemność. |
| Tokeny (D) | "Dane treningowe" | Liczba tokenów treningowych widzianych przez model; określa, jak dobrze parametry są wykorzystane. |
| Moc obliczeniowa (C) | "Wydane FLOPy" | W przybliżeniu `6 × N × D` dla standardowego transformera. |
| Optimum Chinchilli | "D/N ≈ 20" | Stosunek minimalizujący stratę na FLOP pretrenowania. |
| Przetrenowywanie | "Ponad Chinchillę" | Wydawanie dodatkowych FLOPów treningowych, by zaoszczędzić FLOPy inferencji; D/N >> 20. |
| Strata nieprzekraczalna | "Sufit" | Składnik `E` w prawie skalowania; entropia samych danych. |
| Zdolność emergentna | "Nagłe skoki przy skali" | Często artefakt oceniania; ciągła strata jest gładka. |
| Efektywna moc obliczeniowa | "Mnożnik wydajności trenowania" | Lepsze dane / optymalizator / architektura mnożą, jak daleko sięga jeden FLOP. |

## Dalsza lektura

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — pierwszy artykuł o prawach skalowania; niedotrenowany.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — emergencja jako artefakt pomiarowy.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — dlaczego przetrenowywanie Llamy jest właściwe dla jej obciążenia.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2× mnożnik mocy obliczeniowej.