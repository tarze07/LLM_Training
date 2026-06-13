# Prywatność Różnicowa dla LLM

> DP-SGD pozostaje standardem — aktualizacje gradientu z szumem zapewniają formalne gwarancje (epsilon, delta). Narzut w obliczeniach, pamięci i użyteczności jest znaczący; parametry-efektywne dostrajanie DP (LoRA + DP-SGD) to powszechna konfiguracja 2025 (ACM 2025). Dwa ciała dowodów w napięciu: wnioskowanie o członkostwie oparte na kanarkach (Duan et al., 2024) raportuje ograniczony sukces przeciwko modelom językowym; ekstrakcja danych treningowych (Carlini et al., 2021; Nasr et al., 2025) odzyskuje znaczącą dosłowną memorizację. Rozwiązanie (arXiv:2503.06808, marzec 2025): luka jest w tym, co jest mierzone — wstawione kanarki vs "najbardziej ekstrahowalne" dane. Nowe projekty kanarków umożliwiają MIA oparte na stracie bez modeli cieni i dają pierwszy nietrywialny audyt DP LLM wytrenowanego na prawdziwych danych z realistycznymi gwarancjami DP. Alternatywy: PMixED (arXiv:2403.15638) — prywatna predykcja w czasie wnioskowania przez mieszankę ekspertów na rozkładach następnego tokena; DP generowanie danych syntetycznych (Google Research 2024). Pojawiający się atak: Odwrócenie Prywatności Różnicowej przez Informację Zwrotną LLM — wyciek wyników ufności.

**Type:** Build
**Languages:** Python (stdlib, demonstracja wstrzykiwania szumu DP-SGD i księgowego ε-δ)
**Prerequisites:** Phase 01 · 09 (information theory), Phase 10 · 01 (large-model training)
**Time:** ~60 minutes

## Learning Objectives

- Zdefiniować (epsilon, delta)-prywatność różnicową i opisać przepis DP-SGD.
- Wyjaśnić napięcie 2024-2025: MIA kanarków vs ekstrakcja danych treningowych dają różne obrazy.
- Opisać PMixED i dlaczego prywatna predykcja w czasie wnioskowania jest alternatywą dla treningu DP.
- Opisać atak Odwrócenia Prywatności Różnicowej przez Informację Zwrotną LLM.

## The Problem

LLM-y memorizują. Carlini et al. 2021 pokazali, że produkcyjne modele językowe odtwarzają dosłowny tekst treningowy na żądanie. DP jest formalną obroną: trenuj tak, aby output był udowodnialnie niewrażliwy na każdy pojedynczy przykład treningowy. Dowody z lat 2024-2025 pokazują, że DP-SGD jest konieczne, ale wdrożone wartości ε mogą nie odpowiadać modelowi zagrożenia.

## The Concept

### (ε, δ)-prywatność różnicowa

Randomizowany algorytm M jest (ε, δ)-DP, jeśli dla dowolnych dwóch zbiorów danych różniących się jednym przykładem i dowolnego zdarzenia S:
P(M(D) w S) <= e^ε * P(M(D') w S) + δ.

Interpretacja: rozkład outputu jest wystarczająco bliski (parametryzowany przez ε), że wkład żadnej pojedynczej osoby nie może być wiarygodnie wywnioskowany, z wyjątkiem prawdopodobieństwa δ.

### DP-SGD

Abadi et al. 2016. Standardowy przepis:
1. Próbkuj mini-partię.
2. Oblicz gradienty dla każdego przykładu.
3. Przytnij każdy gradient dla przykładu do progu C.
4. Zsumuj przycięte gradienty i dodaj szum Gaussa z std σ * C.
5. Użyj zaszumionej sumy do aktualizacji parametrów.

Koszt prywatności jest śledzony przez księgowego (Moments Accountant, Rényi DP accountant). Raportowane wartości ε w literaturze LLM różnią się znacznie w zależności od modelu zagrożenia, wrażliwości danych i celu użyteczności; nie ma uniwersalnie "bezpiecznej" domyślnej wartości ε. Opublikowane przykłady obejmują w przybliżeniu ε ≈ 1–10 w niektórych ustawieniach treningu LLM, ale są one ilustracyjne — nie zalecane domyślne. Niższe ε generalnie wymaga więcej szumu i może zwiększyć stratę użyteczności.

### LoRA + DP-SGD

Pełny DP-SGD modelu frontier jest niewykonalny. LoRA (Hu et al. 2022) ogranicza aktualizacje gradientu do małego adaptera, redukując przechowywanie gradientów dla przykładów. LoRA + DP-SGD to powszechna konfiguracja 2025. Gwarancje DP dotyczą adaptera; model bazowy pozostaje stały.

### Napięcie 2024-2025

Dwie linie dowodów:

- **MIA kanarków (Duan et al. 2024).** Wstaw unikalne kanarki do danych treningowych, zmierz, czy atak wnioskowania o członkostwie może je zidentyfikować. Raportuje ograniczony sukces na modelach językowych. Sugeruje, że MIA jest trudne.
- **Ekstrakcja danych treningowych (Carlini 2021, Nasr et al. 2025).** Podaj modelowi prefiks; zmierz, czy odzyskuje dosłowny tekst z treningu. Raportuje znaczącą memorizację. Sugeruje, że MIA jest łatwe w istotnym sensie.

Rozwiązanie z marca 2025 (arXiv:2503.06808): te dwa mierzą różne rzeczy. MIA pyta "czy przykład e jest w D?" na wstawionych kanarkach. Ekstrakcja pyta "co mogę odzyskać z D?" "Najbardziej ekstrahowalny" przykład jest tym, co ma znaczenie dla prywatności; kanarki zaniżają to, ponieważ nie są zoptymalizowane do bycia ekstrahowalnymi.

Nowe projekty kanarków. MIA oparte na stracie bez modeli cieni. Pierwszy nietrywialny audyt DP LLM na prawdziwych danych z realistycznymi gwarancjami DP.

### Alternatywy dla treningu DP

- **PMixED (arXiv:2403.15638).** Prywatna predykcja w czasie wnioskowania. Mieszanka ekspertów na rozkładach następnego tokena; każdy ekspert widzi fragment danych treningowych; agregacja dodaje szum dla DP. Unika całkowicie treningu DP.
- **DP generowanie danych syntetycznych (Google Research 2024).** LoRA-dostrajanie z DP-SGD, próbkowanie danych syntetycznych, trenowanie klasyfikatora downstream na danych syntetycznych.

Oba omijają koszt użyteczności pełnego treningu DP kosztem innego modelu zagrożenia.

### Odwrócenie Prywatności Różnicowej przez Informację Zwrotną LLM

Pojawiający się atak z 2025. Użyj wyników ufności modelu trenowanego DP jako wyroczni do ponownej identyfikacji osób. Nawet gdy outputy nie wyciekają, rozkłady ufności mogą.

Obrona: nie ujawniaj ufności lub obetnij/skwantyzuj je przed ujawnieniem. To dodatkowy wymóg poza treningiem (ε, δ)-DP.

### Miejsce w Fazie 18

Lekcje 20-21 to uprzedzenie/uczciwość. Lekcja 22 to prywatność. Lekcja 23 to pochodzenie przez znakowanie wodne. Lekcja 27 obejmuje regulacyjną warstwę pochodzenia danych.

## Use It

`code/main.py` symuluje DP-SGD na zabawkowym zbiorze danych klasyfikacji binarnej. Możesz przemieść mnożnik szumu σ i normę przycinania C i śledzić budżet (ε, δ) oraz koszt dokładności. "Atak kanarkowy" wstawia unikalny przykład treningowy i mierzy, czy test straty logarytmicznej może go wykryć przed i po DP.

## Ship It

Ta lekcja produkuje `outputs/skill-dp-audit.md`. Dla twierdzenia DP na wdrożeniu modelu językowego, audytuje: wartości (ε, δ), użytego księgowego, protokół ewaluacji MIA i czy wektory ekspozycji ufności zostały ocenione.

## Exercises

1. Uruchom `code/main.py`. Przemieć σ w {0.5, 1.0, 2.0} i raportuj kompromis (ε, δ)-dokładność. Zidentyfikuj punkt, w którym użyteczność się załamuje.

2. Zaimplementuj wstawienie kanarka i test straty logarytmicznej. Zmierz wskaźnik wykrywalności przed i po DP-SGD przy σ = 1.0.

3. Przeczytaj Nasr et al. 2025 o ekstrakcji danych treningowych. Dlaczego sukces ekstrakcji nie załamuje się przy umiarkowanym ε? Co to implikuje o MIA-jako-ewaluacji?

4. Zaprojektuj wdrożenie używające PMixED (arXiv:2403.15638), które działa całkowicie w czasie wnioskowania. Jaki model zagrożenia adresuje PMixED, którego DP-SGD nie adresuje?

5. Naszkicuj atak Odwrócenia DP przez Informację Zwrotną LLM. Zaprojektuj przeciwdziałanie, które ogranicza wyciek wyników ufności i oszacuj jego koszt wdrożenia.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| DP | "(ε, δ)-prywatność różnicowa" | Formalna prywatność: rozkład outputu bliski przy zmianie sąsiedniego zbioru danych |
| DP-SGD | "SGD z wstrzykiwaniem szumu" | Przycinanie gradientu + dodawanie szumu Gaussa; standardowy trening DP |
| LoRA + DP-SGD | "wydajne prywatne dostrajanie" | DP-SGD na adapterach niskiego rzędu; standardowa konfiguracja 2025 |
| MIA | "wnioskowanie o członkostwie" | Atak określający, czy przykład był w danych treningowych |
| Kanarek | "wstawiony przykładowy znacznik" | Unikalny przykład treningowy używany do pomiaru wycieku DP |
| PMixED | "prywatna mieszanka wnioskowania" | DP w czasie wnioskowania przez mieszankę ekspertów na rozkładach następnego tokena |
| Odwrócenie DP | "atak wycieku ufności" | Atak używający ufności modelu jako wyroczni do ponownej identyfikacji |

## Further Reading

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — standardowy algorytm treningu DP
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — kanoniczny artykuł o ekstrakcji
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — MIA o ograniczonym sukcesie
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808) — rozwiązanie napięcia
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — prywatna predykcja w czasie wnioskowania
