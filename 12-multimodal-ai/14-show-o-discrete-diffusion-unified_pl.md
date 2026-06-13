# Show-o i Ujednolicone Modele z Dyskretną Dyfuzją

> Transfusion miesza ciągłe i dyskretne reprezentacje. Show-o (Xie et al., sierpień 2024) idzie w drugą stronę: tokeny tekstowe używają przyczynowego przewidywania następnego tokena, tokeny obrazkowe używają maskowanej dyskretnej dyfuzji w duchu MaskGIT. Oba znajdują się w jednym transformerze z hybrydową maską atencji. Rezultat jednoczy VQA, tekst-na-obraz, inpaint i generację o mieszanej modalności na jednym szkielecie, jednym tokenizatorze na modalność, jednej formulacji kosztu (przewidywanie następnego tokena rozszerzone do przewidywania maskowanego). Ta lekcja omawia projekt Show-o — dlaczego maskowana dyskretna dyfuzja jest równoległym, kilkukrokowym generatorem obrazów — i kontrastuje z Transfusion i Emu3.

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## Learning Objectives

- Wyjaśnij maskowaną dyskretną dyfuzję: harmonogram, który maskuje tokeny równomiernie, a następnie prosi transformer o ich odtworzenie.
- Porównaj równoległe dekodowanie obrazu (Show-o, MaskGIT) z autogresywnym dekodowaniem obrazu (Chameleon, Emu3) pod względem szybkości i jakości.
- Wymień trzy zadania, które Show-o obsługuje w jednym checkpointie: T2I, VQA, inpaint obrazu.
- Wybierz harmonogram maskowania (cosinusowy, liniowy, obcięty) i uzasadnij jego wpływ na jakość próbki.

## The Problem

Trening z dwoma kosztami Transfusion działa, ale ma trudniejszą dynamikę — ciągły koszt dyfuzji żyje na innej skali numerycznej niż dyskretny koszt NTP. Równoważenie wag kosztów to wyszukiwanie hiperparametrów. Architektura jest skuteczna, ale złożona.

Odpowiedź Show-o: utrzymaj obie modalności dyskretne (jak Chameleon), ale generuj obrazy równolegle poprzez maskowaną dyskretną dyfuzję zamiast sekwencyjnie. Cel treningowy staje się pojedynczym przewidywaniem maskowanego tokena, które naturalnie uogólnia przewidywanie następnego tokena.

## The Concept

### Maskowana dyskretna dyfuzja (MaskGIT)

Oryginalny trik MaskGIT od Chang et al. (2022) jest elegancki. Zacznij od w pełni zamaskowanego obrazu (każdy token ma specjalne ID `<MASK>`). W każdym kroku przewidź wszystkie maskowane tokeny równolegle, następnie zachowaj najbardziej pewne przewidywania top-K i zamaskuj resztę. Po około 8-16 iteracjach wszystkie tokeny są wypełnione. Harmonogram, ile tokenów odmaskować na krok, jest dostrajany — harmonogramy cosinusowe działają dobrze.

Trening jest prosty: wylosuj stosunek maskowania równomiernie z [0, 1], zastosuj go do tokenów VQ obrazu, trenuj transformer, aby odtworzyć maskowane. Dokładnie to, co BERT robił dla tekstu, przeskalowane do generacji obrazów.

### Show-o: jeden transformer, hybrydowa maska

Show-o umieszcza MaskGIT wewnątrz transformer modelu języka przyczynowego. Maska atencji to:

- Tokeny tekstowe: przyczynowe (standardowy LLM).
- Tokeny obrazkowe: pełna dwukierunkowość w obrębie bloku obrazu (tak aby maskowane tokeny mogły widzieć każdy inny token obrazkowy podczas przewidywania).
- Tekst-do-obrazu: tekst skupia się na poprzednich obrazach, obraz skupia się na poprzedzającym tekście.

Trening przeplata między:
1. Standardowym NTP na sekwencjach tekstowych.
2. Próbkami T2I: tekst → obraz z maskowanymi tokenami obrazkowymi, koszt przewidywania maskowanego tokena.
3. Próbkami VQA: obraz → tekst z maskowanymi tokenami tekstowymi (właściwie tylko NTP).

Ujednolicony koszt to entropia krzyżowa na tokenach `<MASK>`, która obejmuje zarówno NTP tekstu (tylko ostatni token jest "zamaskowany"), jak i maskowaną dyfuzję obrazu (losowy podzbiór jest zamaskowany).

### Równoległe samplowanie

Show-o generuje obraz w około 16 krokach zamiast ~1000 (autogresywnych na token) lub ~20 (dyfuzja). W każdym kroku przewidź wszystkie maskowane tokeny równolegle; zatwierdź top-K pewnych; powtórz.

Porównaj:
- Chameleon / Emu3 (autogresywne na tokenach): N_tokenów przejść w przód, typowo 1024-4096 na obraz.
- Transfusion (dyfuzja ciągła): ~20 kroków, każdy to pełne przejście transformera.
- Show-o (maskowana dyskretna dyfuzja): ~16 kroków, każdy to pełne przejście transformera.

Show-o jest szybszy niż Chameleon przy modelach podobnej skali, mniej więcej dorównuje liczbie kroków Transfusion z niższym kosztem na krok (dyskretne logity słownictwa vs ciągły koszt MSE).

### Zadania w jednym checkpointie

Show-o obsługuje cztery zadania na inferencji, wybrane formatem promptu:

- Generacja tekstu: standardowy autogresywny wynik tekstowy.
- VQA: obraz na wejściu, tekst na wyjściu.
- T2I: tekst na wejściu, obraz na wyjściu przez maskowaną dyskretną dyfuzję.
- Inpainting: obraz z niektórymi maskowanymi tokenami, wypełnij.

Możliwość inpaintingu pojawia się za darmo z treningu przewidywania maskowanego. Zamaskuj region siatki tokenów VQ, podaj resztę plus prompt tekstowy, przewidź maskowane tokeny.

### Harmonogram maskowania

Harmonogram, ile tokenów odmaskować na krok, kształtuje jakość. Show-o zaleca cosinusowy:

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

W kroku 0 wszystkie tokeny maskowane (stosunek 1.0). W kroku T żaden nie jest maskowany. Cosinus skupia masę na średnich zakresach, gdzie przewidywanie jest najbardziej informacyjne. Harmonogramy liniowe również działają, ale osiągają plateau szybciej.

### Show-o2

Show-o2 (kontynuacja 2025, arXiv 2506.15564) skaluje Show-o: większa baza LLM, lepszy tokenizator, ulepszony harmonogram maskowania. Ten sam wzorzec architektoniczny.

### Gdzie leży Show-o

W taksonomii 2026:

- Dyskretne tokeny + NTP: Chameleon, Emu3. Proste, ale wolna inferencja.
- Dyskretne tokeny + maskowana dyfuzja: Show-o, MaskGIT, LlamaGen, Muse. Równoległe samplowanie, wciąż stratne przez tokenizator.
- Ciągłe + dyfuzja: Transfusion, MMDiT, DiT. Najwyższa jakość, bardziej złożony trening.
- Ciągłe + dopasowanie przepływu w VLM: JanusFlow, InternVL-U. Najnowsze.

Wybierz według zadania: Show-o, gdy chcesz T2I + inpainting + VQA w jednym otwartym modelu z rozsądną szybkością; Transfusion, gdy jakość jest najważniejsza i możesz sobie pozwolić na instalację dwóch kosztów.

## Use It

`code/main.py` symuluje samplowanie Show-o:

- Zabawkowa siatka 16 tokenów VQ.
- Atrapa "transformer", który przewiduje logity na podstawie promptu i aktualnie odmaskowanych tokenów.
- Równoległe maskowane samplowanie przez 8 kroków z harmonogramem cosinusowym.
- Wypisuje pośrednie stany (ewolucja wzorca maski) i końcowe tokeny.

Uruchom to i obserwuj, jak maska rozpuszcza się krok po kroku.

## Ship It

Ta lekcja produkuje `outputs/skill-unified-gen-model-picker.md`. Na podstawie produktu, który potrzebuje zarówno rozumienia (VQA, podpisywanie), jak i generacji (T2I, inpainting) z ograniczeniem otwartych wag, wybiera między rodziną Show-o, rodziną Transfusion/MMDiT a rodziną Emu3/Chameleon z konkretnymi kompromisami.

## Exercises

1. Maskowana dyskretna dyfuzja sampluje w ~16 krokach. Dlaczego nie w 1? Co się psuje, jeśli odmaskujesz wszystko w kroku 0?

2. Inpainting jest darmowy z maskowaną dyfuzją. Zaproponuj przypadek użycia produktu (rzeczywisty lub hipotetyczny), w którym inpainting Show-o bije wyspecjalizowany model.

3. Harmonogram cosinusowy vs liniowy: prześledź liczbę odmaskowanych tokenów na krok dla T=8. Który jest bardziej zrównoważony?

4. Obraz Show-o 512x512 to 1024 tokeny. Przy słowniku K=16384, model emituje 1024 * log2(16384) = 14,336 bitów (~1.75 KiB) danych. Stable Diffusion wyprowadza 512*512*24 bitów = 6,291,456 bitów (~768 KiB) surowych pikseli. Jaki jest współczynnik kompresji i jaką jakość to kupuje?

5. Przeczytaj LlamaGen (arXiv:2406.06525). Czym różni się klasowo-warunkowy autogresywny model obrazu LlamaGen od maskowanego podejścia Show-o?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Maskowana dyskretna dyfuzja | "Styl MaskGIT" | Trening do przewidywania maskowanych tokenów; na inferencji, iteracyjne odmaskowywanie najbardziej pewnych przewidywań |
| Harmonogram cosinusowy | "Harmonogram odmaskowywania" | Zanik stosunku maski w trakcie kroków inferencji; koncentruje wzrost pewności w średnim zakresie |
| Równoległe dekodowanie | "Wszystkie tokeny naraz" | Każdy krok przewiduje pełną sekwencję maskowanych tokenów w jednym przejściu w przód, następnie zatwierdza top-K |
| Atencja hybrydowa | "Przyczynowa + dwukierunkowa" | Maska przyczynowa na tokenach tekstowych i dwukierunkowa w obrębie bloków obrazowych |
| Inpainting | "Generacja wypełnienia" | Warunkowanie na obrazie z niektórymi maskowanymi tokenami, przewidywanie brakujących; darmowe z celu treningowego |
| Wskaźnik zatwierdzania | "Top-K na krok" | Ile tokenów jest deklarowanych jako "gotowe" na iterację; kontroluje kompromis inferencja vs jakość |

## Further Reading

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)