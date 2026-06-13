# Visual Autoregressive Modeling (VAR): Predykcja Następnej Skali

> Modele dyfuzyjne próbkują iteracyjnie w czasie (kroki denoisingu). VAR próbkuje iteracyjnie w skali — przewiduje token 1x1, następnie 2x2, potem 4x4, aż do końcowej rozdzielczości, każda skala warunkując się na poprzedniej. Artykuł z 2024 roku pokazał, że VAR osiąga skalowanie zgodne z prawami GPT dla generacji obrazów i pokonuje DiT przy tym samym budżecie obliczeniowym. Ta lekcja buduje podstawowy mechanizm.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## Problem

Generacja autoregresywna zdominowała modelowanie języka, ponieważ skaluje się w przewidywalny sposób: więcej obliczeń, więcej parametrów, niższa perplexity, lepsze wyniki. Generacja obrazów miała dwie główne próby AR przed 2024: PixelRNN/PixelCNN (piksel po pikselu) oraz DALL-E 1 / Parti / MuseGAN (token po tokenie na kodach VQ-VAE).

Obie cierpiały na problem kolejności generowania. Piksele i tokeny są ułożone w siatce 2D, ale model AR musi odwiedzać je w liniowej kolejności rasterowej. Wczesny piksel w rogu nie ma pojęcia, czym obraz ostatecznie się stanie. Jakość generacji skaluje się gorzej niż GPT-na-tekście i nigdy nie osiągnęła jakości modeli dyfuzyjnych przy porównywalnych obliczeniach.

VAR rozwiązuje problem kolejności generowania poprzez zmianę tego, co jest generowane. Zamiast przewidywać tokeny obrazu jeden po drugim w przestrzeni, VAR przewiduje cały obraz w rosnących rozdzielczościach. Krok 1: przewidź token 1x1 („podsumowanie” całego obrazu). Krok 2: przewidź siatkę tokenów 2x2 (grubsze cechy). Krok 3: przewidź siatkę 4x4. Krok K: przewidź końcową siatkę (H/8)x(W/8).

Każda skala zwraca uwagę na wszystkie poprzednie skale (przyczynowo w „kolejności skal”) i równolegle w obrębie własnej skali. Problem kolejności znika: cały obraz w skali k jest produkowany w jednym przejściu transformera.

## Koncepcja

### Tokenizer Multi-Skali VQ-VAE

VAR potrzebuje **dyskretnego tokenizera multi-skali**. Dla obrazu x produkuje sekwencję siatek tokenów o stopniowo wyższej rozdzielczości:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Każda z_k używa tego samego kodeksu (typowy rozmiar 4096-16384). Tokenizacja w każdej skali nie jest niezależna — jest trenowana tak, aby sumowanie residuów w każdej skali rekonstruowało f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

Jest to wariant **residualnego VQ**. Skala k przechwytuje to, co skale 1..k-1 pominęły. Dekoder pobiera sumę embeddingów wszystkich skal i produkuje obraz.

Tokenizer VQ multi-skali jest trenowany raz (jak VQGAN), a następnie zamrażany. Cała praca generatywna jest wykonywana przez model autoregresywny na górze.

### Predykcja Następnej Skali

Model generatywny to transformer, który widzi tokeny ze wszystkich poprzednich skal i przewiduje tokeny w następnej skali.

Struktura sekwencji wejściowej:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Embeddingi pozycyjne kodują zarówno indeks skali, jak i pozycję przestrzenną w obrębie skali. Uwaga jest przyczynowa w kolejności skal: token w skali k, pozycji (i, j) może zwracać uwagę na wszystkie tokeny w skalach 1..k oraz na tokeny w skali k, które występują wcześniej w przyjętej kolejności wewnątrz-skali (VAR używa ustalonej uwagi pozycyjnej bez przyczynowości wewnątrz-skali — wszystkie pozycje w obrębie skali są przewidywane równolegle).

Funkcja straty treningowej: w każdej skali k przewidź tokeny z_k mając wszystkie tokeny z wcześniejszych skal. Strata cross-entropii na dyskretnych kodach VQ. Ta sama struktura co GPT, z tą różnicą, że „sekwencja” jest teraz zorganizowana w skale.

### Generacja

Podczas inferencji:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Dla K = 10 skal, generacja to 10 przejść transformera w przód. Każde przejście produkuje całą swoją skalę równolegle — brak autoregresji na poziomie pojedynczego tokena w obrębie skali. Dla obrazu 256x256 to około 10 przejść vs 28-50 dla DiT.

### Dlaczego Następna Skala Wygrywa z Następnym Tokenem

Trzy strukturalne zalety:
1. **Od ogółu do szczegółu współgra z naturalną statystyką obrazu.** Ludzka percepcja wzrokowa i zbiory danych obrazowych wykazują zależne od skali prawidłowości: niskoczęstotliwościowa struktura jest stabilna i przewidywalna; wysokoczęstotliwościowe detale są warunkowane zawartością niskich częstotliwości. Predykcja następnej skali to wykorzystuje.
2. **Równoległa generacja w obrębie skali.** W przeciwieństwie do tokenowego AR w stylu GPT, VAR produkuje wszystkie tokeny w skali w jednym kroku. Efektywna długość generacji jest logarytmiczna zamiast liniowej.
3. **Bias kolejności generowania znika.** Tokeny w skali k widzą całą skalę k-1; nie ma biasu „na lewo od” czy „powyżej”, który zmuszałby wczesne tokeny do decyzji zanim dostępny jest późniejszy kontekst.

### Prawo Skalowania

Tian i in. wykazali, że VAR podlega potęgowemu prawu skalowania dla FID na ImageNet — dokładnie tak jak GPT dla perplexity. Podwojenie parametrów lub obliczeń niezawodnie zmniejsza błąd o połowę. Był to pierwszy model generatywny obrazów, który wykazał tego rodzaju zachowanie skalowania tak czysto jak modele językowe. Rezultat jest taki, że przewidywania skal VAR stają się przewidywalne na podstawie obliczeń, a nie empirycznych zgadywań na architekturę.

### Relacja z Dyfuzją

VAR i dyfuzja dzielą tę samą historię kompresji danych: oba rozbijają problem generacji na sekwencję łatwiejszych podproblemów.

- Dyfuzja: stopniowo dodawaj szum, naucz się cofać jeden krok.
- VAR: stopniowo dodawaj rozdzielczość, naucz się przewidywać następną skalę.

Są to różne osie przez problem. Obie dają ułatwione rozkłady warunkowe. Empirycznie VAR jest szybszy przy inferencji (mniej przejść, wszystko równolegle w obrębie skali) i dorównuje lub przewyższa DiT na warunkowym ImageNet. Tekstowo-warunkowy VAR (VARclip, HART) to aktywny kierunek badań.

## Zbuduj To

W `code/main.py`:
1. Zbuduj mały **tokenizer VQ multi-skali** na syntetycznych danych „obrazowych” (2D pierścienie Gaussa).
2. Wytrenuj **transformer w stylu VAR** do przewidywania następnej skali tokenów.
3. Próbkuj, wywołując transformer 4 razy (4 skale) i dekodując.
4. Zweryfikuj, że trenowanie w kolejności skal umożliwia równoległą generację w obrębie skali.

Jest to implementacja zabawkowa. Celem jest zobaczenie, jak maska uwagi zorganizowana w skale i równoległa generacja w obrębie skali działają w praktyce.

## Dostarcz To

Ta lekcja produkuje `outputs/skill-var-tokenizer-designer.md` — umiejętność do projektowania tokenizera multi-skali: liczba skal, proporcje skal, rozmiar kodeksu, dzielenie residuów, architektura dekodera.

## Ćwiczenia

1. **Ablacja liczby skal.** Trenuj VAR z 4, 6, 8, 10 skalami. Zmierz jakość rekonstrukcji vs liczba przejść autoregresywnych. Więcej skal = drobniejsze residua = lepsza jakość, ale więcej przejść.

2. **Rozmiar kodeksu.** Trenuj tokenizery z rozmiarami kodeksu 512, 4096, 16384. Większe kodeksy dają lepszą rekonstrukcję, ale trudniejszą predykcję. Znajdź punkt przegięcia.

3. **Sprawdzenie równoległości w obrębie skali.** Dla wytrenowanego VAR zmierz jawnie wzorzec uwagi. W obrębie skali k, czy model zwraca uwagę na pozycje między-skalowe, ale nie wewnątrz-skali? Zweryfikuj implementację maski.

4. **Skalowanie VAR vs DiT.** Dla tego samego zadania warunkowego ImageNet, trenuj VAR i DiT przy dopasowanych budżetach parametrów (np. 33M, 130M, 458M). Wykreśl FID vs obliczenia. VAR powinien przewyższać DiT przy każdym rozmiarze — odtwórz wynik z artykułu w małej skali.

5. **Warunkowanie tekstem.** Rozszerz VAR o przyjmowanie embeddingu tekstowego (CLIP pooled) jako dodatkowego wejścia warunkującego przez adaLN. Jest to przepis HART. O ile poprawia się FID na próbkowaniu dopasowanym do tekstu?

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|--------|-----------------|----------------------|
| VAR | „Visual AutoRegressive” | Generacja obrazów przez predykcję następnej skali na piramidzie siatek tokenów VQ |
| Predykcja następnej skali | „Przewiduj grubsze, potem drobniejsze” | Model przewiduje tokeny w rosnących rozdzielczościach, warunkując się na wszystkich poprzednich skalach |
| Tokenizer VQ multi-skali | „Residualne VQ” | VQ-VAE, które produkuje K siatek tokenów o rosnącej rozdzielczości, z dekoderem sumującym wszystkie skale |
| Skala k | „Poziom piramidy k” | Jeden z K poziomów rozdzielczości, od 1x1 przy k=1 do (H/p)x(W/p) przy k=K |
| Równoległość w obrębie skali | „Jedno przejście w przód na skalę” | Wszystkie tokeny w skali k są przewidywane w jednym przejściu transformera, nie autoregresywnie |
| Przyczynowość między skalami | „Uwaga uporządkowana według skal” | Token w skali k może zwracać uwagę na wszystkie skale 1..k, ale nie na skale k+1..K |
| Residualne VQ | „Tokenizacja addytywna” | Tokeny każdej skali kodują residuum pozostawione przez niższe skale; dekoder sumuje embeddingi wszystkich skal |
| Prawo skalowania VAR | „Skalowanie Image GPT” | FID podlega przewidywalnemu prawu potęgowemu w funkcji obliczeń, jak perplexity w modelach językowych |
| HART | „Hybrydowy VAR + tekst” | Wariant VAR warunkowany tekstem, łączący iteracyjne dekodowanie w stylu MaskGIT ze strukturą skal VAR |
| Embedding pozycji w skali | „Trójka (skala, wiersz, kolumna)” | Kodowanie pozycyjne przenosi zarówno indeks skali, jak i współrzędne przestrzenne w obrębie skali |

## Dalsza Literatura

- [Tian et al., 2024 — „Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction”](https://arxiv.org/abs/2404.02905) — artykuł VAR, kanoniczne źródło
- [Peebles and Xie, 2022 — „Scalable Diffusion Models with Transformers”](https://arxiv.org/abs/2212.09748) — DiT, linia bazowa porównania dyfuzyjnego
- [Esser et al., 2021 — „Taming Transformers for High-Resolution Image Synthesis”](https://arxiv.org/abs/2012.09841) — VQGAN, rodzina tokenizerów, którą rozszerza tokenizer multi-skali VAR
- [van den Oord et al., 2017 — „Neural Discrete Representation Learning”](https://arxiv.org/abs/1711.00937) — VQ-VAE, podstawa dyskretnej tokenizacji obrazów
- [Tang et al., 2024 — „HART: Efficient Visual Generation with Hybrid Autoregressive Transformer”](https://arxiv.org/abs/2410.10812) — tekstowo-warunkowy VAR