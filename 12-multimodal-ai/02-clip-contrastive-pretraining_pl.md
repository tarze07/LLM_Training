# CLIP i Kontrastywne Pretrening Wizyjno-Językowy

> CLIP od OpenAI (2021) udowodnił, że jeden pomysł wystarczy, by napędzać kolejne pięć lat: wyrównaj enkoder obrazu i enkoder tekstu w tej samej przestrzeni wektorowej, używając tylko hałaśliwych par obraz-podpis z internetu i kontrastywnej funkcji straty. Zero nadzorowanych etykiet. 400M par. Powstała przestrzeń osadzeń robi klasyfikację zero-shot, wyszukiwanie obraz-tekst i podłącza się do każdego VLM w 2026 roku jako jego wieża wizyjna. SigLIP 2 (2025) zastąpił softmax sigmoidą i wyprzedził CLIP przy niższym koszcie. Ta lekcja przeprowadza przez matematykę od InfoNCE do sigmoidalnej straty parami i buduje krok treningowy w standardowym Pythonie.

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 minutes

## Learning Objectives

- Wyprowadzenie straty InfoNCE z informacji wzajemnej i implementacja numerycznie stabilnej, zwektoryzowanej wersji.
- Wyjaśnienie, dlaczego sigmoidalna strata parami (SigLIP) skaluje się do batcha 32768+ bez narzutu all-gather, którego wymaga softmax.
- Przeprowadzenie klasyfikacji zero-shot na ImageNet przez konstruowanie szablonów tekstowych (`a photo of a {class}`) i branie argmax po podobieństwie cosinusowym.
- Wymienienie czterech dźwigni, które daje pretrening CLIP / SigLIP: rozmiar batcha, temperatura, szablon promptu, jakość danych.

## Problem

Przed CLIP-em wizja była nadzorowana. Zbieraj oznaczone zbiory danych (ImageNet: 1.2M obrazów, 1000 klas), trenuj CNN, wysyłaj. Etykiety są drogie, etykiety są obciążone tym, na co etykietujący mogą się zgodzić, a etykiety nie przenoszą się na nowe zadania bez dostrajania.

Obrazowo-podpisowa sieć ma miliard plus luźnie oznaczonych par za darmo. Zdjęcie golden retrievera z tekstem alternatywnym "mój pies Max w parku" niesie sygnał nadzorczy — tekst opisuje obraz. Pytanie: czy można to zamienić w użyteczne trenowanie?

Odpowiedź CLIP-a: traktuj pary obraz-podpis jako zadanie dopasowania. Dla batcha N obrazów i N podpisów, naucz się dopasowywać każdy obraz do własnego podpisu przeciwko N-1 dystraktorom. Nadzór brzmi: "te dwie rzeczy należą do siebie; te N-1 nie." Żadnych etykiet klas. Żadnej adnotacji ludzkiej. Tylko kontrastywna funkcja straty.

Powstała przestrzeń osadzeń robi więcej, niż CLIP był trenowany. ImageNet zero-shot działa, ponieważ "a photo of a cat" osadza się blisko zdjęć kotów, które nigdy nie były jawnie oznaczone jako koty. To był zakład, który zapoczątkował każdy VLM w 2026 roku.

## Koncepcja

### Podwójny enkoder

CLIP ma dwie wieże:

- Enkoder obrazu `f`: ViT lub ResNet, wyprowadza D-wymiarowy wektor na obraz.
- Enkoder tekstu `g`: mały transformer, wyprowadza D-wymiarowy wektor na podpis.

Obie wieże normalizują swoje wyjścia do jednostkowej długości. Podobieństwo to `cos(f(x), g(y)) = f(x)^T g(y)`, ponieważ oba są jednostkowo-normowane.

Dla batcha N par (obraz, podpis), zbuduj macierz podobieństwa `S` o kształcie `(N, N)`:

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

gdzie `tau` to uczona temperatura (CLIP inicjalizuje na 0.07; uczona w przestrzeni logarytmicznej).

### Strata InfoNCE

CLIP używa symetrycznej entropii krzyżowej po wierszach i kolumnach:

```
loss_i2t = CE(S, labels=identity)     # pozytywem każdego obrazu jest jego własny podpis
loss_t2i = CE(S^T, labels=identity)   # pozytywem każdego podpisu jest jego własny obraz
loss = (loss_i2t + loss_t2i) / 2
```

To jest InfoNCE. Softmax w CE wymusza, aby każdy obraz pasował do swojego podpisu bardziej niż do każdego innego podpisu w batchu. "Negatywy" to wszystkie inne elementy batcha. Większe batche = więcej negatywów = silniejszy sygnał. CLIP trenował przy batchu 32k; skala ma znaczenie.

### Temperatura

`tau` kontroluje ostrość softmaxa. Niskie tau → ostra dystrybucja, efekt hard negative mining. Wysokie tau → miękka, wszystkie próbki się przyczyniają. CLIP uczy log(1/tau), przycięte, aby zapobiec kolapsowi. SigLIP 2 ustala początkową tau i zamiast tego używa uczonego biasu.

### Dlaczego sigmoid skaluje się lepiej (SigLIP)

Softmax potrzebuje całej macierzy podobieństwa zsynchronizowanej. W treningu rozproszonym musisz all-gather każde osadzenie do każdej repliki, a następnie wykonać softmax. Jest to kwadratowe względem rozmiaru świata dla komunikacji.

SigLIP zastępuje softmax elementarną sigmoidą: dla każdej pary `(i, j)`, strata to binarna klasyfikacja "czy to jest pasująca para?" pozytywne etykiety klas to diagonala, wszystko inne to negatyw. Strata wynosi:

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` jeśli `i == j`, w przeciwnym razie 0. Strata każdej pary jest niezależna. All-gather nie jest potrzebne. Każdy GPU oblicza swój lokalny blok i sumuje. SigLIP 2 skaluje się do batcha 32k-512k tanio, podczas gdy CLIP potrzebowałby proporcjonalnie więcej komunikacji.

### Klasyfikacja zero-shot

Mając N nazw klas, dla każdej klasy zbuduj szablon tekstowy:

```
"a photo of a {class}"
```

Osadź każdy szablon za pomocą enkodera tekstu. Osadź swój obraz za pomocą enkodera obrazu. Argmax podobieństwa cosinusowego = przewidywana klasa. Żadnego trenowania na docelowych klasach.

Szablony promptów mają znaczenie. Oryginalny artykuł CLIP używał 80 szablonów na klasę (zwykły, artystyczny, zdjęcie, malarstwo itp.) i uśredniał osadzenia. +3 punkty na ImageNet. Nowoczesne użycie zazwyczaj wybiera jeden lub dwa szablony.

### Proby liniowe i dostrajanie

Zero-shot to linia bazowa. Proba liniowa (trenuj jedną warstwę liniową na zamrożonych cechach CLIP dla twoich docelowych klas) bije zero-shot w zadaniach wewnątrz domeny. Pełne dostrajanie bije probę liniową wewnątrz domeny, ale może zaszkodzić transferowi zero-shot. Trzy reżimy z trzema kompromisami.

### SigLIP 2: NaFlex i gęste cechy

SigLIP 2 (2025) dodaje:
- NaFlex: pojedynczy model obsługuje zmienne współczynniki kształtu i rozdzielczości.
- Lepsze gęste cechy do segmentacji i estymacji głębokości, celując w użycie jako zamrożony magazyn w VLM.
- Wielojęzyczność: trenowany na 100+ językach, podczas gdy CLIP był tylko angielski.
- Skala 1B parametrów, podczas gdy CLIP osiągnął maksimum przy 400M.

W otwartych VLM w 2026 roku SigLIP 2 SO400m/14 jest domyślną wieżą wizyjną. CLIP pozostaje domyślnym wyborem dla czystego wyszukiwania obraz-tekst, gdzie specyficzna dystrybucja treningowa LAION-2B pasuje do twojego wzorca zapytań.

### ALIGN, BASIC, OpenCLIP, EVA-CLIP

ALIGN (Google, 2021): ten sam pomysł co CLIP, skala 1.8B par, 90% hałaśliwych. Udowodnił, że hałaśliwe dane skalują się. OpenCLIP (LAION): otwarta reprodukcja CLIP na LAION-400M / 2B, wiele skal, punkt odniesienia dla otwartych punktów kontrolnych. EVA-CLIP: inicjalizowany z maskowanego modelowania obrazów; silny magazyn dla VLM. BASIC: hybryda CLIP+ALIGN od Google. Wszystkie z tej samej rodziny, różne dane i strojenie.

### Sufit zero-shot

Modele klasy CLIP osiągają około 76% ImageNet zero-shot (CLIP-G, OpenCLIP-G). Powyżej wymaga albo znacznie większych danych (SigLIP 2 osiąga 80%+), albo zmian architektonicznych (nadzorowane głowy, więcej parametrów). Benchmark się nasyca; prawdziwa wartość to przestrzeń osadzeń, którą konsumują docelowe VLM.

```figure
multimodal-fusion
```

## Użyj Tego

`code/main.py` implementuje:

1. Zabawkowy podwójny enkoder (cechy obrazu oparte na haszowaniu, cechy znaków tekstowych), abyś mógł zobaczyć kształt InfoNCE bez numpy.
2. Stratę InfoNCE w czystym Pythonie (stabilność numeryczna przez log-sum-exp).
3. Sigmoidalną stratę parami do porównania.
4. Procedurę klasyfikacji zero-shot: oblicz podobieństwo cosinusowe względem zestawu promptów tekstowych, argmax dla predykcji.

Uruchom to i obserwuj krzywą straty. Bezwzględne liczby są zabawkowe; kształt pasuje do tego, co emituje prawdziwy trener CLIP.

## Dostarcz

Ta lekcja produkuje `outputs/skill-clip-zero-shot.md`. Dla zestawu obrazów (przez ścieżkę) i listy docelowych klas, buduje prompty tekstowe z szablonem CLIP, osadza obie strony z podanym punktem kontrolnym (np. `openai/clip-vit-large-patch14`) i zwraca predykcje top-1 / top-5 z wynikami podobieństwa. Umiejętność odmawia twierdzeń o klasach, których nie ma na liście promptów.

## Ćwiczenia

1. Zaimplementuj InfoNCE ręcznie dla batcha 4 par. Skonstruuj macierz podobieństwa 4x4, uruchom softmax, wybierz diagonalę, oblicz entropię krzyżową. Zweryfikuj swoją implementację w Pythonie względem tego ręcznego obliczenia.

2. SigLIP używa parametru biasu `b` oprócz temperatury: `S'[i,j] = S[i,j]/tau + b`. Jaką rolę odgrywa `b`, gdy batch ma dużą nierównowagę klas (znacznie więcej negatywów niż pozytywów na wiersz)? Przeczytaj SigLIP Sekcję 3 (arXiv:2303.15343).

3. Zbuduj klasyfikator zero-shot dla kotów vs psów. Wypróbuj dwa szablony promptów: `a photo of a {class}` i `a picture of a {class}`. Zmierz dokładność na 100 obrazach testowych. Czy ensemble szablonów bije pojedynczy?

4. Oblicz koszt komunikacji softmax InfoNCE vs sigmoidalnej straty parami dla uruchomienia na 512 GPU przy batchu 32k. Który skaluje się jako O(N), a który jako O(N^2)? Zacytuj SigLIP Sekcję 4.

5. Przeczytaj artykuł o prawach skalowania OpenCLIP (arXiv:2212.07143, Cherti i in.). Odtwórz ich wniosek dotyczący skalowania danych z wykresów: przy stałym rozmiarze modelu, jaka jest log-liniowa zależność między dokładnością ImageNet zero-shot a rozmiarem danych treningowych?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| InfoNCE | "Strata kontrastywna" | Entropia krzyżowa po macierzy podobieństwa batcha; pozytywem każdego elementu jest jego sparowany element, negatywami wszystko inne |
| Strata sigmoidalna | "Strata SigLIP" | Binarna entropia krzyżowa na parę; brak softmaxa, brak all-gather, skaluje się tanio w treningu rozproszonym |
| Temperatura | "tau" | Skalar skalujący logity przed softmax/sigmoidą; kontroluje ostrość dystrybucji |
| Zero-shot | "klasyfikacja bez dostrajania" | Użyj promptów tekstowych do skonstruowania osadzeń klas i klasyfikuj przez podobieństwo cosinusowe; żadnego trenowania na docelowych klasach |
| Szablon promptu | "a photo of a ..." | Tekstowe rusztowanie wokół nazwy klasy; wpływa na dokładność zero-shot o 1-5 punktów |
| Podwójny enkoder | "Dwie wieże" | Jeden enkoder obrazu + jeden enkoder tekstu, wyjścia we współdzielonej przestrzeni D-wymiarowej |
| Hard negative | "Trudny dystraktor" | Negatyw na tyle podobny do pozytywu, że model musi pracować, aby je rozdzielić |
| Proba liniowa | "Zamrożony + jedna warstwa" | Trenuj tylko klasyfikator liniowy na zamrożonych cechach; mierzy jakość cech |
| NaFlex | "Natywna elastyczna rozdzielczość" | Możliwość SigLIP 2 do przyjmowania obrazów w dowolnym współczynniku kształtu i rozdzielczości bez zmiany rozmiaru |
| Skalowanie temperatury | "log-parametryzowana tau" | CLIP parametryzuje `log(1/tau)`, aby gradienty działały; przycina, aby zapobiec kolapsowi do bliskiej zera tau |

## Dalsza Lektura

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — artykuł CLIP.
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — wielojęzyczność + NaFlex.
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — skala z hałaśliwymi danymi internetowymi.
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — prawa skalowania OpenCLIP.