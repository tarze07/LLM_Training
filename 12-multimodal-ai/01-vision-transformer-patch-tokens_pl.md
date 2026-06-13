# Transformery Wizyjne i Podstawowy Koncept Patch-Token

> Zanim cokolwiek stanie się multimodalne, obraz musi zostać przekształcony w sekwencję tokenów, którą transformer może przetworzyć. Artykuł ViT z 2020 roku odpowiedział na to pytanie za pomocą łat 16x16 pikseli, liniowej projekcji i osadzenia pozycyjnego. Pięć lat później każdy model graniczny z 2026 roku (Claude Opus 4.7 przy natywnym 2576px, Gemini 3.1 Pro, Qwen3.5-Omni) wciąż zaczyna w ten sposób — enkoder zmienił się z ViT na DINOv2 na SigLIP 2, dodano tokeny rejestrowe, schemat pozycyjny stał się 2D-RoPE, ale podstawowy koncept pozostał. Ta lekcja czyta potok patch-token od początku do końca i buduje go w standardowym Pythonie, aby reszta Fazy 12 miała konkretny model mentalny dla "tokenów wizyjnych."

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 minutes

## Learning Objectives

- Konwersja obrazu HxWx3 na sekwencję tokenów łat z poprawnym kodowaniem pozycyjnym.
- Obliczenie długości sekwencji, liczby parametrów i FLOPs dla ViT o zadanych (rozmiar łaty, rozdzielczość, ukryty wymiar, głębokość).
- Wymienienie trzech ulepszeń, które przeniosły ViT z badań w 2020 roku do produkcji w 2026: samonadzorowane pretrening (DINO / MAE), tokeny rejestrowe i pakowanie w natywnej rozdzielczości.
- Wybór między CLS pooling, mean pooling i tokenami rejestrowymi dla zadania docelowego.

## Problem

Transformery operują na sekwencjach wektorów. Tekst jest już sekwencją (bajtów lub tokenów). Obraz to 2D siatka pikseli z trzema kanałami kolorów — nie jest sekwencją. Jeśli spłaszczysz każdy piksel, obraz RGB 224x224 staje się 150 528 tokenami, a self-attention przy tej długości jest nie do wykonania (kwadratowy względem długości sekwencji).

Podejścia sprzed 2020 doklejały CNN ekstraktor cech na przedzie: ResNet produkuje mapę cech 7x7 wektorów o wymiarze 2048, podaj te 49 tokenów do transformera. Działa to, ale dziedziczy uprzedzenia CNN (ekwiwariancja translacyjna, lokalne pola recepcyjne) i traci transformerowy apetyt na skalę.

Dosovitskiy i in. (2020) zadali bezpośrednie pytanie: co jeśli pominiemy CNN? Podziel obraz na łaty o stałym rozmiarze (np. 16x16 pikseli), rzutuj liniowo każdą łatę na wektor, dodaj osadzenie pozycyjne i podaj sekwencję do waniliowego transformera. W tamtym czasie było to herezją — wizja bez konwolucji. Przy wystarczającej ilości danych (JFT-300M, później LAION) pokonało ResNet na ImageNet i stale się poprawiało.

Do 2026 roku ViT jest niekwestionowanym fundamentem. Wieża wizyjna każdego open-weights VLM to jakiś potomek (DINOv2, SigLIP 2, CLIP, EVA, InternViT). Pytanie nie brzmi już "czy używać łat?" ale "jaki rozmiar łaty, jaki harmonogram rozdzielczości, jaki cel pretreningu, jakie kodowanie pozycyjne."

## Koncepcja

### Łaty jako tokeny

Dla obrazu `x` o kształcie `(H, W, 3)` i rozmiarze łaty `P`, wycinasz obraz w siatkę `(H/P) x (W/P)` nienachodzących na siebie łat. Każda łata to sześcian pikseli `P x P x 3`. Spłaszcz każdy sześcian do wektora `3 P^2`. Zastosuj współdzieloną projekcję liniową `W_E` o kształcie `(3 P^2, D)`, aby odwzorować każdą łatę w ukryty wymiar modelu `D`.

Dla konfiguracji kanonicznej ViT-B/16:
- Rozdzielczość 224, rozmiar łaty 16 → siatka 14x14 → 196 tokenów łat.
- Każda łata to `16 x 16 x 3 = 768` wartości pikseli, rzutowanych na `D = 768`.
- Dodaj uczony token `[CLS]` → długość sekwencji 197.

Projekcja łaty jest matematycznie identyczna z konwolucją 2D z rozmiarem jądra `P`, krokiem `P` i `D` kanałami wyjściowymi. Tak to faktycznie implementuje kod produkcyjny — `nn.Conv2d(3, D, kernel_size=P, stride=P)`. Ramka "projekcji liniowej" jest koncepcyjna; ramka jądra jest wydajna.

### Osadzenia pozycyjne

Łaty nie mają inherentnego porządku — transformer widzi je jako worek. Wczesne ViT dodawały uczone 1D osadzenie pozycyjne (jeden wektor 768-wymiarowy na pozycję, 197 sztuk). Działa, ale przywiązuje model do rozdzielczości treningowej: przy inferencji trzeba interpolować tabelę pozycji, jeśli zmienisz siatkę.

Nowoczesne magazyny wizyjne używają 2D-RoPE (M-RoPE Qwen2-VL, domyślny SigLIP 2) lub sfaktoryzowanych pozycji 2D. 2D-RoPE obraca wektory zapytania i klucza na podstawie indeksu (wiersz, kolumna) łaty, dzięki czemu model wnioskuje względną pozycję 2D z kąta obrotu. Brak tabeli pozycji. Model obsługuje dowolne rozmiary siatki przy inferencji.

### Token CLS, wyjście uśrednione i tokeny rejestrowe

Jaka jest reprezentacja na poziomie obrazu? Trzy opcje współistnieją:

1. Token `[CLS]`. Poprzedź uczony wektor ciągowi łat. Po wszystkich blokach transformera stan ukryty tokena CLS jest reprezentacją obrazu. Odziedziczone z BERT. Używane przez oryginalny ViT, CLIP.
2. Uśrednianie (mean pool). Uśrednij wyjściowe stany ukryte tokenów łat. Używane przez SigLIP, DINOv2, większość nowoczesnych VLM.
3. Tokeny rejestrowe. Darcet i in. (2023) zaobserwowali, że ViT trenowane bez jawnego tokena sink rozwijają łatki "artefaktów" o wysokiej normie, które przechwytują self-attention. Dodanie 4–16 uczonych tokenów rejestrowych absorbuje to obciążenie i poprawia jakość gęstej predykcji (segmentacja, głębokość). Zarówno DINOv2, jak i SigLIP 2 są dostarczane z rejestrami.

Wybór ma znaczenie dla zadań docelowych. CLS jest w porządku dla klasyfikacji. Dla VLM, które podają tokeny łat do LLM, pomijasz pooling całkowicie — każda łata staje się tokenem wejściowym LLM. Rejestry są odrzucane przed przekazaniem (są rusztowaniem, nie treścią).

### Pretrening: nadzorowany, kontrastowy, maskowany, samodystylowany

ViT z 2020 roku był pretrenowany z nadzorowaną klasyfikacją na JFT-300M. Szybko zastąpiony przez:

- CLIP (2021): kontrastowy obraz-tekst na 400M parach. Lekcja 12.02.
- MAE (2021, He i in.): zamaskuj 75% łat, rekonstruuj piksele. Samonadzorowany, działa na czystych obrazach.
- DINO (2021) / DINOv2 (2023): samodystylacja z student-nauczyciel, bez etykiet, bez podpisów. DINOv2 ViT-g/14 z 2023 roku jest najmocniejszym czysto-wizyjnym magazynem i domyślnym wyborem dla przypadków użycia "gęstych cech".
- SigLIP / SigLIP 2 (2023, 2025): CLIP z sigmoidalną stratą i NaFlex dla natywnego współczynnika kształtu. Dominująca wieża wizyjna w otwartych VLM w 2026 roku (Qwen, Idefics2, LLaVA-OneVision).

Twój wybór pretreningu określa, do czego magazyn jest dobry: CLIP/SigLIP do semantycznego dopasowania z tekstem, DINOv2 do gęstych cech wizyjnych, MAE jako punkt startowy do docelowego dostrajania.

### Prawa skalowania

Skalowanie ViT (Zhai i in. 2022) ustaliło, że jakość ViT podlega przewidywalnym prawom dotyczącym rozmiaru modelu, rozmiaru danych i mocy obliczeniowej. Przy stałej mocy obliczeniowej:
- Większy model + więcej danych → lepsza jakość.
- Rozmiar łaty to dźwignia na długość sekwencji kontra wierność. Łata 14 (typowa dla DINOv2/SigLIP SO400m) daje więcej tokenów na obraz niż łata 16; lepsza dla OCR i zadań gęstych, gorsza dla szybkości.
- Rozdzielczość to druga duża dźwignia. Przejście z 224 na 384 na 512 prawie zawsze pomaga, przy kwadratowym koszcie FLOPs.

ViT-g/14 (1B parametrów, łata 14, rozdzielczość 224 → 256 tokenów) i SigLIP SO400m/14 (400M parametrów, łata 14) to dwa podstawowe enkodery dla otwartych VLM w 2026 roku.

### Liczba parametrów dla ViT

Pełne obliczenia znajdują się w `code/main.py`. Dla ViT-B/16 przy 224:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Oszacuj każdy ViT w ten sposób przed załadowaniem punktu kontrolnego. Rozmiar magazynu określa minimalne VRAM w każdym docelowym VLM.

### Konfiguracja produkcyjna 2026

Enkoder, z którym większość otwartych VLM jest dostarczana w 2026 roku, to SigLIP 2 SO400m/14 w natywnej rozdzielczości (NaFlex). Ma:
- 400M parametrów.
- Rozmiar łaty 14, domyślna rozdzielczość 384 → 729 tokenów łat na obraz.
- Uśrednianie (mean pool) dla zadań na poziomie obrazu; wszystkie 729 łat trafia do LLM dla VQA.
- 4 tokeny rejestrowe, odrzucane przed przekazaniem do LLM.
- 2D-RoPE ze skalowaniem na poziomie obrazu dla natywnego współczynnika kształtu.

Każda decyzja w tej konfiguracji prowadzi do artykułu naukowego, który możesz przeczytać.

```figure
image-patch-tokens
```

## Użyj Tego

`code/main.py` to tokenizer łat i kalkulator geometrii. Przyjmuje (wysokość obrazu H, szerokość W, rozmiar łaty P, ukryty wymiar D, głębokość L) i raportuje:

- Kształt siatki i długość sekwencji po podziale na łaty.
- Sekwencję tokenów dla syntetycznego obrazu zabawkowego 8x8 pikseli (przejdź przez ścieżkę spłaszcz + projektuj).
- Liczbę parametrów w podziale na osadzenie łat, osadzenie pozycji, bloki transformera i głowę.
- FLOPs na przejście w przód przy docelowej rozdzielczości.
- Tabelę porównawczą dla ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, SigLIP SO400m/14 @ 384.

Uruchom to. Dopasuj liczby parametrów do opublikowanych wartości. Pobaw się rozmiarem łaty i rozdzielczością, aby poczuć koszt liczby tokenów.

## Dostarcz

Ta lekcja produkuje `outputs/skill-patch-geometry-reader.md`. Dla konfiguracji ViT (rozmiar łaty, rozdzielczość, ukryty wymiar, głębokość) produkuje oszacowanie liczby tokenów, liczby parametrów i VRAM z uzasadnieniami. Użyj tej umiejętności, gdy wybierasz magazyn wizyjny dla VLM — zapobiega niespodziankom typu "tokeny eksplodowały i mój kontekst LLM się zapełnił."

## Ćwiczenia

1. Oblicz długość sekwencji tokenów łat dla Qwen2.5-VL przy natywnym wejściu 1280x720 z rozmiarem łaty 14. Jak to się porównuje do reprezentacji tylko CLS?

2. Klatka 1080p (1920x1080) przy łacie 14 produkuje ile tokenów? Przy 30 FPS przez 5-minutowy film, ile wynosi całkowita liczba tokenów wizyjnych? Który koszt oszczędza najwięcej: pooling, próbkowanie klatek, czy scalanie tokenów?

3. Zaimplementuj mean pooling po tokenach łat w czystym Pythonie. Zweryfikuj, że mean pool po 196 tokenach wyjścia DINOv2 zgadza się z tym, co zwraca `forward` modelu, gdy poprosisz o osadzenie uśrednione.

4. Przeczytaj Sekcję 3 artykułu "Vision Transformers Need Registers" (arXiv:2309.16588). Opisz w dwóch zdaniach, jaki artefakt absorbują rejestry i dlaczego ma to znaczenie dla docelowej gęstej predykcji.

5. Zmodyfikuj `code/main.py`, aby obsługiwał patch-n'-pack: dla listy obrazów o różnych rozdzielczościach, wyprodukuj pojedynczą spakowaną sekwencję i blokowo-diagonalną maskę attention. Zweryfikuj względem Lekcji 12.06, gdy do niej dotrzesz.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Łata (Patch) | "Kwadrat 16x16 pikseli" | Niezachodzący na siebie region wejściowego obrazu o stałym rozmiarze; staje się jednym tokenem |
| Osadzenie łaty (Patch embedding) | "Projekcja liniowa" | Współdzielona uczona macierz (lub Conv2d z stride=P) mapująca spłaszczone piksele łaty do wektorów D-wymiarowych |
| Token CLS | "Token klasy" | Poprzedzający uczony wektor, którego końcowy stan ukryty reprezentuje cały obraz; opcjonalny w 2026 |
| Token rejestrowy (Register token) | "Token sink" | Dodatkowe uczone tokeny, które absorbują artefakty attention o wysokiej normie, które ViT rozwijają podczas pretreningu |
| Osadzenie pozycyjne (Position embedding) | "Informacja pozycyjna" | Wektor lub obrót na pozycję, który uświadamia modelowi porządek sekwencji; 2D-RoPE jest nowoczesnym domyślnym wyborem |
| Siatka (Grid) | "Siatka łat" | Tablica 2D (H/P) x (W/P) łat dla danej rozdzielczości i rozmiaru łaty |
| NaFlex | "Natywna elastyczna rozdzielczość" | Cecha SigLIP 2: pojedynczy model obsługuje wiele współczynników kształtu i rozdzielczości bez ponownego trenowania |
| Magazyn (Backbone) | "Wieża wizyjna" | Wstępnie wytrenowany enkoder obrazu, którego wyjścia tokenów łat trafiają do LLM w VLM |
| Pooling | "Podsumowanie poziomu obrazu" | Strategia przekształcania tokenów łat w jeden wektor: CLS, średnia, attention pool lub oparty na rejestrach |
| Łata 14 vs 16 | "Drobniejsza vs grubsza siatka" | Łata 14 produkuje więcej tokenów na obraz, lepszą wierność dla OCR, wolniej; łata 16 to klasyczny domyślny wybór |

## Dalsza Lektura

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — oryginalny ViT.
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE, samonadzorowany pretrening.
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — samodystylacja na skalę, bez etykiet.
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — tokeny rejestrowe i analiza artefaktów.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — domyślna wieża wizyjna 2026 roku.
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — empiryczne prawa skalowania.