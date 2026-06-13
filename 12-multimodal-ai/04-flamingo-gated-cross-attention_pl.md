# Flamingo i Gated Cross-Attention dla Few-Shot VLM

> Flamingo od DeepMind (2022) zrobił dwie rzeczy przed innymi. Pokazał, że pojedynczy model może przetwarzać dowolnie przeplatane sekwencje obrazów, wideo i tekstu. I pokazał, że VLM mogą uczyć się in-context — podaj kilku-shotowy prompt z trzema przykładami (obraz, podpis), a model opisuje nowy obraz bez żadnego kroku gradientowego. Mechanizm: gated cross-attention, wstawione między istniejące warstwy zamrożonego LLM, z uczoną bramą tanh, która zaczyna się od zera, dzięki czemu zdolność tekstowa LLM jest zachowana przy inicjalizacji. Ta lekcja czyta architekturę Perceiver resampler i gated cross-attention Flamingo — przodka przeplatanych wejść Gemini i tokenów wizyjnych Idefics2.

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120 minutes

## Learning Objectives

- Wyjaśnienie, jak gated cross-attention zachowuje zdolność tekstową zamrożonego LLM przy inicjalizacji przez tanh(brama) = 0.
- Przejście przez Perceiver resampler: N łat obrazu → K stałych "latentnych" zapytań przez cross-attention.
- Opisanie, jak Flamingo obsługuje przeplatane sekwencje obraz-tekst z przyczynowym maskowaniem uwzględniającym położenie obrazu.
- Odtworzenie struktury multimodalnego promptu few-shot (3 przykłady obraz-podpis, a następnie obraz zapytania).

## Problem

BLIP-2 podaje 32 tokeny wizyjne do warstwy wejściowej zamrożonego LLM. Działa dla jednego obrazu na prompt. Ale co, jeśli chcesz podać *wiele* obrazów przeplatanych z tekstem, jak w "oto obraz A, opisz go; oto obraz B, opisz go; teraz oto obraz C, opisz go"? Self-attention LLM musiałby obsługiwać tokeny obrazu i tokeny tekstu w jednym strumieniu, a pytanie, które pozycje mogą uwzględniać które obrazy, staje się skomplikowane.

Odpowiedź Flamingo: w ogóle nie zmieniaj strumienia wejściowego LLM. Wstaw dodatkowe warstwy cross-attention między istniejące bloki LLM. Tokeny tekstu nadal przepływają przez przyczynowy self-attention LLM jak zwykle. Między co kilka bloków LLM tokeny tekstu wykonują również cross-attention na cechach obrazu przez nową warstwę z bramą. Brama (zainicjalizowana na zero) oznacza, że w kroku zero nowe warstwy są no-op — model zachowuje się dokładnie jak wstępnie wytrenowany LLM. W miarę postępu treningu brama się otwiera, a informacje wizyjne zaczynają płynąć.

Drugie pytanie, na które odpowiedział Flamingo: jak obsłużyć zmienną liczbę obrazów (0, 1 lub wiele) na prompt? Perceiver resampler — mały moduł cross-attention, który bierze dowolną liczbę łat i produkuje stałą liczbę wizyjnych tokenów latentnych. Warstwa cross-attention LLM widzi ten sam kształt niezależnie od liczby obrazów w prompcie.

## Koncepcja

### Zamrożony LLM

Flamingo zaczyna od zamrożonego LLM Chinchilla 70B. Wszystkie 70B wag nietknięte. Istniejący self-attention tekstu i FFN działają normalnie.

### Perceiver resampler

Dla każdego obrazu w prompcie ViT produkuje N tokenów łat. Perceiver resampler ma K stałych uczonych latentów (Flamingo używa K=64). Każdy blok resamplera to dwa pod-kroki:

1. Cross-attention: K latentów uwzględnia N tokenów łat (Q z latentów, K/V z łat).
2. Self-attention + FFN wewnątrz latentów.

Po 6 blokach resamplera wyjściem jest K=64 tokenów wizyjnych o wymiarze 1024, niezależnie od liczby łat wyprodukowanych przez ViT. Obraz 224x224 (196 łat) i obraz 480x480 (900 łat) oba wychodzą jako 64 tokeny resamplera.

Dla wideo resampler jest stosowany temporalnie: łaty każdej klatki produkują 64 latenty, a temporalne kodowanie pozycyjne pozwala modelowi odróżnić t=0 od t=N. Pełne wideo staje się T * 64 tokenów wizyjnych.

### Gated cross-attention

Między co M warstw zamrożonego LLM (Flamingo używa M=4), wstaw nowy blok gated cross-attention:

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` to uczony skalar zainicjalizowany na zero.
- `tanh(0) = 0`, więc przy inicjalizacji gałąź z bramą wnosi zero.
- Gdy `alpha` oddala się od zera, wkład cross-attention rośnie płynnie.
- Połączenie resztkowe oznacza, że nawet w pełni otwarta brama nie nadpisuje reprezentacji tekstowej LLM; po prostu dodaje informacje wizyjne na wierzch.

To jest najważniejsza decyzja projektowa w Flamingo: warunkowanie wizyjne jest addytywne, bramkowane i zerowe przy inicjalizacji. Flamingo w kroku 0 jest idealnym Chinchilla 70B na wejściach czysto tekstowych.

### Maskowany cross-attention dla przeplatanych wejść

W prompcie takim jak "<image A> caption A <image B> caption B <image C> ?", każdy token tekstu powinien widzieć tylko obrazy, które pojawiły się przed nim w sekwencji. Maska cross-attention wymusza: token tekstu na pozycji `t` uwzględnia tylko tokeny resamplera obrazu, których indeks obrazu `i < i_t`, gdzie `i_t` to ostatni obraz przed pozycją `t`. "Widzi tylko ostatni poprzedzający obraz" lub "widzi wszystkie poprzedzające obrazy" to oba poprawne wybory; Flamingo wybrał to pierwsze.

### In-context few-shot learning

Prompt Flamingo wygląda tak:

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

Model widzi wzorzec uzupełniania i wypisuje "bird" (lub cokolwiek pokazuje image3). Żadnych kroków gradientowych. Zdolność uczenia się in-context zamrożonego LLM przenosi się przez gated cross-attention — to jest sedno artykułu i dlaczego ma to znaczenie.

### Dane treningowe

Flamingo trenował na trzech zbiorach danych:

1. MultiModal MassiveWeb (M3W): 43M stron internetowych z przeplatanymi obrazami i tekstem, rekonstruujących kolejność czytania.
2. Pary obraz-tekst (ALIGN + LTIP): 4.4B par.
3. Pary wideo-tekst (VTP): 27M krótkich klipów wideo.

OBELICS (2023) to otwarta reprodukcja przeplatanego korpusu internetowego, na którym trenują Idefics, Idefics2 i większość otwartych modeli "podobnych do Flamingo."

### OpenFlamingo i Otter

OpenFlamingo (2023) to otwarta reprodukcja. Architektura identyczna (Perceiver resampler + gated cross-attention na zamrożonym LLaMA lub MPT). Punkty kontrolne przy 3B, 4B, 9B. Jakość odstaje od Flamingo z powodu mniejszego bazowego LLM i mniejszej ilości danych.

Otter (2023) buduje na OpenFlamingo z dostrajaniem instrukcji na MIMIC-IT (zbiór danych multimodalnych instrukcji), pokazując, że gated cross-attention działa również dla podążania za instrukcjami.

### Potomkowie

- Idefics / Idefics2 / Idefics3: linia rodowa Hugging Face z gated cross-attention, stopniowo prostsza (Idefics2 porzucił resampler na rzecz bezpośrednich tokenów łat z adaptacyjnym poolingiem).
- Przejście Flamingo-to-Chameleon: do 2024 roku wiele zespołów przeszło na early-fusion (Lekcja 12.11); gated cross-attention w stylu Flamingo pozostaje w produkcji, gdzie wymagane jest zamrażanie magazynu.
- Przeplatane wejścia Gemini: koncepcyjnie dziedziczy elastyczność przeplatanego formatu Flamingo, chociaż dokładny mechanizm jest własnościowy.

### Porównanie z BLIP-2

| | BLIP-2 | Flamingo |
|---|---|---|
| Most wizyjny | Q-Former raz na wejściu | Gated cross-attention przy co M warstwach |
| Tokeny wizyjne | 32 na obraz | 64 na obraz na warstwę cross-attn |
| Zamrożony LLM | Tak | Tak |
| Few-shot in-context | Słaby | Silny — centralny punkt artykułu |
| Przeplatane wejścia | Brak natywnego wsparcia | Tak, cel projektowy |
| Dane treningowe | 130M par | 1.3B par + 43M przeplatanych stron |
| Liczba parametrów | 188M trenowane | ~10B trenowane (warstwy cross-attn) |
| Moc obliczeniowa | Dni na 8 A100 | Tygodnie na tysiącach TPUv4 |

Wybierz BLIP-2 dla jednego obrazu VQA przy ograniczonym budżecie. Wybierz Flamingo/Idefics2 dla przeplatanych, few-shot lub wieloobrazowych zadań rozumowania.

## Użyj Tego

`code/main.py` demonstruje:

1. Perceiver resampler na 36 sztucznych tokenach łat z 8 uczonymi latentami (czysty Python cross-attention).
2. Krok gated cross-attention z `alpha = 0` → wyjście równe wejściu (LLM niezmieniony), następnie `alpha = 2.0` → wkład wizyjny zmieszany.
3. Konstruktor maski przeplatanej, który produkuje 2D maskę attention dla sekwencji "(image 1) (text 1) (image 2) (text 2)".

## Dostarcz

Ta lekcja produkuje `outputs/skill-gated-bridge-diagnostic.md`. Dla konfiguracji otwartego VLM (resampler T/N, częstotliwość cross-attn, schemat bramy), identyfikuje elementy linii rodowej Flamingo i wyjaśnia strategię zamrażania. Przydatne do debugowania, dlaczego dostrajanie pogorszyło wydajność tekstową (odpowiedź: brama otworzyła się zbyt szybko).

## Ćwiczenia

1. Oblicz liczbę parametrów wizyjnych Flamingo-9B: 9B LLM + 1.4B gated cross-attention + 64M resampler. Jaka frakcja wszystkich parametrów jest trenowana?

2. Zaimplementuj gated residual `y = tanh(alpha) * cross + x` w PyTorch. Pokaż eksperymentalnie, że przy `alpha=0`, `y==x` dokładnie przy inicjalizacji.

3. Przeczytaj OpenFlamingo Sekcję 3.2 (arXiv:2308.01390) o tym, jak obsługują wiele obrazów w batchu, gdy każdy prompt ma inną liczbę obrazów. Opisz strategię dopełniania.

4. Dlaczego maska cross-attention Flamingo pozwala tokenowi tekstu uwzględniać *tylko ostatni* poprzedzający obraz, a nie wszystkie poprzedzające obrazy? Przeczytaj artykuł Flamingo Sekcję 2.4 i wyjaśnij kompromis.

5. In-context few-shot: skonstruuj prompt z 4 przykładami "obraz → kolor głównego obiektu" dla nowego wariantu Flamingo. Opisz oczekiwany wzorzec dokładności przy zmienianiu liczby przykładów od 0 do 8.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Perceiver resampler | "Cross-attention stałych latentów" | Moduł produkujący K stałych tokenów ze zmiennej liczby łat wejściowych |
| Gated cross-attention | "Most z bramą tanh" | Warstwa resztkowa `y = tanh(alpha)*cross + x`, uczone alpha, init 0 |
| Przeplatane wejście | "Mieszana sekwencja" | Format promptu z obrazami i tekstem swobodnie wymieszanymi w kolejności czytania |
| Zamrożony LLM | "Brak gradientów LLM" | Wagi tekstowego LLM nie aktualizują się; tylko resampler + warstwy cross-attn się trenują |
| Few-shot | "Przykłady in-context" | Podaj kilka par (obraz, odpowiedź) w prompcie; model generalizuje bez dostrajania |
| OBELICS | "Przeplatany korpus internetowy" | Otwarty zbiór 141M stron internetowych z obrazami i tekstem w kolejności czytania |
| Chinchilla | "70B zamrożona baza" | Zamrożony tekstowy LLM Flamingo, z artykułu Chinchilla DeepMind |
| Harmonogram bramy | "Jak porusza się alpha" | Szybkość, z jaką otwiera się brama cross-attention podczas treningu |
| Częstotliwość cross-attn | "Co M warstw" | Jak często wstawiany jest blok gated cross-attention; Flamingo używa M=4 |
| OpenFlamingo | "Otwarta reprodukcja" | Otwarty punkt kontrolny MosaicML/LAION przy 3-9B; architektonicznie identyczny z Flamingo |

## Dalsza Lektura

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — oryginalny artykuł.
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — otwarta reprodukcja.
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — przeplatany korpus internetowy.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — ogólna architektura Perceiver.
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — potomek Flamingo dostrojony instrukcjami.
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — nowoczesne uproszczenie podejścia Flamingo.