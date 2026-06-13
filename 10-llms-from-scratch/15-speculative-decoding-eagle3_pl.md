# Dekodowanie Spekulatywne i EAGLE-3

> Phase 7 · Lekcja 16 udowodniła matematykę: reguła odrzucenia Leviatana zachowuje dokładnie rozkład weryfikatora. Ta lekcja to widok stosu treningowego produkcyjnego dekodowania spekulatywnego w 2026. EAGLE-3 przekształcił model szkicowy z taniego przybliżenia w celowo zbudowaną małą sieć trenowaną na własnych ukrytych stanach weryfikatora, a następnie dodał pętlę testową w czasie treningu, która dopasowuje rozkłady treningowe i inferencyjne. Wynik: 3× do 6.5× przyspieszenia end-to-end, wskaźniki akceptacji powyżej 0.9 na czacie, bez kompromisu dystrybucyjnego. Każdy produkcyjny stos inferencji w 2026 domyślnie go wysyła.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75 minutes

## Learning Objectives

- Sformułuj twierdzenie Leviatana w jednym zdaniu i udowodnij, że pętla spekulatywna produkuje próbki o identycznym rozkładzie co weryfikator.
- Przejdź dwuletnią progresję od waniliowego dekodowania spekulatywnego (Leviathan 2023) przez EAGLE, EAGLE-2 i EAGLE-3 i nazwij dokładne ograniczenie, które każdy krok usunął.
- Oblicz oczekiwane przyspieszenie ze wskaźnika akceptacji `α` i stosunku kosztów szkicu do weryfikatora `c`, i wybierz optymalną długość szkicu `N` dla każdego reżimu.
- Zaimplementuj pełną pętlę spekulatywną od zera: szkic, weryfikacja, próbkowanie z odrzucenia z rozkładu resztowego, wycofanie cache KV przy odrzuceniu, emisja bonusowego tokena przy pełnej akceptacji.

## Problem

Autoregresyjne dekodowanie na modelu 70B działa z prędkością może 35 tokenów na sekundę na H100. GPU nie jest nawet blisko nasycone. Przepustowość pamięci jest ograniczeniem: każdy token ładuje 70B wag z HBM, wykonuje jeden krok arytmetyki i produkuje jednego floata. Jednostki obliczeniowe siedzą w większości bezczynnie.

Dekodowanie spekulatywne zamienia to w problem przepustowości, który faktycznie można rozwiązać. Tani szkic proponuje `N` tokenów w `N` małych forward passach. Weryfikator uruchamia się raz na prefiksie plus wszystkie `N` szkiców. Jeśli rozkład weryfikatora na pozycji `i` zgadza się ze szkicem (w sensie statystycznym, który sprecyzujemy), akceptujemy; jeśli nie, odrzucamy i próbkujemy korektę z rozkładu resztowego. Pojedynczy forward dużego modelu produkuje do `N+1` zaakceptowanych tokenów zamiast jednego.

Twierdzenie, które ma znaczenie, to Leviathan, Kalman, Matias (ICML 2023): rozkład wyjściowy jest identyczny z tym, co wyprodukowałoby bezpośrednie próbkowanie z weryfikatora. Nie w przybliżeniu. Identyczny. To jest cały powód, dla którego dekodowanie spekulatywne jest akceptowalne w produkcji -- jest to czysta optymalizacja opóźnienia bez kompromisu jakościowego.

To, co dała ci Phase 7 · Lekcja 16, to matematyka. To, co daje ci ta lekcja, to stos treningowy. Dobry szkic jest wart 2× więcej przyspieszenia niż tani szkic. EAGLE, EAGLE-2 i EAGLE-3 (Li et al., 2024–2025) przekształciły "szkic = mniejsza wersja tego samego modelu" w precyzyjną dyscyplinę inżynieryjną. Produkcyjne serwery inferencji w 2026 domyślnie używają EAGLE-3.

## Koncepcja

### Niezmiennik: próbkowanie z odrzuceniem Leviatana

Niech `p(t)` będzie rozkładem szkicu dla następnego tokena przy danym prefiksie, a `q(t)` rozkładem weryfikatora. Próbkuj token szkicu `d ~ p`. Zaakceptuj z prawdopodobieństwem `min(1, q(d) / p(d))`. Przy odrzuceniu, próbkuj z rozkładu resztowego `(q - p)_+ / ||(q - p)_+||_1`. Wynikowe próbki mają rozkład `q`. Jest to prawdziwe niezależnie od tego, jak zły jest `p` -- im gorszy, tym częściej odrzucasz, ale wynik pozostaje dokładny.

Ułóż `N` takich wywołań jeden za drugim, używając jednego forward passu weryfikatora na `prefix + d_1 + ... + d_N`. Weryfikator zwraca `q_1, q_2, ..., q_{N+1}` jednocześnie. Idź od lewej do prawej. Przy pierwszym odrzuceniu na pozycji `j`, próbkuj z `residual(q_j, p_j)` i zatrzymaj się. Przy pełnej akceptacji, próbkuj jeden bonusowy token z `q_{N+1}`.

### Co determinuje przyspieszenie

Niech `α` będzie oczekiwanym wskaźnikiem akceptacji na token szkicu. Niech `c = cost(szkic) / cost(weryfikator)` będzie stosunkiem kosztów. Oczekiwana liczba zaakceptowanych tokenów na forward weryfikatora:

```
E[akceptowane] = (1 - α^(N+1)) / (1 - α)
```

Oczekiwany całkowity czas ścienny na zaakceptowany token to `(N * c + 1) / E[akceptowane]`. Zminimalizuj to względem `N`, a otrzymasz optymalny punkt. Dla `α = 0.8, c = 0.05`: optymalne `N` to około 5-7, przyspieszenie 3.2×. Dla `α = 0.95, c = 0.02`: optymalne `N` to około 8-10, przyspieszenie sięga 5×.

Pojedynczą największą dźwignią jest `α`. Przejście z `α = 0.6` (waniliowy szkic) do `α = 0.9` (EAGLE-3) przy stałym `N = 5` przenosi cię z 2.2 oczekiwanych zaakceptowanych tokenów na forward weryfikatora do 4.1. Prawie 2× więcej przepustowości z tego samego weryfikatora.

### Dwuletnia progresja

**Waniliowe spekulatywne (Leviathan, 2023).** Model szkicowy to niezależnie wytrenowany mniejszy LLM z tej samej rodziny. Łatwy do podłączenia, `α ≈ 0.6`, przyspieszenie około 2× w najlepszym razie.

**EAGLE-1 (Li et al., 2024).** Szkic to mały transformer -- zazwyczaj jedna lub dwie warstwy -- który przyjmuje ukryty stan ostatniej warstwy weryfikatora jako wejście i przewiduje następny token bezpośrednio. Ponieważ szkic widzi reprezentację cech weryfikatora, jego rozkład jest znacznie bliższy rozkładowi weryfikatora. `α` wzrasta do 0.7-0.8.

**EAGLE-2 (Li et al., 2024).** Dodaje dynamiczne drzewo szkiców: zamiast proponować pojedynczą sekwencję `N` tokenów, proponuje małe drzewo kandydatów, ocenia każdego za pomocą weryfikatora w jednym forward passie (tree attention) i podąża ścieżką o najwyższym prawdopodobieństwie. Długość szkicu staje się adaptacyjna na krok. `α` na token zaakceptowanej ścieżki wzrasta powyżej 0.85.

**EAGLE-3 (Li et al., 2025, NeurIPS).** Dwie kolejne zmiany. Po pierwsze, porzuć całkowicie stratę przewidywania cech -- EAGLE-1/2 trenowały szkic, aby dopasować ukryte stany weryfikatora, co ograniczało, ile danych pomaga. EAGLE-3 trenuje bezpośrednio na przewidywaniu tokenów. Po drugie, test w czasie treningu (TTT): podczas treningu szkicu, podawaj własne poprzednie przewidywania szkicu z powrotem jako dane wejściowe przez wiele kroków, tak samo jak działa przy inferencji. To dopasowuje rozkłady treningowe i inferencyjne i zatrzymuje akumulację błędów. Zmierzone przyspieszenie: do 6.5× na czacie, 38% poprawa przepustowości przy batch 64 w SGLang na H100.

### Wycofanie cache KV

Weryfikacja rozszerza cache KV weryfikatora o `N` wpisów w jednym passie. Jeśli odrzucenie nastąpi na pozycji `j`, zawartość cache za pozycją `j-1` jest teraz nieprawidłowa. Dwie powszechne implementacje: zapisz do scratch bufora i zatwierdź przy akceptacji (vLLM, TensorRT-LLM), lub przechowuj fizyczny cache KV plus logiczną długość i obetnij przy odrzuceniu. W obu przypadkach koszt wycofania to bajty na warstwę na głowicę, co jest pomijalne w porównaniu do kosztu forward passu.

Dla przeszukiwania drzewa EAGLE-2, weryfikator uruchamia attention z nieprzyczynową maską, która respektuje topologię drzewa. Inżynieria jest drobiazgowa, ale obliczenia to standardowe wywołanie flash-attention z niestandardową maską.

### Architektury szkiców w 2026

| Strategy | Draft type | `α` | Speedup | Training cost |
|----------|-----------|-----|---------|---------------|
| Vanilla | Oddzielny mały LLM | 0.55-0.70 | 1.8-2.3× | Brak (ponowne użycie istniejącego małego modelu) |
| Medusa | Dodatkowe głowice LM na weryfikatorze | 0.65-0.75 | 2-3× | ~1B tokenów SFT |
| EAGLE-1 | 1-warstwowy transformer na ukrytych stanach | 0.70-0.80 | 2.5-3× | ~60B tokenów |
| EAGLE-2 | EAGLE-1 + dynamiczne drzewo szkiców | 0.80-0.88 | 3-4× | ~60B tokenów |
| EAGLE-3 | Wielowarstwowa fuzja cech + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B tokenów |
| Lookahead | Brak szkicu (iteracja Jacobiego) | N/D | 1.3-1.6× | Brak |

W produkcji 2026: vLLM i SGLang domyślnie używają EAGLE-3, gdy dostępny, w przeciwnym razie EAGLE-2. TensorRT-LLM ma najszybszą ścieżkę Medusa dla modeli Meta i NVIDIA. llama.cpp wysyła waniliowy szkic dla wdrożeń CPU.

## Build It

Zobacz `code/main.py`. To jest pełna pętla spekulatywna Leviatana ze wszystkimi elementami: szkic-N, równoległy pass weryfikatora, odrzucenie na pozycję, próbkowanie resztowe, bonusowy token, wycofanie KV i empiryczna weryfikacja, że rozkład wyjściowy odpowiada bezpośredniemu próbkowaniu z `q`.

### Krok 1: reguła odrzucenia

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Krok 2: rozkład resztowy

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Krok 3: pełny krok spekulatywny

Funkcja `spec_step` szkicuje `N` tokenów z `p`, a następnie weryfikuje wszystkie w jednej równoległej ewaluacji `q`. Dla każdego tokena szkicu stosuje regułę odrzucenia, a przy pierwszym odrzuceniu próbkuje korektę z rozkładu resztowego. Jeśli wszystko zostanie zaakceptowane, emituje bonusowy token z `q_{N+1}`.

### Krok 4: księgowość wycofania KV

Symulator śledzi logiczny `kv_length` na pracownika. Przy akceptacji `k` szkiców, `kv_length += k`. Przy odrzuceniu na pozycji `j`, cache jest już zapisany za `j`, ale logiczna długość jest ustawiona na `prefix_length + j + 1` -- jeden za tokenem korekty. Kolejne odczyty obcinają do logicznej długości.

### Krok 5: sprawdzenie Leviatana

Uruchom 50 000 kroków spekulatywnych. Policz empiryczny rozkład zaakceptowanych tokenów. Porównaj z 50 000 bezpośrednich próbek z `q`. Statystyka chi-kwadrat powinna być znacznie poniżej wartości krytycznej. Twierdzenie przechodzi w praktyce.

### Krok 6: przyspieszenie vs. α

Przeskanuj jakość szkicu, zaburzając `p` od `q` z różnymi amplitudami. Zmierz `α`, a następnie wykreśl oczekiwane tokeny na wywołanie weryfikatora jako funkcję `α` i `N`. Kod wypisuje tabelę pokazującą, jak jakość szkicu klasy EAGLE-3 (`α ≈ 0.9`) odblokowuje 4-5 tokenów na wywołanie weryfikatora.

## Use It

Produkcyjne `vllm serve` z EAGLE-3:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang z EAGLE-3 przy batch 64 na H100: około 1.38× więcej przepustowości niż waniliowe dekodowanie batch-64, zgodnie z pracą EAGLE-3.

Kiedy sięgać po dekodowanie spekulatywne:

- Każde interaktywne obciążenie czatowe, gdzie opóźnienie p50 ma znaczenie bardziej niż szczytowa przepustowość.
- Generowanie kodu i strukturalne wyjście (JSON, SQL). `α` jest powyżej 0.9, ponieważ docelowy rozkład jest wysoce przewidywalny.
- Długa forma generacji (tysiące tokenów). Amortyzowane przyspieszenie wciąż się opłaca.

Kiedy nie:

- Bardzo małe modele (< 3B). Szkic nie jest dużo tańszy niż weryfikator.
- Małe wdrożenia CPU batch-1. Narzut pamięciowy modelu szkicu może nie być wart zachodu.
- Bardzo wysokotemperaturowe kreatywne próbkowanie, gdzie `α` załamuje się.

## Ship It

Ta lekcja produkuje `outputs/skill-eagle3-tuner.md`. Mając obciążenie inferencyjne (model, rozmiar batcha, docelowe opóźnienie, profil zadania), rekomenduje strategię dekodowania spekulatywnego i parametry strojenia (rodzina szkicu, `N`, głębokość drzewa, przełączanie w zależności od temperatury).

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że statystyka chi-kwadrat na sprawdzeniu rozkładu Leviatana pozostaje poniżej 95% wartości krytycznej na 50 000 próbek.

2. Przeskanuj `N` od 1 do 10 z `α` utrzymanym na 0.9 i `c` utrzymanym na 0.04. Wykreśl oczekiwane tokeny na wywołanie weryfikatora i rzeczywisty czas ścienny na token. Znajdź `N`, które minimalizuje czas ścienny. Wyjaśnij kształt krzywej.

3. Zmodyfikuj kod, aby symulować przeszukiwanie drzewa EAGLE-2: na każdym kroku szkic proponuje drzewo o kształcie `[2, 2, 2]` (osiem ścieżek kandydackich). Weryfikator uruchamia się raz, a ścieżka o najwyższym prawdopodobieństwie wygrywa. Oblicz `α` na liść i całkowite tokeny na wywołanie weryfikatora. Porównaj z liniowym łańcuchem dekodowania spekulatywnego przy równoważnych obliczeniach.

4. Zaimplementuj symulator wsadowego wycofania KV dla dwóch równoczesnych sekwencji. Sekwencja A ma wszystkie szkice zaakceptowane; sekwencja B odrzuca na pozycji 2. Pokaż, że poprawny `kv_length` jest aktualizowany na sekwencję i że żadna praca nie jest marnowana.

5. Przeczytaj Sekcję 4 pracy EAGLE-3 (Training-Time Test). Wyjaśnij w dwóch zdaniach, dlaczego naiwny trening szkicu bez TTT cierpi na błąd ekspozycji i dlaczego podawanie szkicowi jego własnych przewidywań podczas treningu to naprawia. Połącz to z literaturą scheduled-sampling w seq2seq.

## Kluczowe Pojęcia

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Leviathan rule | "min(1, q over p)" | Bernoulliego akceptuj/odrzuć z prawdopodobieństwem `min(1, q(d)/p(d))`, zachowuje dokładnie rozkład weryfikatora, gdy próbkujesz z reszty przy odrzuceniu |
| Residual distribution | "(q minus p) plus, znormalizowane" | `(q - p)_+` przycięte do zera i zrenormalizowane -- poprawny rozkład do próbkowania przy odrzuceniu |
| Acceptance rate α | "jak często szkic ma rację" | Oczekiwane prawdopodobieństwo sukcesu Bernoulliego na token przy regule odrzucenia; rządzi całą matematyką przyspieszenia |
| EAGLE-1 | "szkic na ukrytych stanach" | Mały transformerowy szkic warunkowany na ukrytym stanie ostatniej warstwy weryfikatora (Li et al., 2024) |
| EAGLE-2 | "dynamiczne drzewo szkiców" | EAGLE-1 plus drzewo kandydackich kontynuacji ocenianych za pomocą tree attention w jednym passie weryfikatora |
| EAGLE-3 | "test w czasie treningu" | Porzuca stratę przewidywania cech, trenuje na bezpośrednim przewidywaniu tokenów ze szkicem karmionym własnymi wyjściami podczas treningu |
| Training-time test (TTT) | "naprawa błędu ekspozycji" | Uruchom szkic autoregresyjnie podczas treningu, aby rozkłady wejściowe treningu i inferencji były zgodne -- bezpośredni analog scheduled sampling |
| KV rollback | "cofnij odrzucone szkice" | Księgowość resetująca cache KV weryfikatora do długości zaakceptowanego prefiksu po odrzuceniu |
| Bonus token | "ten darmowy" | Gdy wszystkie `N` szkiców zostanie zaakceptowanych, próbkuj jeden dodatkowy z `q_{N+1}` bez dodatkowego kosztu weryfikatora |
| Tree attention | "zweryfikuj wielu kandydatów naraz" | Attention z nieprzyczynową maską respektującą topologię drzewa szkiców; oblicza `q_i` dla każdego węzła w drzewie w jednym forward passie |

## Dalsza Lektura

- [Leviathan, Kalman, Matias -- Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) -- fundamentalna praca i twierdzenie o równoważności
- [Chen et al. -- Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) -- równoczesne niezależne wprowadzenie z czystym dowodem
- [Li et al. -- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) -- EAGLE-1, szkic warunkowany na ukrytym stanie
- [Li et al. -- EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) -- dynamiczne przeszukiwanie drzewa
- [Li et al. -- EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) -- domyślny wybór produkcyjny 2026
- [Cai et al. -- Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) -- alternatywne podejście bez szkicu
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) -- kanoniczne odniesienie produkcyjne ze wszystkimi strategiami