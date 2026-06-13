# Natywne Rzadkie Attention (DeepSeek NSA)

> Przy 64k tokenach attention pochłania 70-80% opóźnienia dekodowania. Każde laboratorium otwartych modeli ma plan, aby to naprawić. NSA DeepSeeka (najlepsza praca ACL 2025) to ten, który się przyjął: trzy równoległe gałęzie attention — skompresowane gruboziarniste tokeny, selektywnie zachowane drobnoziarniste tokeny i przesuwne okna dla lokalnego kontekstu — połączone przez wyuczoną bramkę. Jest dostosowany do sprzętu (przyjazny dla jąder), natywnie trenowalny (działa w pre-treningu, nie doczepiony przy inferencji), a na dekodach 64k działa szybciej niż FlashAttention, dorównując lub przewyższając jakość pełnego attention. Ta lekcja buduje trzy gałęzie end-to-end i pokazuje, dlaczego rzadkość jest end-to-end różniczkowalna.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 minutes

## Learning Objectives

- Wymień trzy gałęzie attention NSA i co każda z nich przechwytuje.
- Wyjaśnij, dlaczego NSA jest "natywnie trenowalne", podczas gdy wcześniejsze metody rzadkiego attention były tylko do inferencji.
- Oblicz oszczędności obliczeniowe attention NSA w porównaniu do pełnego attention przy kontekście 64k jako funkcję rozmiaru bloku kompresji i top-k selekcji.
- Zaimplementuj kombinację trzech gałęzi w stdlib Python na krótkiej syntetycznej sekwencji i zweryfikuj, że wagi bramkujące zachowują się poprawnie.

## Problem

Pełne attention przy długości sekwencji N kosztuje `O(N^2)` czasu i `O(N)` cache KV na warstwę. Przy 64k tokenach liczby dotyczące obliczeń i przepustowości pamięci są katastrofalne. Zmierzone teoretyczne oszacowanie z pracy NSA: attention odpowiada za 70-80% całkowitego opóźnienia dekodowania przy 64k. Wszystko downstream — TTFT, tokeny na sekundę, koszt na milion tokenów — jest zdominowane przez koszt attention.

Rzadkie attention jest oczywistą odpowiedzią. Wcześniejsze próby dzielą się na dwie kategorie. Rzadkość o stałym wzorcu (sliding-window, strided, block-local) wyrzuca informacje i zawodzi na zadaniach długodystansowego przywoływania. Rzadkość w czasie inferencji (przycinanie cache KV, H2O, StreamingLLM) jest stosowana do modelu wstępnie wytrenowanego na gęstym attention i odzyskuje tylko ułamek potencjalnego przyspieszenia, ponieważ model nigdy nie był proszony o kierowanie informacji przez rzadki wzorzec.

Native Sparse Attention (Yuan et al., DeepSeek + PKU + UW, najlepsza praca ACL 2025, arXiv:2502.11089) robi jedno i drugie: wzorzec rzadkości, którego model uczy się podczas pre-treningu, zaimplementowany jako algorytm dostosowany do jądra, który faktycznie dostarcza oszczędności obliczeniowych przy inferencji. Za dwa lata NSA lub bezpośredni potomek będzie domyślnym attention w każdym frontierowym modelu długiego kontekstu.

## Koncepcja

### Trzy równoległe gałęzie

Dla każdego zapytania, NSA uruchamia attention trzy razy, względem trzech różnych widoków cache KV:

1. **Gałąź skompresowana.** Tokeny są grupowane w bloki o rozmiarze `l` (zazwyczaj 32 lub 64). Każdy blok jest kompresowany do pojedynczego tokena podsumowującego przez mały wyuczony MLP. Zapytanie zwraca uwagę na te skompresowane tokeny, uzyskując gruboziarnisty widok całej sekwencji.

2. **Gałąź wybrana.** Używając wyników attention z gałęzi skompresowanej, identyfikowane są top-k bloków najbardziej istotnych dla bieżącego zapytania. Drobnoziarniste (nieskompresowane) tokeny z tych bloków są odczytywane, a zapytanie zwraca uwagę na wszystkie z nich. Myśl o attention gałęzi skompresowanej jako o sygnale routingu dla selekcji.

3. **Gałąź przesuwnego okna.** Zapytanie zwraca uwagę na ostatnie `W` tokenów (zazwyczaj 512) dla lokalnego kontekstu. Ta gałąź przechwytuje wzorce krótkiego zasięgu o dużej strukturze (składnia, lokalna koreferencja), które pozostałe dwie mogą przegapić.

Wyjścia trzech gałęzi są łączone za pomocą wyuczonej bramki na pozycję:

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` to wagi bramek z małego MLP na zapytaniu. Nie muszą sumować się do 1 — mogą ważyć gałęzie niezależnie.

### Dlaczego to jest "natywnie trenowalne"

Krok selekcji (top-k bloków) jest dyskretny. Operacje dyskretne przerywają przepływ gradientu. Wcześniejsze prace nad rzadkim attention albo pomijały wsteczną propagację przez selekcję (ograniczając trening), albo używały ciągłych relaksacji, które nie dawały prawdziwej rzadkości przy inferencji.

NSA omija to: attention gałęzi skompresowanej JEST różniczkowalnym gruboziarnistym attention na całej sekwencji. Operacja top-k po prostu ponownie wykorzystuje najwyższe wyniki attention z gałęzi skompresowanej, aby wybrać, które drobnoziarniste bloki załadować. Gradienty przepływają przez wyniki gałęzi skompresowanej (które wpływają zarówno na skompresowane wyjście, JAK i na logikę selekcji), a wkład wybranych bloków w końcowe wyjście jest również różniczkowalny. Niedróżniczkowalna operacja `top_k` jest no-op na forwardowym grafie obliczeniowym — kontroluje tylko, które bloki są ładowane z pamięci.

Dlatego NSA może być używane w pre-treningu end-to-end. Model uczy się kierować informacją przez trzy gałęzie wspólnie, produkując rzadki wzorzec, który przy inferencji faktycznie dostarcza obiecanego przyspieszenia.

### Jądro dostosowane do sprzętu

Jądro NSA jest zaprojektowane dla nowoczesnych hierarchii pamięci GPU. Jądro ładuje zapytania według grup GQA (pętla zewnętrzna), pobiera odpowiadające rzadkie bloki KV na grupę (pętla wewnętrzna) i uruchamia attention na SRAM. Ponieważ każda grupa zapytań widzi te same wybrane bloki (selekcja jest na grupę zapytań, a nie na głowicę zapytania), ładowania KV są amortyzowane w grupie. Intensywność arytmetyczna pozostaje wysoka.

Praca zgłasza jądra Triton działające 9x szybciej niż FlashAttention na dekodach 64k, przy czym stosunek przyspieszenia rośnie z długością sekwencji. Jądra forward i backward są oba dostarczone.

### Budżet obliczeniowy

Niech `N` będzie długością sekwencji, `l` rozmiarem bloku kompresji, `k` liczbą top-k selekcji, `w` przesuwnym oknem, `b` rozmiarem wybranego bloku (zazwyczaj równy `l`).

- Gałąź skompresowana: `O(N/l)` kluczy na zapytanie, więc `O(N * N / l)` łącznie.
- Gałąź wybrana: `O(k * b)` kluczy na zapytanie, więc `O(N * k * b)`.
- Gałąź przesuwna: `O(w)` kluczy na zapytanie, więc `O(N * w)`.

Razem: `O(N * (N/l + k*b + w))`.

Z `N = 64k, l = 64, k = 16, b = 64, w = 512`: koszt na zapytanie to `1000 + 1024 + 512 = 2536 kluczy`. Pełne attention to `64000 kluczy`. 25x redukcja obliczeń.

Z `N = 128k, l = 64, k = 16, b = 64, w = 512`: koszt na zapytanie to `2000 + 1024 + 512 = 3536 kluczy`. Pełne attention to `128000 kluczy`. 36x redukcja. Korzyść rośnie z długością sekwencji, o co właśnie chodzi.

### Jak to się porównuje

| Method | Differentiable | Real inference speedup | Long-range recall |
|--------|---------------|----------------------|-------------------|
| Tylko przesuwne okno | tak | tak | zawodzi |
| Strided / block-sparse | tak | tak | częściowe |
| Przycinanie KV (H2O, StreamingLLM) | N/D (czas inferencji) | tak | częściowe |
| MoBA (Moonshot) | częściowe | tak | dobre |
| NSA | tak (natywnie) | tak (9x przy 64k) | dorównuje pełnemu attention |

MoBA (Moonshot, arXiv:2502.13189) został opublikowany równocześnie i przyjmuje podobne podejście "trzy to lepsze niż jeden", stosując zasadę MoE do bloków attention. NSA i MoBA to dwie architektury, które warto znać dla pre-treningu długiego kontekstu w 2026.

```figure
sliding-window-attention
```

## Build It

`code/main.py` implementuje trzy gałęzie na krótkiej syntetycznej sekwencji i pokazuje:

- MLP kompresji (dla przejrzystości dydaktycznej używana jest prosta średnia pula; prawdziwe NSA używa wyuczonego MLP).
- Selekcję bloków top-k napędzaną wynikami gałęzi skompresowanej.
- Attention przesuwnego okna na ostatnich `w` tokenach.
- Bramkowaną kombinację.
- Wydruk liczby obliczeń w porównaniu do pełnego attention.

### Krok 1: skompresuj tokeny w bloki

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Krok 2: attention gałęzi skompresowanej

Uruchom softmax attention zapytania względem skompresowanych kluczy. Wyniki gałęzi skompresowanej służą jednocześnie jako sygnał do selekcji top-k.

### Krok 3: selekcja bloków top-k

Wybierz indeksy `k` najwyżej ocenionych skompresowanych bloków. Załaduj oryginalne nieskompresowane tokeny z tych bloków i uruchom na nich attention.

### Krok 4: attention przesuwnego okna

Weź ostatnie `w` tokenów i uruchom na nich standardowe attention.

### Krok 5: bramka + połączenie

Mały MLP na zapytaniu produkuje trzy wagi bramek. Końcowe wyjście to ważona suma wyjść trzech gałęzi.

### Krok 6: liczenie obliczeń

Wypisz liczbę kluczy, na które zwrócono uwagę na zapytanie dla każdej gałęzi i łącznie. Porównaj z `N` (pełne attention). Na syntetyku 1024 tokenów z `l = 32, k = 4, w = 128`, NSA widzi `32 + 128 + 128 = 288` kluczy na zapytanie w porównaniu do 1024 dla pełnego attention — 3.5x mniej.

## Use It

NSA jest wysyłane we własnym potoku pre-treningowym długiego kontekstu DeepSeeka. Status integracji w publicznych stosach inferencji od kwietnia 2026:

- **DeepSeek wewnętrznie**: natywne, opublikowane wagi używają NSA lub jego następcy DSA (Deepseek Sparse Attention).
- **vLLM**: eksperymentalne wsparcie NSA w rozwoju dla wag DeepSeek-V3.x.
- **SGLang**: benchmarki NSA opublikowane; ścieżka produkcyjna podąża za vLLM.
- **llama.cpp / CPU**: nieobsługiwane; narzut dekompozycji jądra nie jest wart zachodu przy przepustowości CPU.

Kiedy sięgać po NSA:

- Pre-trening lub kontynuowany trening celujący w kontekst 64k-plus z poważnym budżetem obliczeniowym.
- Inferencja własnych checkpointów długiego kontekstu DeepSeeka. Wagi są natywne dla NSA.

Kiedy nie:

- Serwowanie istniejącego wstępnie wytrenowanego modelu z gęstym attention. Nie możesz retroaktywnie dodać NSA bez kontynuowanego treningu.
- Kontekst poniżej 16k. Narzut trzech gałęzi dominuje nad oszczędnościami.
- Interaktywny czat batch-1. Dekodowanie wrażliwe na opóźnienie korzysta, ale tylko przy długich kontekstach.

## Ship It

Ta lekcja produkuje `outputs/skill-nsa-integrator.md`. Mając specyfikację uruchomienia pre-treningowego długiego kontekstu, produkuje plan integracji NSA: rozmiar bloku kompresji, top-k, przesuwne okno, szerokość MLP bramki, wybór jądra i konkretne ewaluacje długiego kontekstu, które uzasadniałyby zmianę architektury.

## Ćwiczenia

1. Uruchom `code/main.py` na syntetyku 1024 tokenów. Przeskanuj `(l, k, w)` przez trzy ustawienia wstępne i wypisz liczby obliczeń. Zidentyfikuj ustawienie, które osiąga najniższą liczbę kluczy na zapytanie, utrzymując 95% przywołania względem pełnego attention na teście needle-in-haystack.

2. Zastąp kompresor średniej puli małym wyuczonym MLP (2-warstwowy, ukryty 32). Wytrenuj go na syntetycznym zadaniu, gdzie sygnałem jest średnia bloku. Zmierz lukę perplexity względem linii bazowej średniej puli na danych wstrzymanych.

3. Zaimplementuj MLP bramki. Przyjmuje zapytanie jako wejście i zwraca trzy skalary. Pokaż, że bramka zachowuje się sensownie: wagi bliskie jednostajnym na losowych zapytaniach, duża waga na wybranej gałęzi, gdy zapytanie trafia na blok daleko w tyle.

4. Oblicz budżet pamięci cache KV dla modelu 70B z NSA przy kontekście 128k. Głowice KV to 8, head dim 128, BF16. Porównaj z pełnym attention i z MLA (Phase 10 · 14 pokazała liczby MLA). Zidentyfikuj długość sekwencji, przy której cache KV drobnoziarnistej gałęzi NSA równa się pełnemu attention.

5. Przeczytaj Sekcję 4 pracy NSA (arXiv:2502.11089) i wyjaśnij w trzech zdaniach, dlaczego wyniki attention gałęzi skompresowanej są ponownie wykorzystywane do selekcji top-k, zamiast obliczać osobny wynik routingu. Powiąż odpowiedź z przepływem gradientu.

## Kluczowe Pojęcia

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Compressed branch | "Gruboziarnisty widok" | Attention nad kluczami uśrednionymi w blokach, zapewniający globalny kontekst w O(N/l) kluczy na zapytanie |
| Selected branch | "Bloki top-k" | Drobnoziarniste attention nad `k` blokami z najwyższymi wynikami gałęzi skompresowanej |
| Sliding window | "Lokalny kontekst" | Attention nad ostatnimi `W` tokenami dla wzorców krótkiego zasięgu |
| Native trainability | "Pre-trenuj z włączoną rzadkością" | Wzorzec rzadkości jest uczony podczas pre-treningu, a nie doczepiany przy inferencji |
| Compression block size l | "Rozmiar grupy dla gruboziarnistego widoku" | Ile tokenów jest scalanych w jedno podsumowanie; typowo 32-64 |
| Top-k | "Bloki do zachowania" | Liczba skompresowanych bloków, których nieskompresowane tokeny są odczytywane; typowo 16 |
| Sliding window W | "Promień lokalnego attention" | Zazwyczaj 512; mniejsze szkodzi lokalnej spójności, większe marnuje obliczenia |
| Branch gate | "Jak wymieszać trzy" | Wyjście MLP na pozycję, które waży wkłady trzech gałęzi |
| Hardware alignment | "Rzadkość przyjazna dla jądra" | Wzorzec rzadkości wybrany tak, aby rzeczywiste jądro GPU osiągało teoretyczne przyspieszenie |
| DSA | "Następca NSA" | Deepseek Sparse Attention, architektura, która nastąpiła po NSA w linii DeepSeeka |

## Dalsza Lektura

- [Yuan et al. -- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) -- praca
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) -- rodzina architektur, na którą celuje NSA
- [Moonshot AI -- MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) -- równoczesna praca, attention w stylu MoE nad blokami
- [Beltagy et al. -- Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) -- początki przesuwnego okna
- [Xiao et al. -- StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) -- linia bazowa rzadkości w czasie inferencji, którą NSA ulepsza
- [Dao et al. -- FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) -- linia bazowa pełnego attention, którą jądra NSA biją przy 64k