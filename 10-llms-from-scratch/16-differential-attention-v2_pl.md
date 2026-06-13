# Attention Różnicowe (V2)

> Softmax attention rozprasza niewielką ilość prawdopodobieństwa na każdy niepasujący token. Przy 100k tokenów ten szum narasta i zagłusza sygnał. Differential Transformer (Ye et al., ICLR 2025) naprawia to, obliczając attention jako różnicę dwóch softmaxów, odejmując wspólne podłoże szumu. DIFF V2 (Microsoft, styczeń 2026) to przepisanie stosu produkcyjnego: dopasowanie opóźnienia dekodowania do bazowego Transformera, brak niestandardowych jąder, kompatybilność z FlashAttention. Ta lekcja to V1 do V2 end-to-end, z działającą zabawkową implementacją operacji różnicy, którą możesz uruchomić w stdlib Python.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## Learning Objectives

- Sformułuj precyzyjnie, dlaczego softmax attention ma podłoże szumu i dlaczego rośnie ono z długością kontekstu.
- Wyprowadź wzór attention różnicowego i wyjaśnij, dlaczego odejmowanie anuluje wspólny komponent szumu, zachowując sygnał.
- Przejdź różnicę V1-to-V2: co stało się szybsze, co prostsze, co bardziej stabilne i dlaczego każda zmiana była konieczna dla produkcyjnego pre-treningu.
- Zaimplementuj attention różnicowe od zera w czystym Pythonie i empirycznie zweryfikuj właściwość anulowania szumu na syntetycznym zapytaniu sygnał-plus-szum.

## Problem

Standardowe softmax attention ma właściwość matematyczną, która staje się operacyjnym bólem głowy na dużą skalę. Dla zapytania `q`, wagi attention to `softmax(qK^T / sqrt(d))`. Softmax nigdy nie może wyprodukować dokładnych zer -- każdy niepasujący token otrzymuje pewną dodatnią masę. Ta resztkowa masa to szum i skaluje się z długością kontekstu. Przy 128k tokenach, nawet jeśli każdy niepasujący token otrzymuje tylko 0.001% prawdopodobieństwa, 127 999 z nich łącznie wnosi około 12% całości. Model musi nauczyć się omijać podłoże szumu, które rośnie z kontekstem.

Empirycznie objawia się to jako interferencja głowic attention: halucynacyjne cytaty w długokontekstowym RAG, awarie lost-in-the-middle na zadaniach wyszukiwania na 100k tokenów i subtelna degradacja dokładności na benchmarkach needle-in-haystack powyżej 32k. Praca Differential Transformer (arXiv:2410.05258, ICLR 2025) zmierzyła lukę: DIFF Transformery osiągają niższą perplexity, wyższą dokładność na długich kontekstach i mniej halucynacji niż odpowiedniki tej samej wielkości.

DIFF V1 miał trzy problemy, które trzymały go z dala od frontierowych potoków pre-treningowych. Jego cache wartości musiał być ładowany dwa razy na krok dekodowania, wymagał niestandardowych jąder CUDA, które łamały kompatybilność z FlashAttention, a jego RMSNorm na głowicę destabilizował długotrwały trening w skali 70B-plus. DIFF V2 (blog Microsoft unilm, 20 stycznia 2026) naprawił wszystkie trzy. Ta lekcja omawia obie wersje, buduje operator różnicy i mierzy anulowanie szumu na zabawkowym zapytaniu.

## Koncepcja

### Podłoże szumu softmax

Dla zapytania `q` i kluczy `K = [k_1, ..., k_N]`, wagi attention to:

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

Żadne `w_i` nigdy nie jest zerem. Jeśli `k_i` jest całkowicie niezwiązane z `q`, wynik `q . k_i` nie wynosi 0 -- oscyluje wokół zera z wariancją `||q||^2 / d`. Po normalizacji softmax, każdy niezwiązany token wciąż wnosi `O(1/N)` do ważonej sumy. Całkowity wkład niezwiązanych tokenów wynosi `O((N-1)/N) = O(1)` -- nie jest to mała wielkość.

To, czego chce model, to coś w rodzaju twardego top-k: wysoka waga na pasujących tokenach, waga bliska zeru wszędzie indziej. Softmax jest zbyt gładki, aby zrobić to bezpośrednio.

### Idea różnicowa

Podziel projekcje Q i K każdej głowicy na dwie: Q = (Q_1, Q_2) i K = (K_1, K_2). Oblicz dwie mapy attention:

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

Wyjście:

```
DiffAttn = (A_1 - lambda * A_2) V
```

Odejmowanie anuluje dowolny rozkład szumu, który obie mapy dzielą. Jeśli obie mapy mają w przybliżeniu jednostajną wagę na 127k niezwiązanych tokenów (co będą miały przy losowej inicjalizacji), te się znoszą. Sygnał -- szczytowa waga na kilku faktycznie istotnych tokenach -- znosi się tylko wtedy, gdy pojawia się w obu mapach z tą samą wielkością, co nie nastąpi, gdy model się wytrenuje.

`lambda` to skalarny parametr do nauczenia na głowicę, sparametryzowany jako `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. Może być ujemny. `lambda_init` domyślnie wynosi małą dodatnią liczbę, np. 0.8.

### Dlaczego to przypomina aktywne wyciszanie szumu

Pomyśl o dwóch hałaśliwych mikrofonach nagrywających ten sam głos. Oba rejestrują mówcę plus skorelowany szum tła. Odejmij jeden od drugiego, a wspólny szum znika. Głos przetrwa, ponieważ dwa sygnały różnią się fazą lub amplitudą na tyle, aby zapobiec całkowitemu wyzerowaniu. Per-head `lambda` uczy się dokładnie tej równowagi.

### V1 vs V2: różnica

V1 utrzymywał liczbę parametrów równą bazowemu Transformerowi. Aby uzyskać dwa zapytania na głowicę, zmniejszył o połowę wymiar głowicy. To kosztowało ekspresyjność głowicy i -- co bardziej bolesne -- zmniejszyło o połowę cache wartości na głowicę. Dekodowanie musiało ładować cache wartości dwa razy na krok (raz na gałąź softmax). Wynik: dekodowanie wolniejsze niż bazowe pomimo równej liczby parametrów.

V2 podwaja liczbę głowic zapytań i utrzymuje te same głowice KV (pożyczając parametry z projekcji w górę). Wymiar głowicy pozostaje taki sam jak w bazowym. Po odejmowaniu, dodatkowy wymiar jest rzutowany z powrotem w dół, aby dopasować projekcję O_W bazowego Transformera. Trzy rzeczy dzieją się jednocześnie:

1. Szybkość dekodowania dorównuje bazowemu (cache KV jest ładowany raz).
2. FlashAttention działa bez zmian (brak niestandardowego jądra).
3. Intensywność arytmetyczna przy dekodowaniu wzrasta (więcej obliczeń na bajt załadowany z HBM).

V2 usuwa również RMSNorm na głowicę, którego V1 używał do stabilizacji odejmowania. W skalach pre-treningu klasy 70B, ten RMSNorm destabilizował późny trening. V2 zastępuje go prostszym schematem inicjalizacji, który utrzymuje stabilność treningu bez dodatkowego modułu.

### Kiedy po to sięgać

| Workload | Benefit |
|----------|---------|
| Długokontekstowy RAG (64k+) | Czystsze mapy attention, mniej halucynacyjnych cytatów |
| Benchmarki needle-in-haystack | Znaczący wzrost dokładności powyżej 32k |
| QA na wielu dokumentach | Mniejsza interferencja między dokumentami |
| Uzupełnianie kodu przy 8k | Marginalne, nie warte zmiany architektury |
| Krótki czat (< 4k) | Zasadniczo nie do odróżnienia od bazowego |

Wartość rośnie z długością kontekstu. Przy 4k tokenach podłoże szumu jest wystarczająco małe, aby standardowe attention było w porządku. Przy 128k już ci szkodzi.

### Jak się łączy z innymi pokrętłami 2026

| Feature | Compatible with DIFF V2? |
|---------|------------------------|
| GQA | Tak (V2 zwiększa głowice Q, nie KV) |
| MLA (DeepSeek) | Tak w zasadzie, brak opublikowanej pracy łączącej je |
| MoE | Tak (attention jest niezależne od bloku MLP) |
| RoPE | Tak (bez zmian) |
| YaRN / skalowanie długiego kontekstu | Tak (dokładnie tam, gdzie DIFF pomaga najbardziej) |
| FlashAttention | Tak w V2 (nie w V1) |
| Dekodowanie spekulatywne | Tak (zmiana attention jest niewidoczna dla pętli spekulatywnej) |

```figure
differential-attention
```

## Build It

`code/main.py` implementuje attention różnicowe w czystym Pythonie. Zabawkowe zapytanie ze znaną strukturą sygnał-plus-szum pozwala bezpośrednio zmierzyć współczynnik anulowania szumu.

### Krok 1: standardowe softmax attention

Operacje na macierzach ze stdlib: listy list, ręczny matmul, softmax z numerycznie stabilnym odejmowaniem maksimum.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Krok 2: podziel Q, K na dwie połowy

Styl V1: zmniejsz o połowę wymiar głowicy. Styl V2: zachowaj wymiar głowicy i podwój liczbę głowic. Zabawkowa implementacja używa V1 dla przejrzystości dydaktycznej -- matematyka jest identyczna, różni się tylko księgowość.

### Krok 3: dwie gałęzie softmax + odejmowanie

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Uwaga: wagi wyjściowe mogą być ujemne. To w porządku -- cache wartości nadal obsługuje dodatnie i ujemne wkłady. Kolejna projekcja V absorbuje znak.

### Krok 4: pomiar anulowania szumu

Zbuduj syntetyczną sekwencję o długości 1024. Umieść token sygnałowy na znanej pozycji, wypełnij resztę szumem. Oblicz (a) standardową wagę softmax attention na pozycji sygnału i (b) wagę attention różnicowego. Zmierz stosunek sygnału do szumu w każdym. DIFF attention niezawodnie produkuje wyższy stosunek sygnału do szumu o czynnik 3x-10x, w zależności od tego, jak bardzo dwie gałęzie zostały wytrenowane, aby się różnić.

### Krok 5: księgowość parametrów V1 vs V2

Mając konfigurację (hidden=4096, heads=32, d_head=128), wypisz:

- Bazowy Transformer: Q, K, V każdy rozmiaru `hidden * hidden`, MLP przy 4 * hidden.
- DIFF V1: Q, K każdy rozmiaru `hidden * hidden`, V rozmiaru `hidden * hidden` (bez zmian), head dim zmniejszony o połowę wewnętrznie. Dodaje parametry `lambda` na głowicę (O(heads * d_head)).
- DIFF V2: Q rozmiaru `2 * hidden * hidden`, K rozmiaru `hidden * hidden`, V rozmiaru `hidden * hidden`. Dodatkowy wymiar rzutowany z powrotem w dół przed O_W. Dodaje te same parametry `lambda`.

Zabawka mierzy dodatkowy koszt parametrów dla V2 (mniej więcej `hidden * hidden` dodatkowo na blok attention) i wypisuje go.

## Use It

DIFF V2 nie jest jeszcze wysyłany na każdym produkcyjnym serwerze inferencji od kwietnia 2026, ale integracja trwa w vLLM i SGLang. Tymczasem wzorzec pojawia się w:

- Wewnętrznych produkcyjnych modelach długiego kontekstu Microsoftu.
- Replikacjach badawczych w kilku otwartych uruchomieniach treningowych modeli celujących w kontekst 256k-plus.
- Hybrydowych architekturach łączących DIFF attention z sliding-window attention na naprzemiennych warstwach.

Kiedy sięgnąłbyś po to w 2026:

- Trenujesz nowy model od zera, celując w efektywny kontekst 64k-plus. Dodaj attention różnicowe od początku; późniejszy retrening jest drogi.
- Fine-tunujesz model długiego kontekstu, gdzie awarie lost-in-the-middle dominują w twojej ewaluacji. LoRA na projekcjach Q może przybliżyć strukturę DIFF.

Kiedy byś nie sięgnął:

- Serwujesz wstępnie wytrenowany gęsty model ze stabilną wydajnością na długim kontekście. Koszt retreningu rzadko się zwraca na istniejących wagach.
- Twój kontekst jest zawsze poniżej 16k. Podłoże szumu jest pomijalne.

## Ship It

Ta lekcja produkuje `outputs/skill-diff-attention-integrator.md`. Mając architekturę modelu, docelową długość kontekstu, profil halucynacji i budżet treningowy, produkuje plan integracji dodania attention różnicowego do nowego uruchomienia pre-treningowego lub fine-tuningu LoRA.

## Ćwiczenia

1. Uruchom `code/main.py`. Zweryfikuj, że stosunek sygnału do szumu zgłoszony dla attention różnicowego jest wyższy niż dla standardowego softmax attention na syntetycznym zapytaniu. Zmień amplitudę szumu i pokaż punkt przecięcia, w którym standardowe attention staje się bezużyteczne.

2. Oblicz deltę liczby parametrów od bazowego do DIFF V1 i od bazowego do DIFF V2 dla modelu klasy 7B (hidden=4096, heads=32, d_head=128, 32 warstwy). Pokaż, które komponenty zyskały parametry, a które pozostały bez zmian.

3. Przeczytaj Sekcję 3 pracy DIFF V1 (arXiv:2410.05258) i Sekcję 2 bloga DIFF V2 na Hugging Face. W dwóch zdaniach wyjaśnij, dlaczego RMSNorm na głowicę w V1 był konieczny i dlaczego V2 mógł go usunąć bez powodowania dywergencji treningu.

4. Zaimplementuj ablację: oblicz attention różnicowe z `lambda = 0` (czysty pierwszy softmax) i `lambda = 1` (pełne odejmowanie). Na syntetycznym zapytaniu zmierz, jak stosunek sygnału do szumu zmienia się w całym zakresie. Zidentyfikuj `lambda`, które maksymalizuje stosunek sygnału do szumu.

5. Rozszerz zabawkę o GQA + DIFF V2. Wybierz 8 głowic KV i 32 głowice Q. Pokaż, że rozmiar cache KV odpowiada bazowemu modelowi GQA z tą samą konfiguracją (8, 32).

## Kluczowe Pojęcia

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Differential attention | "Dwa softmaxy minus siebie nawzajem" | Podziel Q, K na dwie połowy, oblicz dwie mapy softmax, odejmij drugą (skalowaną przez lambda) od pierwszej, następnie pomnóż przez V |
| Noise floor | "Niezerowy ogon softmax" | Waga O(1/N), którą softmax nakłada na każdy niezwiązany token, sumująca się do O(1) w długich kontekstach |
| lambda | "Skala odejmowania" | Skalarny parametr do nauczenia na głowicę, sparametryzowany jako `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`; może być ujemny |
| DIFF V1 | "Wersja ICLR 2025" | Oryginalny Differential Transformer; zmniejsza o połowę head dim, aby zachować liczbę parametrów, potrzebuje niestandardowego jądra, wolniejsze dekodowanie |
| DIFF V2 | "Poprawka ze stycznia 2026" | Podwaja głowice Q, zachowując głowice KV; dorównuje szybkości dekodowania bazowego i działa z FlashAttention |
| Per-head RMSNorm | "Stabilizator V1" | Dodatkowa norma, którą V1 stosował po różnicy; V2 usunął ją, aby zapobiec niestabilności późnego treningu |
| Signal-to-noise ratio | "Ile attention jest marnowane" | Stosunek wagi na prawdziwej pozycji sygnału do średniej wagi na niepowiązanych pozycjach |
| Lost in the middle | "Tryb awarii długiego kontekstu" | Empiryczne zjawisko, w którym dokładność wyszukiwania spada dla dokumentów w środku długiego kontekstu -- DIFF attention to redukuje |
| Arithmetic intensity | "FLOPy na załadowany bajt" | Stosunek, który V2 zwiększył przy dekodowaniu poprzez podwojenie zapytań na ładowanie KV; ważne dla dekodowania ograniczonego pamięcią |

## Dalsza Lektura

- [Ye et al. -- Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) -- oryginalna praca z teorią anulowania szumu i ablacjami na długim kontekście
- [Microsoft unilm -- Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) -- przepisanie stosu produkcyjnego, dopasowanie dekodowania bazowego, kompatybilność z FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) -- analiza teoretyczna, dlaczego odejmowanie odzyskuje strukturę wstępnie wytrenowanego attention
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) -- wariant współdzielenia parametrów
- [Vaswani et al. -- Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) -- bazowy Transformer, od którego DIFF odejmuje
- [Liu et al. -- Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) -- benchmark długiego kontekstu, na który celuje DIFF attention