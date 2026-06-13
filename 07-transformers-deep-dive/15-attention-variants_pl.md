# Warianty Attencji — Okno Przesuwne, Rzadka, Różnicowa

> Pełna atencja to koło. Każdy token widzi każdy token, a pamięć płaci cenę. Cztery warianty wyginają kształt koła i odzyskują połowę kosztu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## Problem

Pełna atencja kosztuje `O(N²)` pamięci i `O(N²)` obliczeń w zależności od długości sekwencji. Dla 128K kontekstu Llama 3 70B to 16 miliardów wpisów atencji na warstwę, razy 80 warstw. Flash Attention (Lekcja 12) ukrywa pamięć aktywacji `O(N²)`, ale nie zmienia kosztu arytmetycznego — każdy token nadal zwraca uwagę na każdy inny token.

Trzy klasy wariantów zmieniają topologię samej macierzy atencji:

1. **Atencja z przesuwnym oknem (SWA).** Każdy token zwraca uwagę na stałe okno sąsiadów, a nie na cały prefiks. Pamięć i obliczenia spadają do `O(N · W)`, gdzie `W` to rozmiar okna. Gemma 2/3, pierwsze warstwy Mistral 7B, Phi-3-Long.
2. **Atencja rzadka / blokowa.** Tylko wybrane pary `(i, j)` są oceniane; reszta ma wymuszoną wagę zero. Longformer, BigBird, rzadki transformer OpenAI.
3. **Atencja różnicowa.** Oblicz dwie mapy atencji z oddzielnymi projekcjami Q/K, odejmij jedną od drugiej. Zabija "zlew atencji", który wycieka wagą do pierwszych kilku tokenów. DIFF Transformer Microsoftu (2024).

Te podejścia współistnieją. Model na granicy możliwości w 2026 często je miesza: większość warstw to SWA-1024, co piąta to globalna pełna atencja, a garść to głowice różnicowe, które czyszczą wyszukiwanie. Stosunek 5:1 SWA do globalnej w Gemma 3 to obecny podręcznikowy domyślny.

## Koncepcja

### Atencja z przesuwnym oknem (SWA)

Każde zapytanie na pozycji `i` zwraca uwagę tylko na pozycje w `[i - W, i]` (przyczynowa SWA) lub `[i - W/2, i + W/2]` (dwukierunkowa). Tokeny poza oknem dostają `-inf` w macierzy ocen.

```
pełna przyczynowa:     przesuwne okno (W=4):
pozycje 0-7            pozycje 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

Dla `N = 8192` i `W = 1024`, macierz ocen ma oczekiwanie 1024 × 8192 niezerowych wierszy — 8× redukcja.

**Cache KV kurczy się z SWA.** Tylko ostatnie `W` tokenów K i V musi być przechowywanych na warstwę. Dla konfiguracji w stylu Gemma-3 (okno 1024, kontekst 128K), cache KV spada 128×.

**Koszt jakościowy.** Modele wyłącznie SWA mają problem z długodystansowym wyszukiwaniem. Rozwiązanie: przeplataj warstwy SWA z warstwami pełnej atencji. Gemma 3 używa stosunku 5:1 SWA:globalna. Mistral 7B używał stosu przyczynowej SWA, gdzie informacja "płynie do przodu" przez zachodzące na siebie okna — każda warstwa zwiększa efektywne pole recepcyjne o `W`, a po `L` warstwach model może zwracać uwagę `L × W` tokenów wstecz.

### Atencja rzadka / blokowa

Wybierz z góry wzór rzadkości `N × N`. Trzy kanoniczne kształty:

- **Lokalna + z przeskokiem (rzadki transformer OpenAI).** Atencja do ostatnich `W` tokenów plus co `stride`-ty token przed nimi. Łapie zarówno lokalność, jak i długi zasięg przy `O(N · sqrt(N))` obliczeń.
- **Longformer / BigBird.** Lokalne okno + mały zestaw globalnych tokenów (np. `[CLS]`), które zwracają uwagę na wszystkich i wszyscy zwracają uwagę na nie + losowo-rzadkie połączenia. Empirycznie 2× kontekstu przy dopasowanej jakości.
- **Natynna rzadka atencja (DeepSeek, 2025).** Ucz się, które bloki `(Q, K)` mają znaczenie; pomiń zerowe bloki na poziomie jądra. Zgodne z FlashAttention.

Rzadka atencja to historia inżynierii jąder. Matematyka jest prosta (maskuj macierz ocen); zysk pochodzi z nigdy nieładowania zerowych wpisów do SRAM. FlashAttention-3 i API FlexAttention z 2026 czynią niestandardowe wzory rzadkości pierwszorzędnymi obywatelami w PyTorch.

### Atencja różnicowa (DIFF Transformer, 2024)

Zwykła atencja ma problem "zlewu atencji": softmax wymusza, by każdy wiersz sumował się do 1, więc tokeny, które nie chcą zwracać uwagi na nic konkretnego, zrzucają wagę na pierwszy token (lub kilka pierwszych). To kradnie pojemność, która powinna iść na prawdziwą treść.

Atencja różnicowa naprawia to, obliczając **dwie** mapy atencji i odejmując:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

gdzie `λ` to uczony skalar (typowo 0.5–0.8). A1 łapie wagi prawdziwej treści; A2 łapie zlew. Odejmowanie anuluje zlew, realokując wagę do istotnych tokenów.

Zgłoszone wyniki (Microsoft 2024): 5–10% niższa perplexity, 1.5–2× dłuższy efektywny kontekst przy tej samej trenowanej długości, ostrzejsze wyszukiwanie igły w stogu siana.

### Porównanie wariantów

| Wariant | Obliczenia | Cache KV | Jakość vs pełna | Zastosowanie produkcyjne |
|---------|------------|----------|-----------------|--------------------------|
| Pełna atencja | O(N²) | O(N) na warstwę | punkt odniesienia | domyślna warstwa każdego modelu |
| SWA (okno 1024) | O(N·W) | O(W) na warstwę | -0.1 ppl, dobra z globalnymi warstwami | Gemma 2/3, Phi-3-Long |
| Lokalna + z przeskokiem rzadka | O(N·√N) | mieszana | podobna do SWA | rzadki transformer OpenAI, Longformer |
| BigBird (lokalna + globalna + losowa) | w przybliżeniu O(N) | mieszana | dorównuje pełnej przy 2× kontekście | wczesny długokontekstowy BERT |
| Natynna rzadka (DeepSeek-V3.2) | O(N · aktywny ułamek) | O(N) | w granicach 0.05 ppl | DeepSeek-V3.2, 2025 |
| Różnicowa | O(2·N²) | O(2N) | -5 do -10% ppl | DIFF Transformer, wczesne modele 2026 |

```figure
gqa-kv-sharing
```

## Zbuduj To

Zobacz `code/main.py`. Implementujemy porównywarkę masek przyczynowych, która pokazuje pełną, SWA, lokalną+z-przeskokiem i różnicową atencję obok siebie na zabawkowej sekwencji.

### Krok 1: pełna maska przyczynowa (punkt odniesienia)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Punkt odniesienia z Lekcji 07. Dolna trójkątna; zerowa waga powyżej przekątnej.

### Krok 2: maska przyczynowa z przesuwnym oknem

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

Jeden parametr — `window`. Dla `window >= n` odzyskujesz pełną przyczynową atencję. Dla `window = 1` każdy token zwraca uwagę tylko na siebie.

### Krok 3: maska rzadka lokalna + z przeskokiem

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Gęste lokalne okno plus co `stride`-ty token wstecz do początku sekwencji. Pole recepcyjne rośnie w krokach logarytmicznych z dodatkowymi warstwami.

### Krok 4: atencja różnicowa

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Dwa przebiegi atencji, odejmowanie z uczonym współczynnikiem mieszania. W kodzie porównujemy mapę cieplną zlewu atencji dla pojedynczej vs różnicowej i obserwujemy zapadanie się zlewu.

### Krok 5: rozmiary cache KV

Wypisz rozmiar cache na warstwę przy `N = 131072` dla każdego wariantu. SWA i warianty rzadkie spadają 10–100×. Różnicowa podwaja. Płać rachunek za pamięć świadomie.

## Użyj Tego

Wzorce produkcyjne w 2026:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 miesza SWA (okno=1024) i globalne warstwy w stosunku 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention w PyTorch 2.5+ przyjmuje funkcję maski:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

To kompiluje się do niestandardowego jądra Triton. W granicach 10% prędkości FlashAttention-3 dla typowych wzorców, a funkcja maski to wywoływalny Python.

**Kiedy wybrać które:**

- **Czysta pełna atencja** — każda warstwa do ~16K kontekstu, lub gdy jakość wyszukiwania jest najważniejsza.
- **SWA + globalna mieszanka** — długi kontekst (>32K), trenowanie i inferencja ograniczone pamięcią. Domyślne w 2026 powyżej 32K.
- **Rzadka atencja blokowa** — niestandardowe jądro, niestandardowy wzór. Zarezerwowane dla wyspecjalizowanych obciążeń (wyszukiwanie, audio).
- **Atencja różnicowa** — każde obciążenie, gdzie zanieczyszczenie zlewem atencji szkodzi (długokontekstowy RAG, igła w stogu siana).

## Dostarcz To

Zobacz `outputs/skill-attention-variant-picker.md`. Umiejętność dobiera topologię atencji dla nowego modelu, uwzględniając docelową długość kontekstu, wymagania wyszukiwania oraz profil obliczeniowy trenowania/inferencji.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Sprawdź, czy SWA przy `window=4` zeruje wszystko poza ostatnimi 4 tokenami na wiersz. Sprawdź, czy `window=n` odtwarza pełną przyczynową atencję bit po bicie.
2. **Średnie.** Zaimplementuj przyczynową SWA z `window=1024` na bazie projektu końcowego z Lekcji 07. Trenuj przez 1000 kroków na tinyshakespeare. O ile strata walidacyjna spada w porównaniu do pełnej atencji? O ile spada szczytowe zużycie pamięci?
3. **Trudne.** Zaimplementuj mieszankę warstw w stylu Gemma-3 5:1 (5 SWA, 1 globalna) w modelu projektu końcowego. Porównaj stratę, pamięć i jakość generacji względem bazowych czystej SWA i czystej globalnej przy dopasowanych parametrach.
4. **Trudne.** Zaimplementuj atencję różnicową z uczonym `λ` na głowicę. Trenuj na syntetycznym zadaniu wyszukiwania (jedna igła, 2000 dystraktorów). Zmierz dokładność wyszukiwania względem bazowej pojedynczej atencji przy dopasowanych parametrach.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|--------|-----------------|-----------------------|
| Atencja z przesuwnym oknem (SWA) | "Lokalna atencja" | Każde zapytanie zwraca uwagę na swoje ostatnie `W` tokenów; cache KV kurczy się do `O(W)`. |
| Efektywne pole recepcyjne | "Jak daleko wstecz model widzi" | W `L`-warstwowym stosie SWA z oknem `W`, do `L × W` tokenów. |
| Longformer / BigBird | "Lokalna + globalna + losowa" | Rzadkie wzorce z kilkoma zawsze-uwagowymi tokenami globalnymi; wczesne podejście do długiego kontekstu. |
| Natynna rzadka atencja | "Sztuczka jądrowa DeepSeek" | Ucz się rzadkości na poziomie bloków; pomiń zerowe bloki na poziomie jądra, zachowując jakość. |
| Atencja różnicowa | "Dwie mapy, jedna odejmuje" | DIFF Transformer: odejmij uczone `λ` razy drugą mapę atencji od pierwszej, by anulować zlewy atencji. |
| Zlew atencji | "Waga wycieka do tokena 0" | Normalizacja softmax wymusza sumowanie wierszy do 1; nieinformacyjne zapytania zrzucają wagę na pozycję 0. |
| FlexAttention | "Maska jako Python" | API PyTorch 2.5+, które kompiluje dowolne funkcje maski do jąder w kształcie FlashAttention. |
| Mieszanka typów warstw | "5:1 SWA do globalnej" | Przeplataj rzadkie i pełne warstwy atencji w stosie, by utrzymać jakość przy niższej pamięci. |

## Dalsza lektura

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — kanoniczny artykuł o przesuwnym oknie + globalnych tokenach.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — lokalna + globalna + losowa.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — wzór lokalny+z-przeskokiem OpenAI.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — mieszanka 1:1 SWA:globalna.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — mieszanka 5:1 z oknem=1024, która jest teraz podręcznikowym domyślnym.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — artykuł DIFF Transformer.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — atencja z uczoną rzadkością DeepSeek-V3.2.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — dokumentacja API dla wzorca maski jako funkcji w sekcji Użyj Tego.