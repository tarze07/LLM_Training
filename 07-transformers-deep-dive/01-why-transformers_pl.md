# Dlaczego Transformery — Problemy z RNN

> RNN przetwarzają tokeny jeden po drugim. Transformery przetwarzają wszystkie tokeny naraz. Ten jeden architektoniczny zakład zmienił każdą krzywą skalowania w głębokim uczeniu po 2017 roku.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## Problem

Przed 2017 rokiem każdy najnowocześniejszy model sekwencyjny na świecie — języka, tłumaczenia, mowy — był rekurencyjną siecią neuronową. LSTM i GRU wygrywały benchmarki tłumaczeniowe odpowiedniki ImageNet przez pół dekady. Były jedynym narzędziem, jakie ktokolwiek miał.

Miały trzy śmiertelne słabości. Sekwencyjne obliczenia oznaczały, że nie można było zrównoleglić wzdłuż osi czasu: token `t+1` potrzebuje stanu ukrytego z tokena `t`. Sekwencja 1024 tokenów oznaczała 1024 szeregowe kroki na GPU, które potrafi wykonać 1 000 000 operacji zmiennoprzecinkowych na cykl. Czas treningu rósł liniowo z długością sekwencji na sprzęcie zaprojektowanym do równoległości.

Zanikające gradienty oznaczały, że informacja sprzed 50 tokenów była już skompresowana przez 50 nieliniowości. Bramkowane jednostki rekurencyjne (LSTM, GRU) łagodziły ten problem, ale nigdy go nie wyeliminowały. Zależności dalekiego zasięgu — "książka, którą przeczytałem zeszłego lata w samolocie do Kioto była…" — regularnie zawodziły.

Stany ukryte o stałej szerokości oznaczały, że enkoder ściskał całą źródłową sekwencję do pojedynczego wektora, zanim dekoder cokolwiek zobaczył. Nie ma znaczenia, czy źródło ma 5, czy 500 tokenów; wąskie gardło ma ten sam kształt.

Artykuł z 2017 roku "Attention Is All You Need" zaproponował coś radykalnego: całkowicie porzucić rekurencję. Pozwolić każdej pozycji zwracać uwagę na każdą inną pozycję równolegle. Trenować w jednym dużym mnożeniu macierzy zamiast 1024 sekwencyjnych.

Wynik dominuje w każdej dziedzinie do 2026 roku. Język (GPT-5, Claude 4, Llama 4), widzenie (ViT, DINOv2, SAM 3), audio (Whisper), biologia (AlphaFold 3), robotyka (RT-2). Ten sam blok, różne wejścia.

## Koncepcja

![Obliczenia sekwencyjne RNN vs równoległa uwaga Transformera](../assets/rnn-vs-transformer.svg)

**Rekurencja jako wąskie gardło.** RNN oblicza `h_t = f(h_{t-1}, x_t)`. Każdy krok zależy od poprzedniego. Nie można obliczyć `h_5` przed `h_4`. Na nowoczesnych GPU z 10 000+ równoległymi rdzeniami marnuje to 99% krzemu na długiej sekwencji.

**Uwaga (attention) jako transmisja rozgłoszeniowa.** Samo-uwaga (self-attention) oblicza `output_i = sum_j(a_ij * v_j)` dla każdej pary `(i, j)` jednocześnie. Cała macierz uwagi N×N wypełnia się w jednym wsadowym mnożeniu macierzy. Żaden krok nie zależy od innego. GPU to uwielbiają.

**Przyspieszenie nie jest stałe.** To różnica między `O(N)` głębokością szeregową a `O(1)` głębokością szeregową. W praktyce transformery trenują 5–10× szybciej na epokę na tym samym sprzęcie przy N=512, a luka rośnie z długością sekwencji, aż do osiągnięcia `O(N²)` ściany pamięciowej uwagi (którą Flash Attention później naprawiło — patrz Lekcja 12).

**Koszt transformerów.** Pamięć uwagi skaluje się jako `O(N²)`. Dla kontekstu 2K — w porządku. Dla kontekstu 128K potrzebujesz przesuwnych okien, ekstrapolacji RoPE, kafelkowania Flash Attention lub wariantów liniowej uwagi. Rekurencja miała `O(N)` zarówno w czasie, jak i pamięci; transformery wymieniają czas na pamięć, a następnie odzyskują czas dzięki równoległości.

**Zmiana indukcyjnego uprzedzenia (inductive bias).** RNN zakładają lokalność i świeżość. Transformery nie zakładają niczego — każda para jest kandydatem do uwagi. Dlatego transformery potrzebują więcej danych, by dobrze się trenować, ale skalują się dalej, gdy już je mają. Chinchilla (2022) sformalizowała to: mając wystarczająco dużo tokenów, transformer zawsze pokona RNN o równej liczbie parametrów.

## Zbuduj To

Żadnej sieci neuronowej — symulujemy numerycznie główne wąskie gardło, abyś poczuł różnicę na swoim laptopie.

### Krok 1: zmierz głębokość szeregową

Zobacz `code/main.py`. Tworzymy dwie funkcje. Jedna koduje sekwencję jako łańcuch dodawań (szeregowy, jak RNN). Druga koduje ją jako redukcję równoległą (transmisja rozgłoszeniowa, jak uwaga). Ta sama matematyka, inny graf zależności.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Mierzymy czas obu na sekwencjach do 100 000 elementów. Wersja RNN jest O(N) i działa na pojedynczym potoku CPU. Nawet w czystym Pythonie, redukcja w stylu uwagi wygrywa przy długości ≥ 1 000, ponieważ `sum()` w Pythonie jest zaimplementowana w C i iteruje bez narzutu interpretera na krok.

### Krok 2: policz teoretyczne operacje

Oba algorytmy wykonują N dodawań. Różnica polega na *głębokości zależności*: ile operacji musi nastąpić sekwencyjnie, zanim następna może się rozpocząć. Głębokość RNN = N. Głębokość uwagi = log(N) przy redukcji drzewiastej lub 1 przy skanowaniu równoległym. Głębokość, a nie liczba operacji, decyduje o czasie na GPU.

### Krok 3: empiryczne skalowanie na długich sekwencjach

Wyświetlamy tabelę czasów, która uwidacznia lukę O(N). Na laptopie Mac z 2026 roku sekwencje poniżej 1 000 elementów są zbyt szybkie, by je zmierzyć. Sekwencje 100 000 wykazują czysty liniowy skan. Skaluj to do 16 384-tokenowego transformera z 12-warstwowym odpowiednikiem LSTM i zobaczysz, dlaczego czas treningu był blokerem w 2016 roku.

## Użyj Tego

Kiedy w 2026 roku nadal wybrać RNN:

| Sytuacja | Wybierz |
|-----------|--------|
| Wnioskowanie strumieniowe, jeden token na raz, stała pamięć | RNN lub model przestrzeni stanów (Mamba, RWKV) |
| Bardzo długie sekwencje (>1M tokenów) gdzie pamięć uwagi eksploduje | Liniowa uwaga, Mamba 2, Hyena |
| Urządzenie brzegowe bez akceleratora matmul | Głębokościowo-separowalny RNN wciąż wygrywa na FLOPs/wat |
| Wszystko inne (trening, wsadowe wnioskowanie, kontekst do 128K) | Transformer |

Modele przestrzeni stanów (SSM), takie jak Mamba, są zasadniczo RNN ze strukturalną parametryzacją, która daje im to, co najlepsze z obu światów: `O(N)` pamięć skanowania, równoległy trening przez selektywne skanowanie. Odtwarzają 90% jakości transformera z lepszym skalowaniem długiego kontekstu. W 2026 roku większość laboratoriów pogranicznych trenuje hybrydowe modele SSM+transformer (np. Jamba, Samba) — rekurencja nie umarła, jest komponentem.

## Dostarcz To

Zobacz `outputs/skill-architecture-picker.md`. Umiejętność wybiera architekturę dla nowego problemu sekwencyjnego, biorąc pod uwagę długość, przepustowość i ograniczenia budżetu treningowego. Powinna zawsze odmówić polecenia czystego RNN dla przebiegów treningowych powyżej 1B tokenów bez wskazania kompromisu.

## Ćwiczenia

1. **Łatwe.** Weź `rnn_style` z `code/main.py` i zastąp skalarny stan ukryty wektorem stanów ukrytych o długości 64. Zmierz ponownie. Jak bardzo rośnie narzut szeregowy wraz z wymiarem stanu ukrytego?
2. **Średnie.** Zaimplementuj równoległą sumę prefiksową (skan Hillis-Steele) w czystym Pythonie. Zweryfikuj, że daje takie same wyniki numeryczne jak skan szeregowy na długości 1024. Policz głębokość.
3. **Trudne.** Przeprowadź redukcję w stylu uwagi do PyTorch na GPU. Mierz czas, zmieniając długość sekwencji od 64 do 65 536. Narysuj wykres i wyjaśnij kształt krzywej.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|-----------------|-----------------------|
| Rekurencja (Recurrence) | "RNN są sekwencyjne" | Obliczenia, w których krok `t` zależy od kroku `t-1`, wymuszając szeregowe wykonanie wzdłuż osi czasu. |
| Głębokość szeregowa (Serial depth) | "Jak głęboki jest graf" | Najdłuższy łańcuch zależnych operacji; ogranicza czas ścienny nawet na nieskończonym sprzęcie. |
| Uwaga (Attention) | "Pozwala tokenom patrzeć na siebie" | Ważona suma `sum_j a_ij v_j`, gdzie `a_ij` pochodzi z wyniku podobieństwa między pozycjami i i j. |
| Okno kontekstowe (Context window) | "Ile model widzi" | Liczba pozycji, które warstwa uwagi może przyjąć jako wejście; koszt pamięci kwadratowej skaluje się tutaj. |
| Indukcyjne uprzedzenie (Inductive bias) | "Założenia wbudowane w architekturę" | Wcześniejsze przekonanie o tym, jak wyglądają dane; CNN zakładają niezmienniczość względem przesunięcia, RNN zakładają świeżość. |
| Model przestrzeni stanów (State-space model) | "RNN z algebrą w tle" | Rekurencja sparametryzowana do równoległego treningu poprzez strukturalne macierze przestrzeni stanów. |
| Kwadratowe wąskie gardło (Quadratic bottleneck) | "Dlaczego kontekst kosztuje tak dużo" | Pamięć uwagi = `O(N²)` w długości sekwencji; Flash Attention ukrywa stałe, nie skalowanie. |

## Dalsza Lektura

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — artykuł, który zabił rekurencję w głównonurtowym NLP.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — gdzie narodziła się uwaga, doklejona do RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — oryginalny artykuł LSTM, dla porządku.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — nowoczesna rekurencyjna odpowiedź na transformery.