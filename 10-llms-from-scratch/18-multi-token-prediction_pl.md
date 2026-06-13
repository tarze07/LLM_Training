# Wielotokenowe Przewidywanie (MTP)

> Każdy autoregresyjny LLM od GPT-2 do Llamy 3 trenuje na jednej stracie na pozycję: przewiduj następny token. DeepSeek-V3 dodał drugą stratę na pozycję: przewiduj token po tym. Dodatkowe 14B parametrów (w modelu 671B) zostało zdystylowanych z powrotem do głównego modelu przez przepływ gradientu, a wytrenowane głowice MTP zostały ponownie wykorzystane przy inferencji jako szkicowniki dekodowania spekulatywnego z 80% akceptacją. 1.8× przepustowości generacji przyszło za darmo. Ta lekcja buduje sekwencyjny moduł MTP z raportu technicznego DeepSeek, oblicza stratę i układ parametrów współdzielonej głowicy oraz wyjaśnia, dlaczego MTP utrzymuje łańcuch przyczynowy, podczas gdy oryginalny równoległy MTP Gloeckle et al. go zerwał.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## Learning Objectives

- Sformułuj cel treningowy MTP i wyprowadź łączną stratę na głębokościach przewidywania.
- Wyjaśnij różnicę między równoległymi głowicami MTP Gloeckle et al. (2024) a sekwencyjnymi modułami MTP DeepSeek-V3 i dlaczego sekwencyjny projekt zachowuje łańcuch przyczynowy.
- Oblicz narzut parametrów i pamięci dodawania modułów MTP do uruchomienia pre-treningowego.
- Zaimplementuj jeden moduł MTP od zera: współdzielony embedding, transformer na głębokość, projekcję i współdzieloną głowicę wyjściową.

## Problem

Przewidywanie następnego tokena to standardowy cel treningowy LLM. Każdy ukryty stan jest nadzorowany, aby przewidzieć dokładnie jedną rzecz: bezpośrednio następujący token. To zaskakująco słaby sygnał. Większość informacji w sekwencji wykracza poza jeden token — struktura, spójność, faktyczność, przepływ arytmetyczny. Model musi nauczyć się ich, akumulując wiele jedno-tokenowych sygnałów przez biliony tokenów.

MTP pyta: co by było, gdyby każdy ukryty stan był nadzorowany, aby przewidywać wiele przyszłych tokenów naraz? Gloeckle et al. (Meta, 2024) pokazali, że to pomaga. Ich implementacja umieściła kilka niezależnych głowic wyjściowych na szczycie szkieletu, każdą przewidującą inne przesunięcie. Równoległe, proste, ale głowice widziały ten sam ukryty stan bez hierarchicznego uściślania — a przewidywania nie tworzyły łańcucha przyczynowego, więc nie mogły być użyte do dekodowania spekulatywnego.

DeepSeek-V3 (grudzień 2024) przeprojektował MTP jako sekwencyjne moduły, które utrzymują łańcuch przyczynowy na każdej głębokości przewidywania. Model przewiduje `t+1` z `h_i^(0)`, następnie przewiduje `t+2` z nowego ukrytego stanu `h_i^(1)`, który łączy `h_i^(0)` z embeddingiem `E(t+1)`, i tak dalej. Każda głębokość to własny mały transformer. Współdzielony embedding i współdzielona głowica wyjściowa utrzymują umiarkowany narzut parametrów. W skali DeepSeek-V3, 14B dodatkowych parametrów w modułach MTP na szczycie 671B wag głównego modelu. Ten 2% narzut kupił gęstsze sygnały treningowe ORAZ gotowy szkicownik dekodowania spekulatywnego przy inferencji.

Ta lekcja buduje pojedynczy moduł MTP i stratę D-głębokości od zera. Matematyka jest schludna. Implementacja to 150 linii.

## Koncepcja

### Przepis sekwencyjnego MTP

DeepSeek-V3 dodaje `D` modułów MTP na szczycie głównego modelu. Każdy moduł `k` (dla `k = 1..D`) przewiduje token na głębokości `k` — to znaczy `t_{i+k}` mając prefiks do pozycji `i`.

Moduł `k` składa się z:

- Bloku transformera `T_k` z własnym attention i MLP.
- Macierzy projekcji `M_k`, która łączy poprzedni ukryty stan z embeddingiem następnego tokena ground-truth na głębokości.
- Współdzielonego embeddingu `E` (tego samego co w głównym modelu).
- Współdzielonej głowicy wyjściowej `Out` (tej samej co w głównym modelu).

Podczas treningu, dla prefiksu do pozycji `i`, ukryty stan na głębokość to:

```
h_i^(0) = główny model backbone na pozycji i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   dla k >= 1
```

Przewidywanie na głębokość:

```
logits_{i+k} = Out(h_i^(k-1))   dla k = 1..D
```

Strata na głębokość to cross-entropy względem ground-truth `t_{i+k}`:

```
L_k = CE(logits_{i+k}, t_{i+k})
```

Łączna strata na głębokościach:

```
L_MTP = (lambda / D) * suma_{k=1..D} L_k
```

`lambda` to mały współczynnik wagowy — DeepSeek-V3 używa 0.3 przez pierwsze 10% treningu i 0.1 później. Całkowita strata treningowa to `L_main + L_MTP`.

### Dlaczego sekwencyjny, a nie równoległy

Oryginalny równoległy MTP Gloeckle'a miał D głowic wyjściowych, każdą bezpośrednio zastosowaną do `h_i^(0)`. Każda głowica przewiduje `t_{i+k}` z tego samego ukrytego stanu szkieletu. To trenuje dobrze, ale przewidywania nie są warunkowane na sobie nawzajem. Nie możesz użyć wyjścia `head_1` do pomocy `head_2` — głowice działają równolegle.

Sekwencyjny projekt DeepSeek-V3 buduje `h_i^(k)` z `h_i^(k-1)` plus embedding rzeczywistego następnego tokena `E(t_{i+k})`. To zachowuje łańcuch przyczynowy: aby przewidzieć `t_{i+k+1}`, moduł na głębokości `k+1` widzi, co było na `t_{i+k}`. Jest to strukturalnie identyczne z tym, jak autoregresyjny dekoder konsumuje własne wyjście — czyniąc moduły MTP bezpośrednio użytecznymi jako szkicowniki dekodowania spekulatywnego.

Przy inferencji: podaj `h_i^(k-1)` i wyszicowany `t_{i+k}` do modułu `k+1`, uzyskaj przewidywanie dla `t_{i+k+1}`. Powtórz. To jest dokładnie szkic w stylu EAGLE, używający wytrenowanego modułu MTP jako sieci szkicowej. DeepSeek-V3 zgłasza 80%+ akceptacji na pierwszym module MTP i ~1.8× przyspieszenia.

### Księgowość parametrów

Dla modelu z ukrytym `h` i słownikiem `V`:

- Główny model: miliardy parametrów, plus jedna głowica wyjściowa rozmiaru `V * h`.
- Współdzielona głowica wyjściowa: ponowne użycie głowicy głównego modelu. Zero dodatkowych parametrów.
- Współdzielony embedding: ponowne użycie embeddingu głównego modelu. Zero dodatkowych parametrów.
- Na moduł MTP:
  - Projekcja `M_k`: `(2h) * h = 2h^2`.
  - Blok transformera `T_k`: attention (`4h^2` dla MHA) plus MLP (zazwyczaj `8h^2` dla SwiGLU ze stosunkiem 8/3). Około `12h^2` na blok.

Łącznie dodatkowe na moduł: `~14h^2`. Dla DeepSeek-V3 `h = 7168`, D = 1 moduł: `~14 * 7168^2 = ~720M` parametrów na papierze. DeepSeek-V3 zgłasza 14B — różnica wynika głównie z tego, że warstwy eksperckie są MoE również w module MTP.

### Korzyść dekodowania spekulatywnego

Podczas pre-treningu, moduły MTP spowalniają trening o około 10% (więcej forwardowych obliczeń, dodatkowa strata). Korzyść jest dwojaka:

1. Gęstszy sygnał treningowy. Każdy ukryty stan widzi D+1 celów nadzoru. Zmierzony wpływ na MMLU, GSM8K, MATH, HumanEval: spójne kilkuprocentowe poprawy w ablacjach DeepSeek-V3.

2. Darmowy szkic dekodowania spekulatywnego przy inferencji. Moduł MTP jest już wytrenowany do przewidywania kilku następnych tokenów. Ponownie użyty jako sieć szkicowa, dostarcza 80%+ wskaźników akceptacji. Na tym poziomie, N=3 lub N=5 w dekodowaniu spekulatywnym daje 1.8× przepustowości. 10% kosztu w czasie treningu zwraca się przy pierwszym uruchomieniu inferencji.

### Relacja do EAGLE

EAGLE trenuje mały model szkicowy OSOBNIE po pre-treningu. MTP wbudowuje szkic w pre-trening. Oba podejścia zbiegają się do podobnych wskaźników akceptacji, ale przez różne potoki:

| Dimension | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| Kiedy trenowany | Po pre-treningu | Podczas pre-treningu |
| Wstecznie kompatybilny z istniejącymi wagami | Tak | Nie (potrzebny retrening) |
| Parametry szkicu | 1-2 warstwy transformera | 1 blok transformera + projekcja |
| Wskaźnik akceptacji | 0.88-0.92 | 0.80+ na głębokości 1 |
| Korzyść poza przyspieszeniem | Tylko dekodowanie spekulatywne | Gęstszy sygnał treningowy + przyspieszenie |

## Build It

`code/main.py` buduje pojedynczy moduł MTP end-to-end: współdzielony embedding, projekcję, blok transformera, współdzieloną głowicę wyjściową. Następnie oblicza stratę cross-entropy na głębokość na krótkiej syntetycznej sekwencji i wypisuje liczbę parametrów na komponent. Zabawkowy słownik 32 tokenów utrzymuje liczby czytelnymi.

### Krok 1: współdzielona tabela embeddingu

Pojedyncza tabela `vocab_size x hidden` jest używana przez główny model ORAZ przez każdy moduł MTP na każdej głębokości. Nie druga kopia — dosłownie ten sam tensor.

### Krok 2: połączenie na głębokość

```python
def combine(prev_hidden, next_token_embed, M_k):
    # konkatenacja wzdłuż wymiaru cech, następnie projekcja w dół do hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # namiastka dodawania wektorów
    projected = matvec(M_k, concat)
    return projected
```

Prawdziwy DeepSeek-V3 konkatenuje dwa wektory po RMSNorm do `[2h]` i projektuje za pomocą macierzy `h x 2h`. Zabawka używa dodawania wektorów dla zwięzłości stdlib.

### Krok 3: blok transformera na głębokości k

Self-attention plus MLP. W zabawce, jednowarstwowy liniowy blok attention i MLP SwiGLU utrzymują widoczną strukturę bez numpy.

### Krok 4: współdzielona głowica wyjściowa

Ponowne użycie projekcji wyjściowej głównego modelu. Logits nad słownikiem.

### Krok 5: strata na głębokość

Cross-entropia softmax(logits) względem tokena ground-truth na przesunięciu `k`. Zagregowana na głębokościach ze współczynnikiem skalowania `lambda / D`.

### Krok 6: księgowość parametrów

Wypisz całkowitą liczbę parametrów, liczbę współdzielonych (embedding, głowica) i dodatkową liczbę na moduł. Pokaż stosunek dodatkowego MTP do rozmiaru głównego modelu.

## Use It

MTP jest zintegrowane z DeepSeek-V3 (grudzień 2024) i serią DeepSeek-R1. Przy inferencji:

- Własny stos serwowania DeepSeek konsumuje moduły MTP jako dekodery spekulatywne po wyjęciu z pudełka.
- vLLM i SGLang mają ścieżki integracji dla DeepSeek-V3 MTP od kwietnia 2026.
- Samouczek SGLang AMD ROCm pokazuje konkretną konfigurację MTP dekodowania spekulatywnego ze zmierzonym 1.8× przyspieszeniem na checkpointie V3.

Kiedy użyć MTP w nowym uruchomieniu pre-treningowym:

- Kontrolujesz pełny potok pre-treningowy i chcesz zakumulować gęstszy sygnał treningowy.
- Wiesz, że będziesz serwować model na dużą skalę i chcesz darmowe dekodowanie spekulatywne.
- Twój ukryty rozmiar wynosi co najmniej 4096. W skali 1B narzut boli bardziej niż korzyść pomaga.

Kiedy nie:

- Fine-tuning istniejącego wstępnie wytrenowanego gęstego modelu. Moduł MTP nie jest wytrenowany.
- Modele badawcze, gdzie chcesz czystą linię bazową do porównania. MTP zmienia architekturę.

## Ship It

Ta lekcja produkuje `outputs/skill-mtp-planner.md`. Mając specyfikację uruchomienia pre-treningowego (rozmiar modelu, dane, obliczenia), zwraca plan integracji MTP: liczbę głębokości D, harmonogram `lambda`, narzut pamięciowy i okablowanie dekodowania spekulatywnego w czasie inferencji.

## Ćwiczenia

1. Uruchom `code/main.py`. Pokaż, że strata na głębokość maleje monotonicznie w miarę wzmacniania syntetycznego sygnału. Zmodyfikuj syntetyk, aby używał stałego wzorca i zweryfikuj, że zarówno strata głębokości-1, jak i głębokości-2 zbiegają się.

2. Oblicz narzut parametrów dla gęstego modelu 70B (hidden 8192, 80 warstw) z D=1 modułem MTP. Porównaj z zgłoszonym przez DeepSeek-V3 narzutem 14B. Wyjaśnij, dlaczego liczba DeepSeek jest wyższa: blok transformera MTP dziedziczy tę samą strukturę MoE, zawyżając liczbę parametrów na moduł.

3. Zaimplementuj D=2 w zabawce: dodaj drugi moduł MTP, który przyjmuje h^(1) i przewiduje `t_{i+2}`. Zweryfikuj łączną stratę i księgowość parametrów zgodnie z równaniami 19-21 pracy DeepSeek.

4. Przełącz zabawkę na równoległy MTP (styl Gloeckle'a): dodaj D głowic wyjściowych na szczycie głównego ukrytego stanu, każdą przewidującą inne przesunięcie. Zmierz, jak straty na głębokość porównują się do wersji sekwencyjnej na tym samym syntetycznym sygnale. Wersja sekwencyjna powinna produkować niższą stratę na głębokości k dla k > 1, ponieważ warunkuje się na pośrednich przewidywaniach.

5. Użyj wytrenowanego modułu MTP jako szkicu w stylu EAGLE: wywołaj moduł k, aby zaproponować `t_{i+k}` przy inferencji. Zmierz wskaźnik akceptacji tych tokenów szkicu względem przewidywań głównego modelu na wstrzymanej sekwencji. Jeśli osiągniesz 50%+ na zabawce, odtworzyłeś empiryczną właściwość MTP-jako-szkicu.

## Kluczowe Pojęcia

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MTP module | \"Dodatkowy blok straty\" | Mały blok transformera plus projekcja, który przewiduje token `k` pozycji do przodu względem głównego modelu |
| Prediction depth | \"Które przesunięcie\" | Liczba całkowita `k` taka, że moduł `k` przewiduje `t_{i+k}` z prefiksu do pozycji `i` |
| Parallel MTP | \"Styl Gloeckle'a\" | D niezależnych głowic na tym samym ukrytym stanie szkieletu, bez łańcucha warunkowania |
| Sequential MTP | \"Styl DeepSeek-V3\" | Każdy moduł warunkuje się na ukrytym stanie poprzedniej głębokości plus embeddingu następnego tokena; zachowuje łańcuch przyczynowy |
| Shared output head | \"Ponowne użycie głowicy głównego modelu\" | Moduły MTP wywołują głowicę LM głównego modelu, a nie osobną projekcję wyjściową |
| Shared embedding | \"Ponowne użycie tabeli głównego modelu\" | Ta sama tabela embeddingu słownika jest używana wszędzie; brak duplikacji parametrów |
| Projection matrix M_k | \"Połącz ukryty + następny token\" | Liniowa warstwa `h x 2h`, która składa poprzedni ukryty stan i embedding tokena docelowego w wejście następnej głębokości |
| Joint loss L_MTP | \"Uśrednione dodatkowe straty\" | Średnia arytmetyczna strat cross-entropy na głębokość, skalowana przez `lambda` |
| Acceptance rate at depth 1 | \"Jak często szkic MTP ma rację\" | Częstość, z jaką top-1 przewidywanie modułu MTP D=1 równa się top-1 przewidywaniu głównego modelu; 80%+ na DeepSeek-V3 |
| Lambda weighting | \"Ważność dodatkowej straty\" | Współczynnik skalowania na głębokość; 0.3 na początku treningu, 0.1 później w DeepSeek-V3 |

## Dalsza Lektura

- [DeepSeek-AI -- DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) -- pełny opis sekwencyjnego MTP (Sekcja 2.2), w tym równania łącznej straty i 1.8× przyspieszenia przy inferencji
- [Gloeckle et al. -- Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) -- równoległa linia bazowa MTP, którą projekt DeepSeek ulepsza
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) -- 685B łącznie (671B główny + 14B MTP), notatki dotyczące wdrożenia
- [Leviathan et al. -- Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) -- framework dekodowania spekulatywnego, w który wpisuje się MTP
- [Li et al. -- EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) -- architektura szkicu EAGLE z 2025, odpowiednik, z którym konkuruje MTP