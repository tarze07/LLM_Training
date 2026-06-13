# Jamba — Hybryda SSM-Transformer

> Modele przestrzeni stanów (SSM) i transformery chcą różnych rzeczy. Transformery kupują jakość poprzez attention za kwadratowy koszt. SSM kupują wnioskowanie w czasie liniowym i stałą pamięć poprzez rekurencję, ale tracą na jakości. Jamba AI21 (marzec 2024) i Jamba 1.5 (sierpień 2024) łączą je w jednym modelu: 1 warstwa transformera na każde 7 warstw Mamba, MoE na co drugim bloku i okno kontekstu 256k mieszczące się na pojedynczej karcie GPU 80GB. Mamba-3 (ICLR 2026) uszczelnia stronę SSM za pomocą zespolonych przestrzeni stanów i projekcji MIMO. Ta lekcja czyta obie architektury od początku do końca i wyjaśnia, dlaczego przepis hybrydowy przetrwał trzy lata skalowania, podczas gdy czyste SSM i czyste transformery w przypadku długiego kontekstu nie.

**Type:** Learn
**Languages:** Python (stdlib, kalkulator mieszania warstw)
**Prerequisites:** Phase 10 · 14 (architektury otwartych modeli), Phase 10 · 17 (natywna rzadka attention)
**Time:** ~60 minut

## Cele nauczania

- Wyjaśnić trzy prymitywy w bloku Jamba — warstwy transformera, warstwy Mamba, MoE — i przepis przeplatania 1:7:co druga.
- Opisać, jak na wysokim poziomie wygląda rekurencja SSM i dlaczego umożliwia wnioskowanie ze stałą pamięcią.
- Obliczyć rozmiar pamięci podręcznej KV modelu Jamba przy kontekście 256k i porównać z tym, czego potrzebowałby czysty transformer.
- Wymienić trzy innowacje Mamba-3 (dyskretyzacja wykładniczo-trapezowa, zespolona aktualizacja stanu, MIMO) i problem, który każda z nich rozwiązuje.

## Problem

Attention jest kwadratowa względem długości sekwencji. Modele przestrzeni stanów są liniowe. Ta różnica się kumuluje: przy 256k tokenach mapa attention transformera to 65B wpisów na głowicę; stan rekurencyjny SSM ma stały rozmiar niezależnie od długości sekwencji.

Czyste modele SSM (Mamba, Mamba-2) dorównują perplexity transformera w małych skalach, ale pozostają w tyle w zadaniach śledzenia stanu i zawodzą w niektórych kategoriach wyszukiwania w kontekście. Intuicja: SSM kompresują historię do stałego stanu, a gdy historia jest długa, informacje wyciekają. Attention pamięta wszystko dokładnie, ale płaci kwadratowy koszt.

Oczywiste rozwiązanie: użyj obu. Umieść warstwy transformera tam, gdzie dokładne przywołanie ma znaczenie. Użyj warstw SSM gdzie indziej. Dostrój proporcję. Jamba to pierwszy produkcyjny model, który dostarcza ten hybrydowy przepis w skali (52B łącznie, 12B aktywnych, kontekst 256k, pojedyncze GPU 80GB). Jamba 1.5 rozszerza rodzinę do 398B łącznie / 94B aktywnych. Mamba-3 (ICLR 2026) to obecnie najlepsza czysta linia bazowa SSM, wokół której można odbudowywać hybrydy.

Ta lekcja czyta wszystkie trzy artykuły i tworzy model mentalny „wybierz odpowiednią proporcję".

## Koncepcja

### SSM na jednej stronie

Model przestrzeni stanów przetwarza sekwencję `x_1, ..., x_N` poprzez stan `h` o stałym rozmiarze:

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

Na każdym kroku stan ewoluuje poprzez liniową dynamikę `A`, przyjmuje wejście `B x_t` i emituje wyjście `C h_t`. `A, B, C` mogą być uczone. Zwróć uwagę na kluczową właściwość: obliczenie `y_t` potrzebuje tylko `h_{t-1}` i `x_t`, a nie żadnego wcześniejszego `x`. Pamięć jest stała. Wnioskowanie jest O(1) na token.

Sztuczką dla jakości modelowania jest struktura `A`. S4 (Gu 2021) używał wysoce ustrukturyzowanej macierzy, którą można było efektywnie obliczyć jako długą konwolucję podczas trenowania. Mamba (Gu, Dao 2023) zastąpiła stałe `A, B, C` zależnymi od danych (część „selektywna"). Mamba-2 (2024) dodatkowo uprościła strukturę. Mamba-3 (2026) dodaje złożoność w określonych miejscach.

Kluczowa właściwość: dla dekodera LLM, warstwa SSM jest zamiennikiem warstwy attention, ze stanem o stałym rozmiarze na warstwę zamiast rosnącej pamięci podręcznej KV.

### Blok Jamba

Blok Jamba przeplata warstwy według dwóch liczb:

- `l`: stosunek attention do Mamba. Jamba używa `l = 8`, co oznacza 1 warstwę transformera na każde 7 warstw Mamba (7 Mamba + 1 Attention = 8 warstw na grupę).
- `e`: częstotliwość MoE. Jamba używa `e = 2`, co oznacza, że co druga warstwa stosuje MoE.

Sekwencja warstw w bloku:

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (gdzie | oznacza zastosowanie MoE)
```

Każdy blok Jamba to 8 warstw. Przy głębokości 4 bloków (32 warstwy łącznie) otrzymujesz 28 Mamba i 4 warstwy Attention. 16 z nich używa MoE.

### Dlaczego proporcja 1:7

AI21 przeprowadziło ablacje: jaka proporcja attention do Mamba daje najlepszą perplexity-na-parametr ORAZ wyszukiwanie w kontekście w ich ewaluacjach długiego kontekstu?

- Zbyt dużo attention (1:1): jakość rośnie, ale pamięć i prędkość spadają.
- Zbyt mało attention (1:15): pamięć jest świetna, ale wyszukiwanie w kontekście zawodzi.
- Punkt optymalny: 1:7 lub 1:8.

Intuicja: warstwy transformera obsługują dokładne przywołanie i śledzenie stanu. Warstwy Mamba obsługują tanie, masowe przetwarzanie.

### Kodowanie pozycyjne

Warstwy Mamba są same w sobie świadome pozycji (poprzez rekurencję). Warstwy attention w oryginalnych hybrydach opartych na Mambie nie używały RoPE — warstwy SSM dostarczały informacji o pozycji. Jamba 1.5 dodaje RoPE do warstw attention dla generalizacji na dłuższe konteksty, co jest udoskonaleniem post-hoc opartym na empirycznej ewaluacji długiego kontekstu.

### Budżet pamięci

Dla kształtu Jamba-1 (32 warstwy: 28 Mamba + 4 Attention, hidden 4096, 32 głowice attention):

- Pamięć podręczna KV (tylko warstwy attention): `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` przy 256k BF16. Tylko 4 warstwy attention się przyczyniają.
- Stan SSM: `28 * hidden * state_size` na prefiks tokena, ale jest to stały rozmiar na warstwę, nie skalujący się z długością sekwencji. Typowy stan Mamba to 16 na cechę, hidden 4096: `28 * 4096 * 16 * 2 = 3.7 MB` łącznie.

Porównaj z czystym transformerem przy 32 warstwach, tym samym hidden, pełnym MHA przy 32 głowicach: `2 * 32 * 32 * 128 * 256k * 2 = 128 GB` przy 256k BF16. 8-krotna redukcja pamięci podręcznej KV. Nawet w porównaniu z linią bazową GQA(8), której większość modeli z 2024 używa (`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`), hybryda Jamba 1:7 przy 16 GB jest wciąż 2 razy mniejsza.

To właśnie AI21 ma na myśli przez „kontekst 256k na pojedynczej karcie GPU 80GB." Pamięć podręczna KV czystego transformera z pełnym MHA by się nie zmieściła; nawet linia bazowa GQA nie zostawia miejsca na wagi i aktywacje; Jamba tak.

### Mamba-3: czysta linia bazowa SSM w 2026

Mamba-3 (ICLR 2026, arXiv:2603.15569) wprowadza trzy innowacje po stronie czystego SSM:

1. **Dyskretyzacja wykładniczo-trapezowa.** Zastępuje dyskretyzację metodą Eulera z Mamba-2 bardziej ekspresyjną rekurencją. Operacja podobna do konwolucji zastosowana na stanie-wejściu w ramach podstawowej rekurencji, a nie jako zewnętrzna konwolucja na `x_t`.

2. **Zespolona aktualizacja stanu.** Poprzednie Mamby zredukowały macierz stanu z zespolonej (S4) do rzeczywistej diagonalnej (Mamba) do skalowanej jednostkowej (Mamba-2). Mamba-3 dodaje z powrotem wartości zespolone — odpowiednik zależnego od danych obrotowego osadzenia na stanie. Przywraca to zdolności śledzenia stanu, które kosztowały poprzednie uproszczenia na wartościach rzeczywistych.

3. **Projekcje wielowejściowe wielowyjściowe (MIMO).** Zamiast skalarnych projekcji na cechę, używa projekcji macierzowych. Poprawia moc modelowania i wykorzystanie sprzętu podczas wnioskowania bez zwiększania opóźnienia dekodowania.

Przy 1.5B parametrów, Mamba-3 poprawia średnią dokładność downstream o 0.6 punktu w stosunku do Gated DeltaNet; wariant MIMO dodaje 1.2 więcej, co daje łączny zysk 1.8 punktu. Przy tym samym rozmiarze stanu, Mamba-3 dorównuje Mamba-2 z połową stanu.

Mamba-3 nie jest jeszcze dostarczana w produkcyjnej hybrydzie na dużą skalę — ale jest oczywistym kandydatem na stronę SSM następnego modelu klasy Jamba.

### Kiedy sięgać po hybrydę

Hybrydy wygrywają, gdy:

- Kontekst jest wystarczająco długi, aby pamięć podręczna KV czystego transformera stała się bolesna (64k+).
- Zadania mieszają strukturę krótkiego zasięgu (dobra dla SSM) z przywołaniem długiego zasięgu (potrzebuje transformera).
- Chcesz wdrożyć na budżecie pamięci pojedynczego GPU, gdzie sama pamięć podręczna KV transformera by się nie zmieściła.

Hybrydy przegrywają, gdy:

- Kontekst jest krótki (poniżej 16k). Narzut SSM jest marnowany; czysty transformer jest w porządku.
- Zadania wymagają attention wszędzie-do-wszędzie (głębokie rozumowanie, krzyżowe odniesienia między dokumentami). Rzadkość warstw attention w hybrydzie szkodzi.
- Skalujesz do granicznych modeli o bilionach parametrów. Czysty transformer + MLA + MoE (styl DeepSeek-V3) obecnie wygrywa wyścig zdolności.

### Krajobraz konkurencyjny

| Model | Rodzina | Skala | Unikalne twierdzenie |
|-------|--------|------|-------------|
| Mamba-2 | czysty SSM | 3B | czas liniowy, stała pamięć |
| Jamba | hybryda | 52B/12B | 256k na 80GB |
| Jamba 1.5 Large | hybryda | 398B/94B | długi kontekst klasy korporacyjnej |
| Mamba-3 | czysty SSM | 1.5B (artykuł) | przywrócone śledzenie stanu |
| DeepSeek-V3 | czysty transformer + MoE | 671B/37B | zdolność graniczna |

Krajobraz 2026: czysty transformer MoE dominuje na granicy, ale hybrydy posiadają niszę kontekstu 256k+. Zwycięstwa Mamba-3 w śledzeniu stanu mogą przesunąć proporcje hybryd w dół (więcej SSM, mniej attention) w następnej generacji.

```figure
swiglu-ffn
```

## Użyj

`code/main.py` to kalkulator pamięci dla architektur hybrydowych. Dla danej proporcji SSM-Transformer i konfiguracji hidden-size / liczba-warstw oblicza:

- Pamięć podręczną KV przy docelowym kontekście.
- Pamięć stanu SSM.
- Całkowitą pamięć przy kontekście N dla zakresu kształtów modelu.

Kalkulator obsługuje:

- Linię bazową czystego transformera (pamięć podręczna KV rośnie z N).
- Hybrydę w stylu Jamba 1:7.
- Czysty SSM (brak pamięci podręcznej KV).

Liczby pochodzą bezpośrednio z artykułów Jamba-1 i Jamba-1.5 dla opublikowanych kształtów i są ekstrapolowane dla hipotetycznych wariantów.

Uwagi dotyczące integracji dla rzeczywistego wdrożenia:

- Większość produkcyjnych serwerów wnioskowania (vLLM, SGLang) obsługuje Jamba i Mambę. Sprawdź konkretną wersję.
- Przy kontekście 256k przewaga pamięciowa Jamba objawia się w przepustowości równoczesnych żądań. Na tym samym VRAMie mieścisz więcej sekwencji Jamba niż transformera.
- Mamba-3 jako samodzielny model nie jest jeszcze dostarczana produkcyjnie — podgląd badawczy przy 1.5B.

## Dostarcz

Ta lekcja produkuje `outputs/skill-hybrid-picker.md`. Dla danej specyfikacji obciążenia (profil długości kontekstu, mieszanka zadań, budżet pamięci) zaleca między czystym transformerem, hybrydą w stylu Jamba a czystym SSM, z wyraźnym uzasadnieniem kompromisów pamięciowych i jakościowych.

## Ćwiczenia

1. Uruchom `code/main.py`, aby obliczyć pamięć podręczną KV przy kontekście 256k dla 32-warstwowego czystego transformera (hidden 4096, 32 głowice) i dla hybrydy Jamba-1 o tym samym kształcie. Zweryfikuj ~8-krotną redukcję pamięci, którą twierdzi artykuł AI21.

2. Zmodyfikuj kalkulator, aby modelował hybrydę 1:3 (4 Mamba : 1 Attention) i hybrydę 1:15 (14 Mamba : 1 Attention). Wykreśl pamięć podręczną KV w stosunku do proporcji. Przy jakiej proporcji pamięć podręczna KV równa się pamięci stanu SSM?

3. Przeczytaj Sekcję 3 artykułu Jamba (arXiv:2403.19887). Wyjaśnij, dlaczego AI21 używa Mamba-1 zamiast Mamba-2, mimo że Mamba-2 jest szybsza. Wskazówka: sekcja ablacji hybryd dokumentuje to.

4. Oblicz narzut parametrów MoE-co-drugą-warstwę w Jamba 1.5 Large (398B łącznie, 94B aktywnych). Porównaj stosunek aktywny do DeepSeek-V3 (37B/671B) i wyjaśnij, dlaczego architektura Jamba podnosi stosunek aktywny.

5. Przeczytaj Sekcję 3 artykułu Mamba-3 (arXiv:2603.15569). Wyjaśnij w trzech zdaniach, dlaczego zespolona aktualizacja stanu jest równoważna zależnemu od danych obrotowemu osadzeniu. Powiąż odpowiedź z wyprowadzeniem RoPE z Fazy 7 · Lekcji 04.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Model przestrzeni stanów (SSM) | „Rekurencja ze stałym stanem" | Warstwa z uczoną rekurencją `h_t = A h_{t-1} + B x_t`; stała pamięć na token |
| Selektywny SSM | „Sztuczka Mamby" | Zależne od danych parametry A, B, C, które dają modelowi selektywność podobną do bramkowania w czasie liniowym |
| Stosunek attention do Mamba | „Ile warstw attention" | W Jamba, `l = 8` oznacza 1 warstwę attention na 7 warstw Mamba |
| Blok Jamba | „8-warstwowa grupa" | Jedno attention + siedem Mamba + MoE na naprzemiennych pozycjach |
| Stan SSM | „Bufor ukryty" | Stan o stałym rozmiarze na warstwę, który zastępuje pamięć podręczną KV dla warstw Mamba |
| Kontekst 256k | „Flagowa liczba Jamba" | Długość sekwencji, którą Jamba-1 mieści na pojedynczej karcie GPU 80GB; czysty transformer nie może przy tym rozmiarze |
| Mamba-3 | „Czysty SSM 2026" | Obecnie najlepsza czysta architektura SSM z zespolonym stanem + MIMO; linia bazowa, wokół której odbudowuje się hybrydy |
| MIMO | „Multi-input multi-output" | Innowacja Mamba-3 używająca projekcji macierzowych zamiast skalarnych na cechę |
| Dyskretyzacja wykładniczo-trapezowa | „Rekurencja Mamba-3" | Bardziej ekspresyjna rekurencja, która zawiera w sobie dyskretyzację metodą Eulera z Mamba-2 |
| Architektura hybrydowa | „Mieszaj attention i SSM" | Dowolny model przeplatający warstwy transformera i SSM; Jamba jest produkcyjnym archetypem |

## Dalsze czytanie

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — oryginalny artykuł Jamba, ablacje proporcji, twierdzenie o kontekście 256k
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — rozszerzona rodzina, publiczne wydania 398B/94B i 12B/52B
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — artykuł o selektywnym SSM, na którym opiera się Jamba
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — uproszczony następca ustrukturyzowanej przestrzeni stanów
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — zespolony stan, MIMO, granica czystego SSM w 2026
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — artykuł S4, punkt startowy genealogii SSM dla LLM