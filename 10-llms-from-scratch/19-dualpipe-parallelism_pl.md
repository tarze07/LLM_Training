# DualPipe — Równoległość DualPipe

> DeepSeek-V3 został wytrenowany na 2 048 kartach H800 z ekspertami MoE rozproszonymi pomiędzy węzłami. Międzywęzłowa komunikacja all-to-all dla ekspertów kosztowała 1 godzinę GPU komunikacji na każdą 1 godzinę GPU obliczeń. Karty GPU były bezczynne przez połowę czasu. DualPipe (DeepSeek, grudzień 2024) to dwukierunkowy potok, który nakłada na siebie obliczenia w przód i wstecz z wywoływaną przez nie komunikacją all-to-all. Pęcherzyki (bubble) maleją, przepustowość rośnie, a utrzymywanie dwóch kopii parametrów modelu („dual" w nazwie) jest tanie, gdy Równoległość Ekspertów (Expert Parallelism) już rozprasza ekspertów pomiędzy rangami. Ta lekcja to opisowy (Learn) przegląd tego, co DualPipe faktycznie robi i dlaczego udoskonalenie DualPipeV z Sea AI Lab usuwa 2-krotny koszt parametrów kosztem nieznacznie większego pęcherzyka.

**Type:** Learn
**Languages:** Python (stdlib, symulator harmonogramu)
**Prerequisites:** Phase 10 · 05 (trenowanie rozproszone, FSDP, DeepSpeed), Phase 10 · 14 (architektury otwartych modeli i MoE)
**Time:** ~60 minut

## Cele nauczania

- Wymienić cztery komponenty fragmentu (chunk) forward-backward w DualPipe i uzasadnić, dlaczego każdy ma własne okno nakładania.
- Wyjaśnić problem pęcherzyka potoku w skali oraz co oznacza „bez pęcherzyka" w praktyce w porównaniu do marketingu.
- Ręcznie prześledzić harmonogram DualPipe dla 8 rang PP i 16 mikro-partii oraz potwierdzić, że strumienie w przód i w tył wypełniają nawzajem swoje wolne sloty.
- Określić kompromis DualPipeV (Sea AI Lab, 2025): usuwa 2-krotną replikację parametrów kosztem nieco większego pęcherzyka, gdy Równoległość Ekspertów jest nieaktywna.

## Problem

Trenowanie modelu MoE o 671B parametrach na 2k kartach H800 napotyka trzy kumulujące się wąskie gardła:

1. **Presja pamięci.** Każda karta GPU przechowuje wycinek modelu. Pamięć aktywacji przy sekwencji 8k przez 61 warstw na 128 głowicach jest ogromna.
2. **Pęcherzyki potoku.** Tradycyjna równoległość potokowa (GPipe, 1F1B) pozostawia karty GPU bezczynne, gdy czekają na wejście lub gradient swojego etapu. Przy 8 etapach około 12% czasu GPU może stanowić pęcherzyk nawet przy harmonogramie 1F1B.
3. **Międzywęzłowe all-to-all.** MoE z równoległością ekspertów rozprasza ekspertów pomiędzy węzłami. Każde przejście w przód wywołuje all-to-all w celu wysłania tokenów do ekspertów i kolejne w celu połączenia. Przy 2k kartach łatwo osiąga to stosunek obliczeń do komunikacji 1:1.

Każdy z tych problemów ma osobne rozwiązania: punkt kontrolny gradientu (gradient checkpointing) dla pamięci, Zero Bubble (Sea AI Lab, 2023) dla pęcherzyków potoku, jądra komunikacji równoległości ekspertów dla all-to-all. DualPipe sprawia, że działają one razem. Harmonogram nakłada obliczenia i komunikację w ramach pojedynczego fragmentu forward-backward, wstrzykuje mikro-partie z obu końców potoku jednocześnie i wykorzystuje powstały harmonogram do ukrycia all-to-all wewnątrz okien obliczeniowych.

Zgłoszony wynik: prawie całkowita eliminacja pęcherzyków potoku, ponad 95% wykorzystania GPU w 14,8-bilionowym trenowaniu DeepSeek-V3.

## Koncepcja

### Przypomnienie równoległości potokowej

Podziel model o N warstwach na P urządzeń. Urządzenie `i` przechowuje warstwy `i * N/P .. (i+1) * N/P - 1`. Mikro-partia przepływa w przód przez urządzenia 0 do P-1, a następnie wstecz od P-1 do 0. Każde urządzenie może rozpocząć swój etap w przód dopiero, gdy poprzednie urządzenie wyśle swoje wyjście, i może rozpocząć etap wstecz dopiero, gdy następne urządzenie wyśle gradient w górę.

GPipe (Huang i in., 2019) planuje jedną mikro-partię na raz, co marnuje większość czasu GPU. 1F1B (Narayanan i in., 2021) przeplata przejścia w przód i wstecz dla wielu mikro-partii. Zero Bubble (Qi i in., 2023) dzieli przejście wstecz na dwie części — wstecz-dla-wejścia (B) i wstecz-dla-wag (W) — i planuje je tak, aby wypełnić pęcherzyk. Po Zero Bubble potok jest prawie szczelny.

DualPipe to kolejny krok. Dodaje dwa pomysły:

### Pomysł 1: dekompozycja fragmentu

Każdy fragment w przód jest podzielony na cztery komponenty:

- **Attention.** Projekcje Q/K/V, attention, projekcja wyjściowa.
- **All-to-all dispatch.** Międzywęzłowa komunikacja wysyłająca tokeny do ich ekspertów.
- **MLP.** Obliczenia eksperta MoE.
- **All-to-all combine.** Międzywęzłowa komunikacja zwracająca wyniki ekspertów.

Fragment wstecz dodaje wersje gradientowe każdego z nich. DualPipe planuje je tak, aby all-to-all dispatch zachodził równolegle z obliczeniami attention następnego fragmentu, a all-to-all combine równolegle z obliczeniami MLP kolejnego fragmentu.

### Pomysł 2: planowanie dwukierunkowe

Większość harmonogramów potokowych wstrzykuje mikro-partie z etapu 0 i kieruje je w stronę etapu P-1. DualPipe wstrzykuje mikro-partie z OBU końców. Etap 0 widzi mikro-partie w przód pochodzące stamtąd; etap P-1 również widzi mikro-partie w przód pochodzące stamtąd. Dwa strumienie spotykają się w środku.

Aby to działało, urządzenie `i` musi przechowywać ZARÓWNO wczesną warstwę potoku `i`, JAK I późną warstwę potoku `P - 1 - i`. To jest część „dual" DualPipe: każde urządzenie przechowuje dwie kopie warstw modelu, które musi obsługiwać (po jednej dla każdego kierunku). W skali DeepSeek-V3 jest to 2-krotny koszt replikacji parametrów. Jest to opłacalne, ponieważ Równoległość Ekspertów już tak bardzo rozrzedza ekspertów MoE, że dwukrotna replikacja warstw niebędących ekspertami to drobnostka.

Co kluczowe, strumień w przód w jednym kierunku i strumień wstecz w drugim kierunku nakładają się dokładnie tam, gdzie w harmonogramie jednokierunkowym występowałyby pęcherzyki. Pęcherzyki znikają.

### Ręcznie prześledzony harmonogram

Rozważ P = 4 rangi, 8 mikro-partii, podzielone 4 w przód / 4 w tył. Czas płynie od lewej do prawej; wiersze to rangi urządzeń.

```
           Czas →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

Odczytywanie notacji „F4/F5R": rank 1 wykonuje forward mikro-partii 4 (idący od lewej do prawej w potoku) ORAZ forward mikro-partii 5 (idący od prawej do lewej) w tym samym slocie czasowym. To właśnie oznacza „dwukierunkowość" operacyjnie.

Na randze 2 strumienie krzyżowe nakładają się wcześniej, na randze 0 i P-1 nakładają się najpóźniej. W stabilnej fazie środkowej harmonogramu każda ranga wykonuje forward-w-kierunku-X nałożony na backward-w-kierunku-Y. Obliczenia są zajęte. All-to-all dispatch dla przejścia w przód ukrywa się wewnątrz obliczeń wstecz. All-to-all combine ukrywa się wewnątrz obliczeń w przód. Pęcherzyki są wyciśnięte.

### Rachunek pęcherzyka

Standardowy pęcherzyk potoku 1F1B (czas stracony na rangę):

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Udoskonalenie Zero Bubble zmniejsza go, ale nie do zera. DualPipe w stabilnej fazie ma zerowy pęcherzyk, jeśli liczba mikro-partii jest podzielna przez 2 razy głębokość potoku. Poza fazą stabilną (rozgrzewanie i schładzanie) występuje pewien pęcherzyk, ale nie rośnie on wraz z liczbą mikro-partii — kluczowa właściwość podkreślona w artykule.

W kategoriach marketingowych: „bez pęcherzyka". W kategoriach technicznych: pęcherzyki nie rosną wraz z liczbą mikro-partii. Analiza uzupełniająca Sea AI Lab (DualPipeV / Cut-in-half) pokazuje, że pełny zerowy pęcherzyk występuje tylko wtedy, gdy Równoległość Ekspertów nie jest wąskim gardłem; przy all-to-all napędzanym przez EP zawsze występuje pewien kompromis w planowaniu.

### DualPipeV — udoskonalenie

Sea AI Lab (2025) zauważyło, że 2-krotna replikacja parametrów jest marnotrawstwem, gdy nakładanie komunikacji EP nie jest celem. Ich harmonogram DualPipeV składa dwukierunkowe wstrzykiwanie w harmonogram w „kształt V", który działa na pojedynczej kopii parametrów. Pęcherzyk jest nieco większy niż w DualPipe, ale oszczędności pamięci są znaczące. DeepSeek przyjął DualPipeV w swojej otwartej implementacji DualPipe jako tryb z wyłączonym EP.

Kompromis:

| Cecha | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Kopie parametrów na urządzenie | 2 | 1 | 1 | 1 |
| Pęcherzyk vs mikro-partie | stały | mały wzrost | rośnie | rośnie |
| Nakładanie obliczeń-komunikacji | pełne | częściowe | minimalne | częściowe |
| Użyj gdy | EP-ciężkie MoE | gęsty lub lekki EP | linia bazowa | dowolny potok |

### Co to oznacza dla trenowania na 14,8T tokenów

Trenowanie wstępne DeepSeek-V3 zużyło 14,8T tokenów na 2 048 kartach H800 w około 2,8M godzin GPU. Przy naiwnym 1F1B straciliby 12-15% na pęcherzyki potoku — 340-420K godzin GPU, wystarczająco, by wytrenować pełny model 70B. DualPipe odzyskał większość tego. Bezpośrednie określenie wkładu jest trudne bez wewnętrznych logów, ale twierdzenie w artykule to ponad 95% średniego wykorzystania GPU podczas trenowania.

Dla mniejszych przebiegów (poniżej 1k GPU) DualPipe jest przesadą — pęcherzyki potoku są mniejsze w stosunku do całkowitego kosztu, a trenowanie gęstych modeli rzadko napotyka wąskie gardło all-to-all. Dla trenowania MoE na granicy możliwości w skali wielu tysięcy GPU jest to praktycznie wymagane.

### Miejsce w stosie

- Uzupełnia **FSDP** (Faza 10 · 05). FSDP dzieli parametry modelu pomiędzy rangi; DualPipe planuje obliczenia pomiędzy rangami. Łączą się.
- Kompatybilny z **ZeRO-3** dzieleniem gradientów. Księgowość dla dwukrotnej replikacji musi współpracować z dzielonymi gradientami ZeRO.
- Wymaga **niestandardowych jąder all-to-all** dostrojonych do konkretnej topologii klastra. Otwarte jądra DeepSeek są implementacją referencyjną.

```figure
expert-capacity
```

## Użyj

`code/main.py` to symulator harmonogramu potoku. Przyjmuje `(P, n_micro_batches, schedule)` i wypisuje wykorzystanie w fazie stabilnej dla każdego z 1F1B, Zero Bubble, DualPipe i DualPipeV. Jest narzędziem dydaktycznym — liczby odpowiadają jakościowym twierdzeniom z artykułów, nie są twierdzeniem o zmierzonym przyspieszeniu produkcyjnym.

Wartość symulatora: uruchom go z różnymi P i liczbami mikro-partii i obserwuj, jak frakcja pęcherzyka rośnie dla 1F1B, ale nie dla DualPipe.

Uwagi dotyczące integracji dla rzeczywistego trenowania:

- Wybierz głębokość równoległości potokowej, która dzieli się równo przez liczbę mikro-partii.
- Upewnij się, że siatka równoległości ekspertów obsługuje dwukierunkowe all-to-all. Jądra DeepSeek są referencją.
- Spodziewaj się spędzić tydzień na debugowaniu samego harmonogramu za pierwszym razem. Księgowość jest drobiazgowa.
- Monitoruj wykorzystanie GPU na rangę, nie tylko zagregowane. Korzyść DualPipe pochodzi z zaciskania najwolniejszych ogniw.

## Dostarcz

Ta lekcja produkuje `outputs/skill-dualpipe-planner.md`. Dla danej specyfikacji klastra treningowego (liczba GPU, topologia, połączenia, kształt modelu) zaleca strategię równoległości potokowej, algorytm planowania i oczekiwaną frakcję pęcherzyka w docelowej skali.

## Ćwiczenia

1. Uruchom `code/main.py` dla `(P=8, micro_batches=16, schedule=dualpipe)` i `(P=8, micro_batches=16, schedule=1f1b)`. Oblicz różnicę w wykorzystaniu GPU i wyraź ją jako odzyskane godziny GPU na milion tokenów trenowania.

2. Naszkicuj ręcznie tabelę harmonogramu dla `(P=4, micro_batches=8, schedule=dualpipe)`. Oznacz każdy slot czasowy ID mikro-partii i kierunkiem. Zidentyfikuj pierwszy slot czasowy, w którym pęcherzyki są nieobecne.

3. Przeczytaj Rysunek 5 raportu technicznego DeepSeek-V3 (arXiv:2412.19437). Zidentyfikuj okno nakładania dla all-to-all dispatch wewnątrz fragmentu forward DualPipe. Wyjaśnij, jak harmonogram obliczeń je ukrywa.

4. Oblicz 2-krotny narzut parametrów DualPipe dla gęstego modelu 70B z P=8 etapami potoku i modelu MoE 671B z P=16 etapami potoku. Pokaż, dlaczego narzut w przypadku MoE jest proporcjonalnie mniejszy (większość parametrów to eksperci, rozproszeni po dużej grupie EP).

5. Porównaj DualPipe z Chimerą (konkurencyjnym dwukierunkowym planistą z 2021). Zidentyfikuj dwie konkretne właściwości, które DualPipe dodał, a których Chimera nie miała, korzystając z Sekcji 3.4 artykułu jako referencji.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Pęcherzyk potoku (pipeline bubble) | „Czas bezczynności na rangę" | Cykle GPU zmarnowane, ponieważ etap potoku czeka na swoje wejście lub gradient |
| 1F1B | „Domyślny harmonogram potoku" | Jeden forward / jeden backward przeplatany harmonogram; linia bazowa, którą DualPipe pokonuje |
| Zero Bubble | „Sea AI Lab 2023" | Dzieli backward na B (gradient wejścia) i W (gradient wag); prawie całkowicie uszczelnia potok |
| DualPipe | „Harmonogram DeepSeek-V3" | Dwukierunkowy potok + nakładanie obliczeń i komunikacji; pęcherzyki nie rosną z liczbą mikro-partii |
| DualPipeV | „Cut-in-half" | Udoskonalenie w kształcie V, które usuwa 2-krotną replikację parametrów kosztem nieco większych pęcherzyków |
| Fragment (chunk) | „Jednostka pracy potoku" | Przejście w przód lub wstecz jednej mikro-partii przez jeden etap potoku |
| All-to-all dispatch | „Wyślij tokeny do ekspertów" | Międzywęzłowa komunikacja kierująca tokeny do przypisanych ekspertów MoE |
| All-to-all combine | „Przynieś wyniki ekspertów z powrotem" | Międzywęzłowa komunikacja zbierająca wyniki ekspertów po MLP |
| Równoległość Ekspertów (EP) | „Eksperci na GPU" | Dzieli ekspertów MoE pomiędzy rangi, tak że różne GPU przechowują różnych ekspertów |
| Równoległość Potokowa (PP) | „Warstwy na GPU" | Dzieli warstwy modelu pomiędzy rangi; wymiar, który planuje DualPipe |
| Frakcja pęcherzyka | „Zmarnowany czas GPU" | (bubble_time / total_time); frakcja, którą DualPipe dąży do zera |

## Dalsze czytanie

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — główne źródło DualPipe
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — otwarta implementacja referencyjna, w tym tryb DualPipeV (Cut-in-half)
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — poprzednik Zero Bubble
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — analiza DualPipeV, która zainspirowała tryb EP-off DeepSeek
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — harmonogram 1F1B, z którym porównuje się DualPipe
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — oryginalna praca o równoległości potokowej i problemie pęcherzyka