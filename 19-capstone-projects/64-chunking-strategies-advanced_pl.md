# Strategie Dzielenia na Fragmenty w Porównaniu

> Dzielenie na fragmenty decyduje o tym, co twój wyszukiwacz może kiedykolwiek wydobyć. Ustal granice źle, a żaden model osadzania, żaden reranker, żaden LLM nie naprawi szkód w dalszych etapach.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 11, lekcje 04 (osadzania), 06 (RAG), 07 (zaawansowany RAG); Faza 19, Track B foundations (lekcje 20-29)
**Czas:** ~90 minut

## Cele dydaktyczne
- Zaimplementować pięć strategii dzielenia na fragmenty od podstaw: stałe okno, zdanie, podział rekurencyjny, grupowanie semantyczne i strukturalne nagłówki markdown.
- Zmierzyć recall@k na korpusie testowym z oznaczonymi złotymi zakresami odpowiedzi i wyjaśnić, dlaczego jedna strategia wygrywa na prozie, a inna na dokumentach technicznych.
- Odczytać rozkład długości fragmentów i rozpoznać tryby awarii, które każda strategia wprowadza: osierocone zdania, cięcia w środku symbolu, fragmenty tylko-nagłówkowe, dryf semantyczny.
- Wybrać domyślną strategię dla nowego korpusu bez uruchamiania benchmarku, sprawdzając trzy właściwości: typ dokumentu, średnią długość akapitu i czy format przenosi jawną strukturę.

## Problem

Każdy potok RAG zaczyna się od cięcia dokumentów źródłowych na kawałki wystarczająco małe, aby model osadzania je zmieścił i wystarczająco duże, aby każdy kawałek niósł samodzielną myśl. Wybór miejsca cięcia nie jest hiperparametrem. Jest górną granicą tego, co wyszukiwacz może kiedykolwiek zwrócić.

Zapytanie "jak wygląda próg przerwania budżetu" może się udać tylko wtedy, gdy fragment zawierający próg przerwania jest osiągalny. Jeśli dzielnik stałego okna odciął wartość progu od otaczającego kontekstu, osadzenie przesuwa się do innego klastra, wynik BM25 spada, rerankery widzą szum, a odpowiedź generowana przez LLM jest błędna. Artykuł z 2024 "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs" zmierzył 35-procentową bezwzględną zmianę w recall wyszukiwania wynikającą wyłącznie z wyboru fragmentacji. Prace następcze z 2025 na kontekstowych nagłówkach fragmentów zmniejszyły lukę, ale jej nie zamknęły.

Ta lekcja buduje pięć strategii obok siebie, uruchamia je na korpusie testowym z oznaczonymi złotymi zakresami odpowiedzi i pozwala odczytać liczby recall samodzielnie.

## Koncepcja

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### Stałe okno

Brutalna linia bazowa. Tnij co N znaków. Opcjonalnie nakładaj, aby zdanie przecięte na pozycji N pojawiło się w całości wewnątrz fragmentu zaczynającego się na pozycji N - overlap. Szybkie, deterministyczne, fatalne na granicach. Użyj jako kontroli, nie jako domyślnego.

### Zdanie

Dziel na granicach zdań za pomocą regexa lub prostej maszyny stanów. Pakuj jedno lub więcej zdań do fragmentu aż do docelowego budżetu znaków. Przestaje ciąć w środku słowa. Nadal tnie w środku akapitu i w środku sekcji. Wartość domyślna w wielu wczesnych potokach RAG i rozsądny wybór dla prozy bez innej struktury.

### Podział rekurencyjny

Strategia hierarchiczna spopularyzowana przez biblioteki z ery 2023. Spróbuj podzielić na najsilniejszym separatorze (podwójny newline, akapit), spadając do następnego (pojedynczy newline), a następnie do zdań, a następnie do znaków. Rekursja kończy się, gdy fragment mieści się w budżecie. Silny na dokumentach, które mają niespójną strukturę, ponieważ adaptuje się na region.

### Grupowanie semantyczne

Osadź każde zdanie. Grupuj ciągłe zdania, które dzielą centroid tematyczny. Tnij, gdy bieżące podobieństwo do centroidu spada poniżej progu. Granice odzwierciedlają znaczenie, a nie znaki. Wolniejsze do zbudowania i zależne od modelu osadzania, ale odporne na dokumenty, które zmieniają temat wewnątrz akapitu.

### Strukturalne nagłówki markdown

Dla dokumentów, które przenoszą jawną strukturę (markdown, reStructuredText, sekcje numerowane w stylu RFC), tnij na granicach nagłówków. Każdy fragment staje się nagłówkiem plus wszystko pod nim aż do następnego nagłówka na tym samym lub wyższym poziomie. Najmniejsze fragmenty na temat, ale dostępne tylko wtedy, gdy korpus jest dobrze uformowany.

### Jak recall@k mierzy wybór granicy

Oznaczone zapytanie niesie dokładne przesunięcia znaków zakresu odpowiedzi wewnątrz dokumentu źródłowego. Po fragmentacji pytasz: czy któryś z top-k fragmentów zwróconych przez wyszukiwacz nakłada się na złoty zakres? Jeśli tak, recall@k dla tego zapytania wynosi 1. Jeśli nie, wynosi 0. Średnia po zestawie zapytań. Uruchom tę samą ewaluację dla każdej strategii, a rozrzut pokaże, która polityka granic przetrwa na twoim korpusie.

## Zbuduj to

`code/main.py` implementuje:

- `fixed_window(text, size, overlap)` - linia bazowa.
- `sentence_chunks(text, target)` - prosty pakowacz zdań.
- `recursive_split(text, separators, target)` - rekurencja hierarchiczna.
- `semantic_chunks(text, similarity_threshold)` - grupowanie oparte na centroidzie na wierzchu deterministycznego mockowego osadzenia.
- `structural_markdown(text)` - dzielnik świadomy nagłówków.
- `mock_embed(text, dim)` - osadzenie oparte na haszu, aby pętla działała offline.
- `DenseIndex` - ten sam kształt używany w lekcji wyszukiwania hybrydowego Track B Fazy 19.
- `eval_recall(strategy, corpus, queries, k)` - pętla porównawcza.
- `main()`, która uruchamia każdą strategię na korpusie testowym i wypisuje tabelę recall@k.

Uruchom:

```bash
python3 code/main.py
```

Wynik to mała tabela z jednym wierszem na strategię i jedną kolumną na k. Zdanie przegrywa na ustrukturyzowanym zestawie testowym. Strukturalny-markdown wygrywa na markdown. Rekurencyjny trzyma się dobrze na mieszanym zestawie, ponieważ rekursja się adaptuje. Grupowanie semantyczne wygrywa na prozie, gdzie nie ma użytecznych wskazówek strukturalnych.

## Tryby awarii, których tabela nie ukryje

**Osierocone zdania.** Pakowanie zdań produkuje fragmenty, które pomijają zdanie tematyczne. Osadzenie wskazuje wtedy na zły klaster.

**Cięcia w środku symboli.** Stałe okno wewnątrz kodu lub YAML podzieli identyfikator na pół. Dwie połówki osadzają się do szumu.

**Fragmenty tylko-nagłówkowe.** Strukturalny markdown emituje fragment zawierający nic poza `## Tytuł`. Odfiltruj je lub dołącz pierwszy akapit następnego fragmentu.

**Dryf semantyczny.** Grupowanie semantyczne za bardzo tnie, gdy korpus jest jednolicie na temat. 5000-znakowy fragment pakuje wiele konkretnych odpowiedzi w jedno rozproszone osadzenie. Połącz semantyczne z twardym limitem znaków.

**Nieaktualne osadzenia.** Grupowanie semantyczne używa modelu osadzania. Jeśli zmienisz model, zmieniasz również fragmenty. Przypnij model fragmentacji oddzielnie od modelu wyszukiwania lub przebuduj indeks razem.

## Wybór domyślnego bez uruchamiania benchmarku

Trzy właściwości decydują o domyślnym fragmentatorze dla nowego korpusu.

| Właściwość | Wartość | Domyślny |
|-----------|---------|----------|
| Typ dokumentu | Proza bez struktury | Podział rekurencyjny, cel 800 |
| Typ dokumentu | Markdown / RFC / API docs | Strukturalny markdown |
| Typ dokumentu | Kod | Świadomy AST (poza zakresem; patrz Faza 19, lekcja 02) |
| Długość akapitu | Długi, pojedynczy temat | Zdanie, cel 500 |
| Długość akapitu | Krótki, mieszane tematy | Semantyczny, próg 0.6 |

W razie wątpliwości wybierz podział rekurencyjny. To najsilniejsza pojedyncza linia bazowa.

## Użyj tego

Wzorce produkcyjne:

- Uruchom ewaluację, zanim wdrożysz nowy potok; nie ufaj strategii, do której domyślnie ustawia się twoja biblioteka.
- Uruchom ponownie ewaluację, gdy zmienisz model osadzania lub skład korpusu; zwycięzca zależy od korpusu.
- Utrwal nazwę strategii w metadanych każdego fragmentu, abyś mógł później przypisać regresje.

## Dostarcz to

System RAG end-to-end Toru F w lekcji 69 używa fragmentatora wybranego tutaj jako pierwszego etapu. Środowisko ewaluacyjne w lekcji 68 czyta recall@k z tego samego kształtu, który `eval_recall` zwraca w tej lekcji. Wybierz strategię, która wygrywa na twoim korpusie i przekaż ją dalej.

## Ćwiczenia

1. Dodaj szóstą strategię: okno tokenów używające `tiktoken` zamiast zliczania znaków. Porównaj ze stałym oknem na tym samym zestawie testowym.
2. Wstrzyknij 30-procentową frakcję bloków kodu do zestawu prozy. Uruchom ponownie tabelę. Wyjaśnij, dlaczego każda strategia oprócz strukturalnego markdown traci recall.
3. Zastąp deterministyczne osadzenie tym z prawdziwego dostawcy twojego projektu. Zmierz różnicę recall grupowania semantycznego. Zgłoś, czy rozrzut między strategiami się poszerza czy zwęża.
4. Dodaj pole `summary` na fragment: jednozdaniowy opis centroidu. Uruchom ponownie ewaluację z podsumowaniem dodanym do treści fragmentu. Zmierz wzrost recall.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|--------|-----------------|-----------------------|
| Recall@k | "Czy dostaliśmy właściwy fragment?" | Ułamek zapytań, w których któryś z top-k fragmentów nakłada się na złoty zakres odpowiedzi |
| Nakładanie fragmentów | "Przesuwne okno" | Ponownie dołącz ostatnie N znaków poprzedniego fragmentu do następnego |
| Dzielnik strukturalny | "Fragmenty świadome nagłówków" | Tnij na granicach H1/H2/H3; tekst nagłówka jest częścią fragmentu |
| Fragmentator semantyczny | "Fragmenty świadome tematu" | Osadź zdania, grupuj według podobieństwa centroidu, tnij na dryfie |
| Dryf centroidu | "Zmiana tematu" | Podobieństwo cosinusowe między bieżącą średnią a następnym zdaniem spada poniżej progu |

## Dalsza lektura

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Faza 11, lekcja 06 - podstawy RAG
- Faza 11, lekcja 07 - zaawansowany RAG
- Faza 19, lekcja 65 - wyszukiwanie hybrydowe, które rankuje fragmenty wyprodukowane tutaj
- Faza 19, lekcja 68 - środowisko ewaluacyjne, które ocenia wybór strategii w produkcji