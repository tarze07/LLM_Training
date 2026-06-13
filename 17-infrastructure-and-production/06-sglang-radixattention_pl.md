# SGLang i RadixAttention dla obciążeń z dużą liczbą prefiksów

> SGLang traktuje pamięć podręczną KV jako pierwszorzędny, wielokrotnego użytku zasób przechowywany w drzewie radix. Podczas gdy vLLM planuje żądania FCFS (kto pierwszy, ten lepszy), scheduler SGLang świadomy pamięci podręcznej priorytetyzuje żądania z dłuższymi współdzielonymi prefiksami — efektywnie przejście w głąb drzewa radix, aby gorące gałęzie pozostały w HBM. Na Llama 3.1 8B z promptami 1K w stylu ShareGPT, SGLang osiąga ~16 200 tok/s do ~12 500 vLLM, czyli ~29% przewagi. Na obciążeniach RAG z dużą liczbą prefiksów przewaga sięga 6,4x. Na obciążeniach w kształcie klonowania głosu współczynnik trafień pamięci podręcznej przekroczył 86%. Wdrożony na 400 000+ GPU w 2026 w xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS. Haczyk polega na tym, że liczba 6,4x wyparowuje, gdy kolejność prefiksów jest niespójna — kolejność jest dźwignią inżyniera.

**Type:** Learn
**Languages:** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 14 (Agentic RAG)
**Time:** ~75 minutes

## Learning Objectives

- Narysuj diagram RadixAttention: jak prefiksy są przechowywane w drzewie radix i jak bloki KV są współdzielone między sekwencjami zakorzenionymi w tej samej gałęzi.
- Wyjaśnij planowanie świadome pamięci podręcznej i dlaczego FCFS jest złe dla ruchu z dużą liczbą prefiksów.
- Oblicz oczekiwane przyspieszenie dla obciążenia, mając współczynnik trafień pamięci podręcznej prefiksów i dystrybucję długości promptów.
- Nazwij dyscyplinę kolejności promptów, która sprawia, że liczba 6,4x jest realna vs utracona korzyść.

## The Problem

Klasyczne serwowanie traktuje prompt każdego żądania jako nieprzezroczysty. Nawet gdy 5000 żądań RAG zaczyna się od tego samego 2000-tokenowego promptu systemowego plus tego samego preambuły wyszukiwania, vLLM wykonuje prefill tego 2000-tokenowego prefiksu 5000 razy. GPU wykonuje tę samą pracę w kółko.

Obserwacja: prompty w obciążeniach agencyjnych i RAG prawie zawsze współdzielą długie prefiksy. Prompt systemowy, schematy narzędzi, przykłady few-shot, nagłówki wyszukiwania, historia konwersacji — wszystko powtarza się między żądaniami. Gdybyś przechował pamięć podręczną KV dla tego prefiksu raz i użył jej ponownie, nie musiałbyś go ponownie prefillsować.

RadixAttention robi dokładnie to. Tokeny są indeksowane w drzewie radix; każdy węzeł posiada bloki KV dla sekwencji tokenów na swojej ścieżce od korzenia. Nowe żądanie przechodzi drzewo: każdy węzeł, którego token pasuje, ponownie używa bloków KV tego węzła. Koszt prefillu staje się proporcjonalny do „nowego" sufiksu, a nie całego promptu.

Wyzwaniem jest planowanie. Jeśli dwa żądania współdzielą 2000-tokenowy prefiks, a trzecie współdzieli tylko 200 tokenów tego samego prefiksu, chcesz serwować dwa długie współdzielone żądania razem, aby długi prefiks pozostał w HBM. FCFS robi odwrotnie — serwuje tego, kto przyszedł pierwszy, potencjalnie wywłaszczając gorącą gałąź, zanim dotrze następne żądanie z długim prefiksem.

## The Concept

### Drzewo radix jako indeks KV

Drzewo radix (zwarte trie) przechowuje sekwencje tokenów. Każdy węzeł posiada zakres tokenów i bloki KV obliczone dla tego zakresu. Dzieci rozszerzają sekwencję o jeden lub więcej tokenów.

```
root
 |- "Jesteś pomocnym asystentem..."  (2000 tokenów, 124 bloki KV)
      |- "Kontekst: <doc A>..."        (500 tokenów, 31 bloków)
           |- "Pytanie: Alicja..."    (80 tokenów, 5 bloków)
           |- "Pytanie: Bob..."      (95 tokenów, 6 bloków)
      |- "Kontekst: <doc B>..."        (520 tokenów, 33 bloki)
```

Przychodzi nowe żądanie z promptem systemowym + "Kontekst: <doc A>" + "Pytanie: Carol". Scheduler przechodzi: prefiks systemowy pasuje (124 bloki ponownie użyte), gałąź doc-A pasuje (31 bloków ponownie użytych), następnie alokuje świeże bloki tylko dla "Pytanie: Carol" (4 bloki). Koszt prefillu: 4 bloki nowych tokenów. Bez drzewa: 160 bloków. ~40x oszczędności na prefillu.

### Planowanie świadome pamięci podręcznej

Ponowne użycie oparte na drzewie radix jest bezcelowe, jeśli pamięć podręczna się zmienia. Dwie kluczowe polityki:

1. **Wywołanie w głąb (depth-first)**. Przy wyborze następnego żądania z kolejki, preferuj żądania zakorzenione w tej samej gałęzi co aktualny zbiór uruchomionych. To utrzymuje gorącą gałąź przypiętą.
2. **LRU na poziomie gałęzi, a nie bloku**. Wywłaszczaj całe gałęzie (zaczynając od najrzadziej używanych liści) zamiast pojedynczych bloków, aby kształt pamięci podręcznej odpowiadał kształtowi radix.

FCFS narusza obie. Żądanie współdzielące 2000 tokenów siedzi za żądaniem współdzielącym 50, a następnie gałąź 2000 tokenów zostaje wywłaszczona, aby przyjąć to z 50 tokenami.

### Liczby benchmarków, które powinieneś zapamiętać

- Llama 3.1 8B, H100, ShareGPT 1K promptów: SGLang ~16 200 tok/s vs vLLM ~12 500 (~29% przewagi).
- RAG z dużą liczbą prefiksów (ten sam system + ten sam dokument, różne pytania): do 6,4x na SGLang.
- Obciążenia klonowania głosu: 86,4% współczynnik trafień pamięci podręcznej prefiksów.
- Produkcyjne współczynniki trafień u klientów SGLang: 50-99% w zależności od dyscypliny promptów.
- Wdrożone na 400 000+ GPU w 2026.

### Haczyk kolejności

Liczba 6,4x opiera się na spójnej kolejności szablonu promptów. Jeśli Twój klient konstruuje prompty jako `[system, narzędzia, kontekst, historia, pytanie]` w niektórych żądaniach i `[system, kontekst, narzędzia, historia, pytanie]` w innych, drzewo nie może znaleźć współdzielonego prefiksu. To, co dla człowieka wygląda jak współdzielony prefiks, dla drzewa radix jest dwiema odrębnymi sekwencjami.

Dźwignia inżyniera: Twój szablon promptu jest kluczem pamięci podręcznej. Ustal kolejność. Umieść wszystko niezmienne (system, narzędzia, schematy) najpierw. Umieść kontekst wyszukiwania następny. Umieść pytanie użytkownika na końcu. Nie przeplataj dynamicznej treści w prefiksie.

Rzeczywisty przypadek z badań: przeniesienie dynamicznej treści poza możliwy do buforowania prefiks przeniosło jedno wdrożenie z 7% do 74% współczynnika trafień w jednej zmianie.

### Gdzie RadixAttention wygrywa i przegrywa

Wygrywa:
- RAG (ten sam preambuła wyszukiwania, różne pytania).
- Agenci (te same schematy narzędzi, różne zapytania).
- Czat z długim promptem systemowym.
- Obciążenia głosowe/wizyjne z powtarzającymi się preambułami.

Przegrywa (wraca do przepustowości na poziomie vLLM):
- Generowanie pojedyncze z unikalnymi promptami (uzupełnianie kodu, otwarty czat bez promptu systemowego).
- Dynamiczne prompty, gdzie każde żądanie przeplata unikalną treść w prefiksie.

### Dlaczego to problem schedulera, a nie tylko jądra

Możesz zaimplementować ponowne użycie KV jako sztuczkę jądra. Insight SGLang polega na tym, że ponowne użycie opłaca się tylko wtedy, gdy scheduler utrzymuje gorącą gałąź. Naiwna polityka „użyj ponownie, jeśli dostępne" będzie zmieniać pamięć podręczną pod mieszanym obciążeniem. Scheduler indeksowany drzewem radix jest tym, co zamienia sztuczkę jądra w 29% przewagę produkcyjną.

### Interakcja z vLLM

Te dwa systemy nie są ścisłymi konkurentami. W 2026 vLLM dodał buforowanie prefiksów (`--enable-prefix-caching`) i router świadomy pamięci podręcznej (vLLM Router w Rust). Luka się zamknęła, ale nie zniknęła całkowicie — cały stos SGLang jest najpierw radix; vLLM przeszczepił go. Dla obciążeń zdominowanych przez ponowne użycie prefiksów, SGLang pozostaje domyślnym. Dla ogólnego serwowania bez silnych wzorców prefiksów, vLLM pozostaje równy lub lepszy.

```figure
roofline
```

## Use It

`code/main.py` implementuje zabawkowe drzewo radix KV plus scheduler z dwiema politykami: FCFS i świadomą pamięci podręcznej. Uruchamia to samo obciążenie przez obie, raportuje współczynnik trafień pamięci podręcznej prefiksów i różnicę przepustowości. Następnie uruchamia obciążenie z „pomieszaną kolejnością", aby pokazać załamanie 6,4x.

## Ship It

Ta lekcja produkuje `outputs/skill-radix-scheduler-advisor.md`. Biorąc opis obciążenia (kształt szablonu promptu, wzorzec wyszukiwania, liczba równoczesnych dzierżawców), produkuje zalecenie dotyczące kolejności promptów i decyzję go/no-go dla adopcji SGLang.

## Exercises

1. Uruchom `code/main.py`. Porównaj FCFS i świadomą pamięci podręcznej na tym samym obciążeniu. Skąd bierze się różnica — oszczędności prefillu, oszczędności dekodowania, czy opóźnienie w kolejce?
2. Zmodyfikuj obciążenie, aby prompty losowo permutowały `[system, narzędzia, kontekst]`. Uruchom ponownie. Co dzieje się ze współczynnikiem trafień? Dlaczego?
3. Oblicz koszt HBM utrzymywania 2000-tokenowego promptu systemowego jako jednej gałęzi radix na Llama 3.1 8B. Porównaj do kosztu 16-sekwencyjnej partii bez ponownego użycia prefiksów.
4. Przeczytaj artykuł SGLang RadixAttention. Wyjaśnij w trzech zdaniach, dlaczego wywłaszczanie LRU w kształcie drzewa bije LRU w kształcie bloku pod obciążeniem z dużą liczbą prefiksów.
5. Klient raportuje tylko 8% współczynnika trafień pamięci podręcznej. Wymień trzy prawdopodobne przyczyny i diagnostykę, którą byś uruchomił dla każdej.

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| RadixAttention | „ta rzecz SGLang" | Pamięć podręczna KV indeksowana jako drzewo radix, więc współdzielone prefiksy ponownie używają bloków |
| Radix tree | „zwarte trie" | Drzewo, gdzie każdy węzeł posiada zakres tokenów i swoje bloki KV |
| Cache-aware scheduler | „gorąca-gałąź-najpierw" | Scheduler, który preferuje żądania współdzielące rezydentną gałąź |
| Prefix-cache hit rate | „jaka część Twojego promptu była darmowa" | Frakcja tokenów promptu obsłużonych z ponownie użytych bloków KV |
| FCFS | „kto pierwszy, ten lepszy" | Domyślne planowanie, które psuje lokalność prefiksów |
| Branch-level LRU | „wywłaszcz liść" | Polityka wywłaszczania dopasowana do kształtu radix |
| Prompt template ordering | „klucz pamięci podręcznej" | Kolejność komponentów promptu określa, co drzewo może współdzielić |
| System prompt pinning | „rezydentny prefiks" | Utrzymuj niezmienną część systemową przypiętą, aby uniknąć wymiany wywłaszczeń |

## Further Reading

- [SGLang GitHub](https://github.com/sgl-project/sglang) — źródło i dokumentacja.
- [SGLang documentation](https://sgl-project.github.io/) — szczegóły RadixAttention i planowania.
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — odniesienie projektowe.
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — liczby benchmarków i uzasadnienie schedulera.
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — własna implementacja podobna do radix vLLM, dla porównania.

(End of file - total 128 lines)