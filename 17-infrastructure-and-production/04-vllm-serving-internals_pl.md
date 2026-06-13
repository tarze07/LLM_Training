# Wewnętrzne mechanizmy vLLM: PagedAttention, Continuous Batching, Chunked Prefill

> Dominacja vLLM w 2026 opiera się na trzech kumulujących się domyślnych ustawieniach, a nie na jednej sztuczce. PagedAttention jest zawsze włączone. Continuous batching wstrzykuje nowe żądania do aktywnej partii między iteracjami dekodowania. Chunked prefill dzieli długie prompty, aby tokeny dekodowania nigdy nie głodowały. Włącz wszystkie trzy, a Llama 3.3 70B FP8 na jednym H100 SXM5 osiąga 2200-2400 tok/s przy 128 równoczesnych — około 25% powyżej domyślnych ustawień vLLM i 3-4x naiwnej pętli PyTorch. Ta lekcja czyta scheduler i jądro attention na poziomie, który możesz narysować, i kończy się zabawkowym ciągłym batcherem w `code/main.py`, który planuje prefill i decode tak, jak robi to vLLM.

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## Learning Objectives

- Wyjaśnij PagedAttention jako alokator pamięci podręcznej KV: bloki, tablice bloków i dlaczego fragmentacja pozostaje poniżej 4% przy obciążeniu produkcyjnym.
- Narysuj diagram ciągłego batchingu na poziomie iteracji: jak zakończone sekwencje opuszczają partię, a nowe dołączają bez opróżniania.
- Opisz chunked prefill w jednym zdaniu i nazwij, którą metrykę latencji chroni (podpowiedź: to ogon TTFT, a nie średnia przepustowość).
- Nazwij zasadzkę vLLM v0.18.0 z 2026, która gryzie zespoły włączające każdą optymalizację naraz.

## The Problem

Naiwna pętla serwowania PyTorch uruchamia jedno żądanie na raz: tokenizacja, prefill, dekoduj aż do EOS, zwróć. Przy jednym użytkowniku to działa. Przy stu, to kolejka cierpliwych ludzi. Oczywista poprawka — statyczne batching — dopełnia każde żądanie do najdłuższego promptu w oknie, dopełnia każde dekodowanie do najdłuższego oczekiwanego wyjścia i blokuje całą partię na najwolniejszej sekwencji. Płacisz za dopełnienie, którego nie używasz, a szybkie żądania czekają na wolne.

vLLM rozwiązuje trzy problemy naraz. PagedAttention zatrzymuje fragmentację pamięci podręcznej KV przed pożeraniem 60-80% pamięci GPU, tak jak robi to klasyczna ciągła alokacja. Ciągłe batching pozwala żądaniom dołączać i opuszczać partię między każdą iteracją dekodowania, więc partia jest zawsze pełna rzeczywistej pracy. Chunked prefill dzieli 32k-tokenowy prompt na ~512-tokenowe kawałki przeplatane z dekodowaniem, więc długi prompt nie zamraża każdego tokena dekodowania na GPU.

Produkcyjny domyślny na 2026 to wszystkie trzy włączone. Musisz zrozumieć, co każdy robi, ponieważ tryby awarii są wszystkie w schedulerze, a nie w modelu.

## The Concept

### PagedAttention jako system pamięci wirtualnej

Pamięć podręczna KV to `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element` na sekwencję. Dla Llama 3.3 70B przy 8192 tokenach, to około 1,25 GB na sekwencję w BF16. Jeśli zarezerwujesz 8192 slotów dla każdego żądania, ale średnie żądanie używa tylko 1500 tokenów, marnujesz około 82% HBM, które zarezerwowałeś. Klasyczne batching płaci te straty.

PagedAttention zapożycza pomysł z pamięci wirtualnej OS. Pamięć podręczna KV nie jest ciągła na sekwencję. Jest alokowana w blokach o stałym rozmiarze (domyślnie 16 tokenów). Każda sekwencja ma tablicę bloków, która mapuje jej logiczne pozycje tokenów na fizyczne ID bloków. Gdy sekwencja rośnie poza przydzielone bloki, dodawany jest jeden blok. Gdy się kończy, jej bloki wracają do puli.

Fragmentacja spada z 60-80% (klasyczna) do poniżej 4% (PagedAttention). Nie włączasz PagedAttention flagą — to jedyny alokator, który vLLM dostarcza. Pokrętłem jest `--gpu-memory-utilization` (domyślnie 0,9), które mówi vLLM, ile HBM zarezerwować na bloki KV po załadowaniu wag i aktywacji.

### Ciągłe batching na poziomie iteracji

Stare „dynamiczne batching" czekało na okno (powiedzmy 10 ms), aby wypełnić partię, a następnie uruchamiało prefill + decode + decode + decode, aż każda sekwencja się zakończyła. Szybkie sekwencje wychodziły wcześniej i siedziały bezczynnie, podczas gdy GPU kończyło wolne.

Ciągłe batching działa między każdym krokiem dekodowania. Nazwijmy zbiór uruchomionych sekwencji listą `RUNNING`. W każdej iteracji:

1. Każda sekwencja w `RUNNING`, która właśnie osiągnęła EOS lub max_tokens, jest usuwana.
2. Scheduler patrzy na kolejkę oczekujących. Jeśli są wolne bloki KV, przyjmuje nowe sekwencje (prefill lub wznowione).
3. Przejście do przodu działa na tym, co jest teraz w `RUNNING`, emitując jeden nowy token na sekwencję.

Rozmiar partii nigdy nie jest dopełniany do stałej liczby. Sekwencje na różnych pozycjach w swoim wyjściu współdzielą jedno połączone przejście do przodu. W vLLM 2026 nazywa się to `V1 scheduler`. Kluczowa niezmienniczość: scheduler działa raz na iterację dekodowania, a nie raz na żądanie.

### Chunked prefill chroni ogon TTFT

Prefill jest ograniczony obliczeniowo. 32k-tokenowy prompt na Llama 3.3 70B zajmuje ~800 ms czystego prefillu na jednym H100. Podczas gdy prefill działa, tokeny dekodowania dla każdej innej sekwencji w partii czekają. W pętli serwowania, latencja pierwszego tokena (TTFT) jednego długiego promptu staje się pikiem latencji międzytokenowej (ITL) dla kilkudziesięciu innych użytkowników.

Chunked prefill dzieli prefill na kawałki o stałym rozmiarze (domyślnie 512 tokenów) i planuje każdy kawałek jako jednostkę. Między kawałkami scheduler może przesunąć sekwencje dekodowania o jeden token. Poświęcasz mały, bezwzględny narzut latencji prefillu (kilka ms na kawałek) za znacznie niższy dżitter czasu dekodowania. P99 ITL przy mieszanym obciążeniu spada z ~50 ms do ~15 ms w opublikowanych benchmarkach.

### Trzy domyślne ustawienia współdziałają

Wszystkie trzy funkcje zakładają się nawzajem. PagedAttention daje schedulerowi drobnoziarnisty zasób KV do handlu. Ciągłe batching potrzebuje tego drobnoziarnistego zasobu, aby przyjęcie nowej sekwencji nie wymuszało globalnej przetasowania. Chunked prefill to decyzja, którą scheduler podejmuje na tej samej liście `RUNNING` — to jedna więcej polityka schedulera, a nie oddzielny system.

Nie musisz znać każdej flagi. Musisz wiedzieć, co scheduler optymalizuje: goodput pod budżetem bloków KV, z zastrzeżeniem krojenia chunked prefill.

### Zasadzka v0.18.0 z 2026

W vLLM v0.18.0 nie można połączyć `--enable-chunked-prefill` ze spekulacyjnym dekodowaniem z modelem szkicowym (`--speculative-model`). Udokumentowanym wyjątkiem jest N-gram GPU spekulacyjne dekodowanie w schedulerze V1. Zespoły, które włączają każdą flagę bez czytania notatek wydania, dostają błąd w czasie uruchamiania, a nie miękką regresję. Jeśli Twój zysk spekulacyjny był wart włączenia chunked prefill, ponownie rozważ wybór — właściwą odpowiedzią w 2026 jest często EAGLE-3 bez chunked prefill, a nie model szkicowy plus chunked prefill, który się nie kompiluje.

### Liczby, które powinieneś zapamiętać

- Llama 3.3 70B FP8, H100 SXM5, 128 równoczesnych, wszystkie trzy włączone: 2200-2400 tok/s.
- Ten sam model, domyślny vLLM (bez chunked prefill): ~1800 tok/s.
- Ten sam model, naiwna pętla forward PyTorch: ~600 tok/s.
- Straty fragmentacji KV pod PagedAttention przy obciążeniu produkcyjnym: <4%.
- P99 ITL przy mieszanym obciążeniu: ~15 ms z chunked prefill, ~50 ms bez.

### Jak wygląda scheduler

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # zaplanuj kawałki prefill + decode w jednej partii
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # np. 512 tokenów
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # jedno połączone wywołanie GPU
```

`code/main.py` to dokładnie ta pętla w stdlib Python z fałszywymi licznikami tokenów i fałszywą latencją forward. Uruchomienie pokazuje, jak chunked prefill utrzymuje sekwencje dekodowania przy życiu podczas długiego prefillu.

```figure
tensor-parallel
```

## Use It

`code/main.py` symuluje scheduler w stylu vLLM z przełączalnymi funkcjami. Uruchom, aby zobaczyć:

- Tryb `NAIVE`: jedno żądanie na raz, bez batchingu.
- Tryb `STATIC`: dopełnianie i czekanie, klasyczne batching.
- Tryb `CONTINUOUS`: przyjmowanie i zwalnianie na poziomie iteracji.
- Tryb `CONTINUOUS + CHUNKED`: kawałki prefillu przeplatane z dekodowaniem.

Wynik pokazuje całkowitą przepustowość (tokeny na wirtualną sekundę), średnią TTFT i P99 ITL. Wiersz `CONTINUOUS + CHUNKED` powinien dominować na mieszanym ruchu.

## Ship It

Ta lekcja produkuje `outputs/skill-vllm-scheduler-reader.md`. Biorąc konfigurację serwowania (rozmiar partii, wykorzystanie pamięci KV, rozmiar chunked prefill, konfiguracja spekulacyjna), produkuje diagnozę schedulera, która nazywa, które z trzech domyślnych ustawień jest wąskim gardłem i co dostroić.

## Exercises

1. Uruchom `code/main.py`. Porównaj `STATIC` z `CONTINUOUS` na obciążeniu z mieszanymi krótkimi i długimi żądaniami. Skąd bierze się różnica w przepustowości — wydajność prefillu, wydajność dekodowania, czy latencja ogona?
2. Zmodyfikuj zabawkowy scheduler, aby dodać `--max-num-batched-tokens`. Jaka jest właściwa wartość dla H100 uruchamiającego Llama 3.3 70B FP8? (Podpowiedź: to funkcja rozmiaru bloku KV i liczby wolnych bloków, a nie surowego HBM.)
3. Przeczytaj ponownie notatki wydania vLLM v0.18.0. Które kombinacje flag są wzajemnie wykluczające? Wymień je.
4. Oblicz straty fragmentacji pamięci podręcznej KV dla śladu 1000 żądań ze średnią 1500 tokenów wyjściowych, std 600 tokenów, przy (a) ciągłej alokacji na żądanie przy maksimum 8192, (b) PagedAttention z 16-tokenowymi blokami.
5. Wyjaśnij w jednym akapicie, dlaczego chunked prefill pomaga P99 ITL, ale nie przepustowości w izolacji. Skąd w praktyce bierze się zysk przepustowości?

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| PagedAttention | „sztuczka KV" | Alokator bloków o stałym rozmiarze dla pamięci podręcznej KV; fragmentacja <4% |
| Block table | „tablica stron" | Mapowanie na sekwencję od logicznej pozycji tokena do fizycznego bloku KV |
| Continuous batching | „dynamic batching, ale dobrze" | Decyzje o przyjęciu/zwolnieniu podejmowane co iterację dekodowania |
| Chunked prefill | „dzielenie prefillu" | Podział długiego prefillu na 512-tokenowe kawałki przeplatane z dekodowaniem |
| TTFT | „czas do pierwszego tokena" | Prefill + kolejka + sieć; zdominowany przez prefill przy długich promptach |
| ITL | „latencja między tokenami" | Czas między kolejnymi tokenami dekodowania; zdominowany przez rozmiar partii |
| Goodput | „przepustowość spełniająca SLO" | Tokeny/s, gdzie każde żądanie wciąż osiąga cele TTFT i ITL |
| V1 scheduler | „nowy scheduler" | Scheduler vLLM 2026; N-gram spec decode to ścieżka kompatybilna z chunked prefill |
| `--gpu-memory-utilization` | „pokrętło pamięci" | Frakcja HBM zarezerwowana na bloki KV po wagach i aktywacjach |

## Further Reading

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — oficjalne źródło o kompatybilności chunked-prefill i spekulacyjnego dekodowania.
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — cykl wydań 2026 i zachowanie specyficzne dla wersji.
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — oryginalny artykuł, który wciąż definiuje, jak myśleć o alokatorze.
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — analiza fragmentacji i projekt schedulera.
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — szczegółowy przegląd schedulera V1 z wykresami płomieni.

(End of file - total 143 lines)