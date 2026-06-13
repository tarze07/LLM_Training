# Multimodalny RAG i Międzymodalne Wyszukiwanie

> Natywnie wizyjny RAG dokumentowy to jeden wycinek. Produkcyjny multimodalny RAG idzie szerzej — wyszukiwanie w tekstach, obrazach, audio i wideo dla przepływów pracy takich jak planowanie podróży ("znajdź mi ciche wegańskie brunchowe miejsce z naturalnym światłem"), triaż medyczny ("jaki uraz pasuje do tego zdjęcia + tych notatek"), e-commerce ("stroje podobne do tego selfie, w moim rozmiarze") i serwis terenowy ("zdiagnozuj ten dźwięk silnika plus zdjęcie części"). Trzy przeglądy z 2025 roku — Abootorabi i in., Mei i in., Zhao i in. — skodyfikowały podproblemy: wyszukiwanie międzymodalne, fuzja wyników wyszukiwania, ugruntowanie generacji, ocena multimodalna. Ta lekcja czyta te przeglądy i projektuje produkcyjny potok.

**Type:** Build
**Languages:** Python (stdlib, cross-modal retriever with fusion + grounded generator)
**Prerequisites:** Phase 12 · 23 (ColPali), Phase 11 (RAG basics)
**Time:** ~180 minutes

## Learning Objectives

- Zaprojektować wyszukiwanie międzymodalne: tekst → obraz, obraz → tekst, audio → wideo itp.
- Porównać trzy strategie fuzji: fuzja wyników, fuzja oparta na uwadze, fuzja MoE.
- Wyjaśnić ugruntowanie generacji: jak wygląda "cytuj swoje źródła", gdy źródła są mieszanką modalności.
- Wymienić trzy kanoniczne przeglądy multimodalnego RAG z 2025 roku i ich taksonomię podproblemów.

## The Problem

RAG jedno-modalny to rozwiązany wzorzec: osadź zapytanie, osadź fragmenty, wyszukaj, włóż do LLM. Multimodalny RAG wymaga:

1. Wielu głowic wyszukiwania (każda modalność potrzebuje osadzeń w kompatybilnej przestrzeni).
2. Fuzji wyników wyszukiwania między modalnościami.
3. Ugruntowania generacji, które cytuje źródła między modalnościami.
4. Metryk oceny obejmujących sygnał międzymodalny.

Przeglądy z 2025 roku wszystkie dochodzą do tej samej taksonomii.

## The Concept

### Wyszukiwanie międzymodalne

Pobierz dokumenty modalności B dla zapytania modalności A. Trzy wzorce:

1. Wspólna przestrzeń osadzeń. CLIP i CLAP produkują osadzenia tekst + obraz / tekst + audio we wspólnej przestrzeni. Podobieństwo cosinusowe między modalnościami działa bezpośrednio. Ograniczone do par trenowanych przez CLIP.

2. Enkoder per-modalność + translacja. Enkoder tekstu + enkoder obrazu + mały moduł translatora mapujący między przestrzeniami. Sen2Sen autorstwa Gupta i in. oraz inne projekty z 2024. Elastyczne, ale dodaje złożoności.

3. VLM jako enkoder. Użyj ukrytych stanów VLM jako reprezentacji wyszukiwania. Każda modalność obsługiwana przez VLM działa. Wyższa jakość, droższe.

Wybór: CLIP / SigLIP 2 dla tekst+obraz; CLAP dla tekst+dźwięk; VLM-hidden-states dla międzymodalności na granicznej jakości.

### Strategie fuzji

Pobrałeś 10 wyników: 5 obrazów, 3 fragmenty tekstu, 2 klipy audio. Jak je połączyć?

Fuzja wyników (najtańsza). Każda modalność ma własny retriever, każdy zwraca wyniki. Normalizuj wyniki wewnątrz modalności, a następnie sumuj. Proste, często działa.

Fuzja oparta na uwadze. Połącz wszystkie pobrane elementy, pozwól małej sieci uwagi je ważyć. Wymaga treningu.

Fuzja MoE. Sieć bramkująca kieruje do ekspertów specyficznych dla modalności. Różne typy zapytań są kierowane inaczej — pytanie wizualne waży obrazy wyżej.

Domyślne produkcyjne: fuzja wyników z lekkim nachyleniem w stronę dominującej modalności zapytania. Aktualizuj do MoE, jeśli testy A/B wykażą wyraźne korzyści w twojej domenie.

### Ugruntowanie generacji

LLM powinien cytować, który pobrany element był podstawą każdego twierdzenia. Dla multimodalnych:

- Źródło tekstowe: standardowy cytat `[1]`.
- Źródło obrazowe: `[img 3]` z krótkim podpisem.
- Audio: `[audio 2 at 0:34]`.

Trenuj generator z danymi uwzględniającymi ugruntowanie: każde twierdzenie w celu treningowym jest oznaczone indeksem źródła. Przy inferencji model naturalnie emituje cytaty.

### Przeglądy z 2025 roku

Abootorabi i in. (arXiv:2502.08826, "Ask in Any Modality"): taksonomia multimodalnego RAG. Obejmuje wyszukiwanie, fuzję, generację. Najszersze pokrycie.

Mei i in. (arXiv:2504.08748, "A Survey of Multimodal RAG"): skupia się na benchmarkach podzadań i trybach awarii. Przydatne do projektowania ewaluacji.

Zhao i in. (arXiv:2503.18016): przegląd skoncentrowany na wizji. Mocny na pracach z rodziny ColPali.

Przeczytanie wszystkich trzech daje ci stan wiedzy na wiosnę 2025. Większość podproblemów jest wciąż otwarta.

### MuRAG — fundamentalna publikacja

MuRAG (Chen i in., 2022) był pierwszym multimodalnym RAG. Wyszukiwał obraz + tekst z multimodalnej bazy wiedzy, generował odpowiedzi. Pokazał wykonalność przed falą VLM. Nowoczesne systemy (REACT, VisRAG, M3DocRAG) budują na nim.

### Przykład produkcyjny planera podróży

Zapytanie: "znajdź mi ciche wegańskie brunchowe miejsce z naturalnym światłem."

Potok:

1. Dekompozycja zapytania. "ciche" → słowo kluczowe audio/recenzja; "wegańskie brunch" → pozycja menu; "naturalne światło" → cecha obrazu.
2. Wyszukiwanie per modalność:
    - Wyszukiwanie tekstowe w recenzjach: "wegańskie brunch, cicha atmosfera."
    - Wyszukiwanie obrazowe w zdjęciach restauracji: "naturalne światło, przewiewnie."
    - Wyszukiwanie audio w klipach dźwięku otoczenia: "niski decybel, brak muzyki."
3. Fuzja wyników. Każda restauracja ma złożony wynik.
4. Top-k restauracji → generator VLM z wszystkimi dowodami → odpowiedź z cytatami.

To wykracza daleko poza text-RAG. Każda modalność dodaje sygnał, którego sam tekst nie wychwytuje.

### Agentowy multimodalny RAG

Wielo-hopowy: jeśli pierwsze wyszukiwanie nie zwróci odpowiedzi o wysokim poziomie ufności, LLM przeformułowuje i wyszukuje ponownie. Wzorce agentowego RAG z Fazy 14 mają tu zastosowanie. Przykłady:

- Wyszukaj wstępne top-10 → LLM pyta "zbyt głośne, filtruj dla <40 dB" → ponowne wyszukiwanie.
- Wyszukaj obrazy → LLM widzi, że jeden ma menu → wyszukaj tekst menu → odpowiedź.

Dodaje złożoności, ale obsługuje zapytania, których jednorazowe wyszukiwanie nie jest w stanie obsłużyć.

### Ewaluacja

Ocena międzymodalna jest wciąż niedojrzała. Typowe proxy:

- Recall@k per modalność.
- Dokładność fuzji top-k.
- Satysfakcja typu end-to-end oceniana przez człowieka.
- Specyficzne dla zadania (zakończone rezerwacje, dokonane zakupy).

Nie ma standardowego benchmarku obejmującego wszystkie modalności. Większość publikacji ocenia na zadaniach specyficznych dla domeny.

## Use It

`code/main.py`:

- Trzy pozorowane retrievery (tekst, obraz, audio) działające na wspólnym korpusie restauracji.
- Fuzja wyników łącząca wyniki modalności z konfigurowalnymi wagami.
- Szablon generatora emitujący ostateczną odpowiedź z cytatami.
- Prosta agentowa pętla przeformułowująca zapytanie, jeśli poziom ufności jest niski.

## Ship It

Ta lekcja produkuje `outputs/skill-multimodal-rag-designer.md`. Dla specyfikacji produktu z multimodalnym przepływem zapytań projektuje retrievery, fuzję, generator i ewaluację.

## Exercises

1. Zaproponuj multimodalny RAG do triażu medycznego: zapytanie = zdjęcie urazu + tekstowe objawy. Jakie modalności wyszukują z jakiej bazy wiedzy?

2. Fuzja wyników to prosta ważona suma. Jaki tryb awarii ma, którego unika fuzja MoE?

3. Przeczytaj taksonomię Abootorabi i in. (Sekcja 3). Jakie są trzy kanoniczne podproblemy i jak odwzorowują się na twój wybrany produkt?

4. Zaprojektuj specyfikację ewaluacji dla multimodalnego RAG planera podróży. Jakie metryki obejmują recall obrazów, recall audio i poprawność złożoną?

5. Agentowy wielo-hopowy RAG ma narzut opóźnienia na każdą rundę. Przy jakim poziomie trudności zapytania zysk dokładności uzasadnia opóźnienie?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Cross-modal retrieval | "Query one modality, retrieve another" | Zapytanie tekstowe wyszukuje obrazy; zapytanie obrazowe wyszukuje tekst; wymaga wspólnej przestrzeni lub translatora |
| Score fusion | "Combine scores" | Ważona suma wyników wyszukiwania per modalność; najprostsza fuzja |
| MoE fusion | "Modality-routed experts" | Sieć bramkująca wybiera, której modalności wynikom ufać per zapytanie |
| Grounded generation | "Cite your sources" | Każde twierdzenie w odpowiedzi oznaczone indeksem źródła |
| MuRAG | "First multimodal RAG" | Publikacja z 2022 roku, która ustanowiła wzorzec multimodalnego RAG |
| Agentic multi-hop | "Reformulate and retry" | LLM ponownie odpytuje retrievery, gdy poziom ufności pierwszego przejścia jest niski |

## Further Reading

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)