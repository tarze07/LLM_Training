# Capstone 02 — RAG nad Bazą Kodu (Semantyczne Wyszukiwanie Międzyrepozyjne)

> Każda poważna organizacja inżynieryjna w 2026 roku prowadzi wewnętrzne wyszukiwanie kodu, które rozumie znaczenie, a nie tylko ciągi znaków. Sourcegraph Amp, odpowiedzi na bazie kodu w Cursorze, graf korporacyjny Augmenta, repomap Aidera, wewnętrzne MCP Pinteresta — ten sam kształt. Zaindeksuj wiele repozytoriów, sparsuj tree-sitterem, osadź fragmenty na poziomie funkcji i klas, wyszukaj hybrydowo, posegreguj, odpowiedz z cytowaniami. Ten capstone polega na zbudowaniu systemu, który obsługuje 2M linii kodu w 10 repozytoriach i przetrwa przyrostowe przeindeksowanie po każdym git push.

**Type:** Capstone
**Languages:** Python (ingestion), TypeScript (API + UI)
**Prerequisites:** Phase 5 (NLP foundations), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P17
**Time:** 30 hours

## Problem

Do 2026 roku każdy graniczny agent programistyczny ma wbudowaną warstwę wyszukiwania w bazie kodu, ponieważ same okna kontekstu nie rozwiązują pytań międzyrepozyjnych. 1M-tokenowy kontekst Claude'a pomaga; nie eliminuje potrzeby rankingowego wyszukiwania. Naiwne wyszukiwanie cosinusowe po surowych fragmentach zatruwa wyniki na wygenerowanym kodzie, na duplikacjach w monorepozytorium i na długim ogonie rzadko importowanych symboli. Produkcyjną odpowiedzią jest hybrydowe (gęste + BM25) wyszukiwanie po fragmentach świadomych AST z re-rankrem, wsparte grafem referencji symboli.

Uczysz się tego indeksując prawdziwą flotę — nie jedno repozytorium tutorialowe — i mierząc MRR@10, wierność cytowań oraz przyrostową świeżość. Tryby awarii są infrastrukturalne: monorepozytorium ze 100k plików, push który modyfikuje połowę plików, zapytanie które musi przeciąć cztery repozytoria, aby poprawnie odpowiedzieć.

## Koncepcja

Potok indeksacji świadomy AST parsuje każdy plik tree-sitterem, wyodrębnia węzły funkcji i klas, oraz dzieli na fragmenty na granicach węzłów, a nie w stałych oknach tokenowych. Każdy fragment otrzymuje trzy reprezentacje: gęste osadzenie (Voyage-code-3 lub nomic-embed-code), rzadkie terminy BM25 i krótkie podsumowanie w języku naturalnym. Podsumowanie dodaje trzecią możliwą do wyszukania modalność — użytkownicy pytają "jak X jest autoryzowany", a podsumowanie wspomina "authz", nawet jeśli kod ma tylko `check_permission`.

Wyszukiwanie jest hybrydowe. Zapytanie uruchamia zarówno wyszukiwanie gęste, jak i BM25, scala top-k i przekazuje unię do re-rankra typu cross-encoder (Cohere rerank-3 lub bge-reranker-v2-gemma-2b). Posegregowana lista trafia do syntezatora z długim kontekstem (Claude Sonnet 4.7 z prompt caching, lub samodzielnie hostowany Llama 3.3 70B) z instrukcją cytowania każdego twierdzenia przez plik i zakres linii. Odpowiedzi bez cytowań są odrzucane przez filtr po-processingu.

Przyrostowa świeżość jest problemem infrastrukturalnym. Git push uruchamia diff: które pliki się zmieniły, które symbole się zmieniły. Tylko zmienione fragmenty są ponownie osadzane. Zmienione krawędzie symboli między plikami (importy, wywołania metod) są przeliczane. Indeks pozostaje spójny bez przetwarzania 2M linii przy każdym commicie.

## Architektura

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## Stack

- Parsowanie: tree-sitter z 17 gramatykami języków (Python, TS, Rust, Go, Java, C++, itd.)
- Gęste osadzenia: Voyage-code-3 (hostowane) lub nomic-embed-code-v1.5 (samodzielnie hostowane), bge-code-v1 jako zapasowe
- Rzadki indeks: Tantivy (Rust) z BM25F, ważone polem na nazwę symbolu vs. treść
- Baza wektorowa: Qdrant 1.12 z wyszukiwaniem hybrydowym, lub pgvector + pgvectorscale dla zespołów poniżej 50M wektorów
- Model podsumowujący fragmenty: Claude Haiku 4.5 lub Gemini 2.5 Flash, z prompt caching
- Re-ranker: Cohere rerank-3 lub bge-reranker-v2-gemma-2b samodzielnie hostowany
- Orkiestracja: LlamaIndex Workflows do indeksacji, LangGraph do agenta zapytań
- Syntezator: Claude Sonnet 4.7 (1M kontekst) z prompt caching
- Graf symboli: Neo4j (zarządzany) lub kuzu (osadzony) dla krawędzi importów i wywołań
- Obserwowalność: Langfuse spany na krok wyszukiwania + syntezy

## Build It

1. **Przechodzenie indeksacji.** Iteruj historię gita na każdym hooku push. Zbierz zmienione pliki. Dla każdego pliku, sparsuj tree-sitterem, wyodrębnij węzły funkcji i klas z ich pełnym zakresem źródłowym. Emituj rekordy fragmentów `{repo, path, start_line, end_line, symbol, body}`.

2. **Podsumowujący fragmenty.** Przetwarzaj wsadowo fragmenty przez wywołania Haiku 4.5 z prompt caching na preambule systemowej. Prompt: "Podsumuj tę funkcję w jednym zdaniu, podając jej publiczny kontrakt i efekty uboczne." Przechowuj podsumowanie obok fragmentu.

3. **Pula osadzeń.** Dwie równoległe kolejki: gęste (Voyage-code-3 batch 128) i podsumowanie (ten sam model, ale na ciągu podsumowania). Zapisz wektory do Qdrant z ładunkiem `{repo, path, start_line, end_line, symbol, kind}`.

4. **Indeks BM25.** Indeks Tantivy z wagami pól: nazwa symbolu waga 4, treść symbolu waga 1, podsumowanie waga 2. Umożliwia zapytania "znajdź funkcję o nazwie X" obok "znajdź funkcję która robi X".

5. **Graf symboli.** Dla każdego fragmentu, zapisz krawędzie: importy (ten plik używa symbolu Y z repozytorium Z), wywołania (ta funkcja wywołuje metodę M na klasie C), dziedziczenie. Przechowuj w kuzu. Używane w czasie zapytania do rozszerzenia wyszukiwania poza granice repozytoriów.

6. **Agent zapytań.** LangGraph z trzema węzłami. `retrieve` uruchamia gęste + BM25 równolegle, deduplikuje po (repo, path, symbol). `rerank` uruchamia cross-encoder na top-50 i zachowuje top-10. `synth` wywołuje Claude Sonnet 4.7 z posegregowanymi fragmentami w kontekście, cache'uje prompt systemowy, wymaga cytowań plik:linia.

7. **Egzekwowanie cytowań.** Sparsuj wyjście modelu; każde twierdzenie bez zakotwiczenia `(repo/path:start-end)` jest oznaczane do ponownego zapytania lub odrzucane. Zwróć użytkownikowi odpowiedź tylko z cytowaniami.

8. **Przyrostowe przeindeksowanie.** Na każdym webhooku, oblicz diff na poziomie symboli. Ponownie osadź tylko fragmenty, których tekst się zmienił. Przelicz krawędzie symboli dla fragmentów, których importy się zmieniły. Pomiar: push 50 plików przeindeksowany w ciągu 60 sekund dla floty 2M LOC.

9. **Ewaluacja.** Oznacz 100 pytań międzyrepozyjnych z referencyjnymi odpowiedziami plik:linia. Zmierz MRR@10, nDCG@10, wierność cytowań (frakcja twierdzeń z weryfikowalnymi kotwicami) i p50/p99 opóźnienia.

## Use It

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Ship It

Umiejętność będąca rezultatem `outputs/skill-codebase-rag.md`. Mając korpus repozytoriów, uruchamia potok indeksacji, hybrydowy indeks i agenta zapytań, i zwraca cytowaną odpowiedź na każde pytanie międzyrepozyjne. Rubryka:

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Jakość wyszukiwania | MRR@10 i nDCG@10 na 100-pytaniowym zestawie wstrzymanym |
| 20 | Wierność cytowań | Frakcja twierdzeń odpowiedzi z weryfikowalnymi kotwicami plik:linia |
| 20 | Opóźnienie i skala | p95 opóźnienia zapytania przy 10k QPS na rozmiarze indeksowanego korpusu |
| 20 | Poprawność przyrostowego indeksowania | Czas od git push do przeszukiwalności na commicie 50 plików |
| 15 | UX i formatowanie odpowiedzi | Klikalność cytowań, podglądy fragmentów, możliwość zadawania pytań uzupełniających |
| **100** | | |

## Ćwiczenia

1. Zamień Voyage-code-3 na nomic-embed-code samodzielnie hostowany. Zmierz różnicę MRR@10. Raportuj, czy luka zamyka się po włączeniu re-rankingu.

2. Wstrzyknij 20% wygenerowanego kodu (boilerplate wyprodukowany przez LLM) do korpusu i przeprowadź ponowną ewaluację. Zaobserwuj zatrucie wyszukiwania. Dodaj flagę "wygenerowany" do ładunku i zmniejsz wagę tych trafień.

3. Porównaj benchmark Qdrant hybrydowego wyszukiwania vs pgvector + pgvectorscale na swoim rozmiarze korpusu. Raportuj p99 przy batch size 1.

4. Dodaj próbkującą kontrolę dryfu: co tydzień, uruchom ponownie 100-pytaniową ewaluację. Alarmuj przy spadku MRR@10 > 5%.

5. Rozszerz na międzyjęzykową rezolucję symboli: funkcja w Pythonie, która wywołuje serwis Go przez gRPC. Użyj grafu symboli, aby je połączyć.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| AST-aware chunking | "Podział na poziomie funkcji" | Cięcie kodu na granicach węzłów tree-sitter zamiast stałych okien tokenowych |
| Hybrid search | "Gęste + rzadkie" | Uruchom BM25 i wyszukiwanie wektorowe równolegle, scal top-k, posegreguj |
| Cross-encoder rerank | "Ranking drugiego etapu" | Model który punktuje każdą parę (zapytanie, kandydat) razem, dokładniejszy niż cosinus |
| Prompt caching | "Zaciszowany prompt systemowy" | Funkcja Claude/OpenAI 2026, która rabatuje powtarzające się tokeny prefiksu do 90% |
| Symbol graph | "Graf kodu" | Krawędzie dla importów, wywołań, dziedziczenia między plikami i repozytoriami |
| Citation faithfulness | "Wskaźnik ugruntowanych odpowiedzi" | Frakcja twierdzeń, które użytkownik może zweryfikować klikając kotwicę i czytając referencyjny zakres |
| Incremental re-index | "Czas od pusha do wyszukania" | Czas od git push do momentu, gdy zmienione symbole są przeszukiwalne |

## Dalsza lektura

- [Sourcegraph Amp](https://ampcode.com) — production cross-repo code intelligence
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — the reference deep-dive for this capstone
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter ranked repo view
- [Augment Code enterprise graph](https://www.augmentcode.com) — commercial symbol-graph RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — reference implementation
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 details
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder reference
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — internal-platform reference