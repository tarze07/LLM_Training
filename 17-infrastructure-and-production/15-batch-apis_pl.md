# Batch API — 50% zniżki jako standard branżowy

> Każdy większy dostawca oferuje asynchroniczne Batch API z 50% zniżką i ~24-godzinnym czasem realizacji. OpenAI, Anthropic, Google oraz większość platform inferencyjnych (Fireworks batch tier, Together batch) implementują ten sam wzorzec. Połączenie batcha z cachowaniem promptów w potokach nocnych obniża koszt do ~10% kosztu synchronicznego bez cache. Zasada jest brutalnie prosta: jeśli nie jest interaktywne, powinno być w batchu. Potoki generowania treści, klasyfikacja dokumentów, ekstrakcja danych, generowanie raportów, masowe etykietowanie, tagowanie katalogów — wszystko, co toleruje 24-godzinną latencję, to pieniądze zostawione na stole, dopóki nie trafi do batcha. Wzorzec produkcyjny na 2026 rok to triaż każdego nowego obciążenia LLM do trzech pasów: interaktywne (synchroniczne z cache), pół-interaktywne (asynchroniczna kolejka z fallbackiem), batch (nocny, z wejściem cachowanym). Obciążenia udające interaktywne, ale tolerujące minuty latencji, marnują najwięcej.

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## Learning Objectives

- Wymienić trzy Batch API od dostawców (OpenAI, Anthropic, Google) i wspólne gwarancje 50% zniżki + 24h czasu realizacji.
- Obliczyć koszt połączenia batcha z cachowanym wejściem dla nocnego obciążenia klasyfikacyjnego i porównać z bazowym kosztem synchronicznym bez cache.
- Przeprowadzić triaż obciążenia na interaktywne / pół-interaktywne / batch i uzasadnić wybór pasa.
- Wymienić dwie pułapki: częściowa interaktywność (użytkownik oczekuje szybciej niż 24h) i dryf schematu wyjścia (format pliku batch różni się u poszczególnych dostawców).

## The Problem

Twój zespół uruchamia nocny potok generowania raportów. 50 000 dokumentów, każdy do streszczenia, klastrowanie podsumowań, przygotowanie briefu wykonawczego. Uruchomione synchronicznie zajmuje 4 godziny za $2000/noc. Słyszysz o Batch API.

Batch daje 50% zniżki. Włączasz też cachowanie promptów dla system promptu (współdzielonego przez wszystkie 50k wywołań). Po połączeniu rachunek spada do $180/noc — ~9% wartości bazowej. Ten sam potok, trzy zmiany konfiguracji.

Batch to najtańsza dźwignia w zestawie narzędzi do optymalizacji kosztów LLM, której nikt nie używa. Powód jest głównie organizacyjny: zespoły myślą "czas rzeczywisty", gdy SLA brzmi "do rana". Ta lekcja mówi o tym, jak nie zostawiać 90% rachunku na stole.

## The Concept

### Trzy Batch API

**OpenAI Batch API**: przesyłanie pliku JSONL z listą żądań. Obiecany czas realizacji 24 godziny (zwykle ~2-8 godzin w praktyce). 50% zniżki na tokeny wejściowe i wyjściowe. Endpoint `/v1/batches`. Dane kwalifikujące się do cache otrzymują dodatkowo cenę cachowanego wejścia.

**Anthropic Message Batches**: przesyłanie JSONL. 24-godzinny czas realizacji. 50% zniżki. Obsługuje `cache_control` — zapisy do cache są jawne, odczyty odbywają się automatycznie w ramach batcha.

**Google Vertex AI Batch Prediction**: wejście przez BigQuery lub GCS. Podobna 50% zniżka dla Gemini. Integracja z potokami Vertex.

### Znaczenie: asynchroniczne, nie wolne

Batch to "obiecuję zwrócić wyniki w ciągu 24 godzin" — nie "to zajmie 24 godziny". Typowy P50 to 2-6 godzin. Dostawca planuje Twojego batcha w oknach poza szczytem, gdy zasoby GPU są niewykorzystane.

### Połączenie z cachowaniem

Streszczanie 50k dokumentów z tym samym 4K-tokenowym system promptem:

- Synchroniczne bez cache: 50000 × ($input × 4000 + $output × 200) według pełnych stawek.
- Synchroniczne z cache: system prompt cachowany po pierwszym zapisie; pozostałe 49999 otrzymuje 10x tańsze wejście.
- Batch z cache: wszystko powyższe plus 50% zniżki na odczyt i zapis.

Połączenie: batch + cache = ~10% rachunku synchronicznego bez cache. Każde obciążenie działające nocą ze współdzielonym system promptem powinno tego używać.

### Triaż obciążenia

**Interaktywne** — użytkownik czeka na odpowiedź. TTFT ma znaczenie. Wywołanie synchroniczne z cachowaniem promptu. Nie nadaje się do batcha.

**Pół-interaktywne** — użytkownik wysyła zadanie, sprawdza za kilka minut. Kolejka asynchroniczna z fallbackiem do synchronicznego, jeśli batch niedostępny. Np. indeksowanie RAG o umiarkowanym wolumenie.

**Batch** — użytkownik oczekuje wyników "do rana" lub "w ciągu godziny". Potoki treści, klasyfikacja na dużą skalę, analiza offline. Zawsze batch, zawsze z cachowaniem.

Częsty błąd: klasyfikowanie wszystkiego jako interaktywnego, ponieważ potok działa produkcyjnie. Produkcja to nie specyfikacja latencji — SLA nią jest.

### Pułapka częściowej interaktywności

Niektóre funkcje wyglądają na interaktywne, ale tolerują 5-10 minut. Przykład: nocny raport kondycji klienta z przyciskiem "odśwież". Użytkownik klika odśwież; poczekanie 10 minut jest w porządku. Zespół wdraża jako synchroniczne. 50 równoczesnych odświeżeń kosztuje 10x więcej niż wysłane batchowo przez e-mail.

Pytanie, które należy zadać: "Co oznacza 24 godziny dla tego użytkownika?" Jeśli odpowiedź brzmi "nie zauważyliby", wyślij do batcha.

### Pułapka schematu wyjścia

Formaty plików batch różnią się u poszczególnych dostawców:

- OpenAI: JSONL, jedno żądanie na linię.
- Anthropic: JSONL, jedna wiadomość na linię; format odpowiedzi osadzony.
- Vertex: tabela BigQuery lub prefiks GCS z TFRecord.

Pisanie "jednego klienta batch" dla wielu dostawców oznacza kod adaptera dla każdego dostawcy. Bramki reklamujące multi-provider batch (Portkey, niektóre poziomy LiteLLM) wciąż cienko opakowują surowy format.

### Liczby, które warto zapamiętać

- Zniżka batch u dostawców: 50% flat na wejściu + wyjściu.
- SLA czasu realizacji: 24 godziny gwarantowane, 2-6 godzin typowy P50.
- Połączony batch + cachowane wejście: ~10% kosztu synchronicznego bez cache.
- Zasada triażu obciążenia: jeśli 24h latencja jest akceptowalna, zawsze batch.

## Use It

`code/main.py` oblicza koszty dla synchronicznego, synchronicznego z cache, batch i batch z cache dla obciążenia 50k dokumentów. Raportuje oszczędności w $ i procentach.

## Ship It

Ta lekcja produkuje `outputs/skill-batch-triager.md`. Na podstawie charakterystyki obciążenia przeprowadza triaż na interaktywne/pół-interaktywne/batch i szacuje oszczędności.

## Exercises

1. Uruchom `code/main.py`. Dla potoku 100k dokumentów z system promptem 3K tokenów i wyjściem 500 tokenów, oblicz oszczędności pełnego stosu (batch + cache) względem bazowego synchronicznego.
2. Wybierz trzy funkcje w rzeczywistym produkcie, który znasz. Przeprowadź triaż każdej na interaktywne/pół-interaktywne/batch.
3. Użytkownik skarży się, że jego raport zajął 3 godziny. Czy to błąd triażu batch, czy uzasadnione interaktywne? Zapisz kryterium decyzyjne.
4. SLA zwrotu Twojego Batch API wynosi 24h, ale P99 to 20 godzin. Jak komunikujesz to użytkownikowi — jakie jest zachowanie systemu niższego szczebla w przypadku brzegowym?
5. Oblicz próg rentowności: przy jakiej długości współdzielonego prefiksu batch + cache staje się tańszy niż uruchamianie przez noc na własnym zarezerwowanym GPU?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Batch API | "async discount" | 50% off with 24h turnaround |
| JSONL | "batch format" | One JSON request per line; OpenAI/Anthropic standard |
| Message Batches | "Anthropic batch" | Anthropic's batch API product name |
| Batch prediction | "Vertex batch" | Vertex AI's batch API product |
| Turnaround SLA | "24h promise" | Guarantee, not typical; typical is 2-6h |
| Workload triage | "interactivity decision" | Interactive / semi / batch routing decision |
| Output schema | "response format" | Per-provider JSONL layout; not portable |
| Stacked discount | "batch + cache" | ~10% of uncached sync bill when both apply |

## Further Reading

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL format and `/v1/batches` semantics.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch format and `cache_control` interaction.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch semantics.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)