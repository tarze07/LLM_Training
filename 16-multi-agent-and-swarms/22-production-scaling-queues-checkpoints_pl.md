# Skalowanie Produkcyjne — Kolejki, Punkty Kontrolne, Trwałość

> Skalowanie systemów wieloagentowych do tysięcy równoczesnych uruchomień wymaga **trwałego wykonywania**. Środowisko uruchomieniowe LangGraph zapisuje punkt kontrolny po każdym super-kroku, indeksowany przez `thread_id` (domyślnie Postgres); awarie procesów roboczych zwalniają dzierżawę, a inny proces roboczy wznawia działanie. Agenci mogą spać w nieskończoność, czekając na dane od człowieka. **MegaAgent** (arXiv:2408.09955) uruchomił kolejkę producent-konsument dla każdego agenta z trzema stanami (Bezczynny / Przetwarzanie / Odpowiedź) i dwuwarstwową koordynacją (czat wewnątrz grupy + czat administracyjny między grupami). **Fiber/async** bije thread-per-job dla strumieniowania LLM: wątki są bezczynne przez 99% czasu czekając na tokeny, włókna kooperatywnie ustępują na I/O. Kontrapunkt: "Scaling Agentic Software" Ashpreeta Bediego argumentuje za **FastAPI + Postgres + niczym więcej**, dopóki obciążenie nie udowodni inaczej — proste architektury sięgają dalej niż oczekiwano. Ta lekcja buduje trwały dziennik punktów kontrolnych, kolejkę pracy dla agenta z przejściami stanów, demonstrację async vs wątki i ugruntowuje pragmatyczną zasadę "zacznij prosto".

**Type:** Learn + Build
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## Problem

Prototypowy system wieloagentowy działa na jednym laptopie z trzema agentami w pamięciowej pętli zdarzeń. Przechodzisz do produkcji:

- Agenci czasami działają godzinami (długie badania, oczekiwanie na człowieka w pętli).
- Procesy robocze ulegają awarii. Ponowne uruchomienie traci stan.
- Szczytowe obciążenie jest 10x średniej; potrzebujesz skalowania poziomego.
- Użytkownicy płacą za uruchomienie agenta; potrzebujesz semantyki dokładnie raz do rozliczeń.

Pamięciowa pętla zdarzeń nie robi nic z tego. Potrzebujesz trwałej warstwy wykonywania pod spodem. Kanoniczne opcje z 2026 to:

1. Silnik przepływu pracy z punktami kontrolnymi (Temporal, LangGraph runtime).
2. Kolejka komunikatów z magazynem stanu (Postgres + SQS/RabbitMQ).
3. Frameworki modelu aktorów (kolejka producent-konsument MegaAgent dla każdego agenta).
4. Ręcznie wykonane FastAPI + Postgres (argument Bediego).

Ta lekcja buduje miniaturową wersję każdego z nich.

## Koncepcja

### Trwałe wykonywanie — wzorzec

Silnik trwałego wykonywania utrwala pełny stan programu po każdym "kroku" (super-krok, w języku LangGraph). W razie awarii:

```
proces roboczy ulega awarii w środku kroku
  -> limit czasu dzierżawy
  -> inny proces roboczy przejmuje thread_id
  -> wznawia od ostatniego punktu kontrolnego
  -> bez zduplikowanych efektów ubocznych
```

Wymagania, aby to działało:

- **Stan serializowalny.** Cały stan agenta musi być możliwy do utrwalenia. Domknięcia funkcji z aktywnymi połączeniami do bazy danych nie przetrwają.
- **Deterministyczne wznowienie.** Przy tym samym stanie i tych samych wejściach, agent generuje te same akcje (lub odwołuje się do zewnętrznego deterministycznego wyroczni dla wywołań LLM).
- **Idempotentne efekty uboczne.** Wywołania zewnętrzne (wywołania narzędzi, płatności) muszą być idempotentne lub używać klucza deduplikacji.

LangGraph zapisuje punkt kontrolny po każdym super-kroku; Temporal zapisuje po każdej aktywności; Restate używa dzienników opartych na zdarzeniach. Wszystkie trzy implementują ten sam wzorzec.

### Środowisko uruchomieniowe LangGraph

Każdy agent ma `thread_id`; stan to typowany słownik; każdy super-krok zapisuje wiersz w tabeli punktów kontrolnych. Przy wznowieniu, środowisko uruchomieniowe odtwarza z ostatniego punktu kontrolnego, nie od zera. Agenci mogą `interrupt()` czekając na dane wejściowe od człowieka; środowisko uruchomieniowe utrwala i zwalnia proces roboczy. Gdy dane wejściowe dotrą, każdy proces roboczy może wznowić.

To jest referencyjny projekt produkcyjny w kwietniu 2026.

### Kolejka per-agent MegaAgent

arXiv:2408.09955 opisuje eksperyment skalowania: tysiące równoczesnych agentów w jednym klastrze. Architektura:

```
agent i:
  stan ∈ {Bezczynny, Przetwarzanie, Odpowiedź}
  kolejka_we   <- wiadomości adresowane do agenta i
  kolejka_wy  -> odpowiedzi + efekty uboczne

koordynatory:
  czat wewnątrz grupy  (agenci w tej samej grupie)
  czat administracyjny między grupami  (routing wysokiego poziomu)
```

Dwuwarstwowa koordynacja pozwala, aby rozmowa wewnątrz grupy była gęsta, podczas gdy między grupami pozostaje rzadka — wzorzec używany do utrzymania kosztów liniowych przy tysiącach agentów.

### Async vs thread-per-job

Wywołania LLM są zależne od I/O. Wątek czekający na następny token jest bezczynny przez 99% czasu. Wątki kosztują ~1MB RAM każdy; przy 10 000 równoczesnych wywołań, to 10GB tylko na stosy.

Włókna (Python `asyncio`, gorutyny Go, Rust `tokio`) kooperatywnie ustępują na I/O. Te same 10 000 wywołań mieści się wygodnie w procesie. W skali agentów LLM, async nie jest optymalizacją — to architektura.

Wyjątek: obliczeniowo intensywne przetwarzanie końcowe (embedding, sztuczki tokenizatora) wciąż chce wątków lub procesów. Oddziel swoją warstwę I/O od warstwy obliczeniowej.

### Kontrapunkt Bediego

"Scaling Agentic Software" (Ashpreet Bedi, 2026) argumentuje, że większość zespołów przesadza z inżynierią, zanim zmierzy obciążenie. Pragmatyczna domyślna konfiguracja:

- FastAPI + Postgres.
- Każde uruchomienie agenta to wiersz; stan aktualizowany w miejscu z optymistyczną współbieżnością.
- Zadania w tle przez `pg_notify` lub prostego workera Celery.
- Polityka ponawiania w kodzie aplikacji.

Dla obciążeń poniżej ~100 równoczesnych uruchomień agentów przy zarządzalnych zadaniach, to często wszystko, czego potrzebujesz. Modernizuj, gdy zmierzysz, że to zawodzi.

Zasada: adoptuj frameworki trwałego wykonywania, gdy napotkasz konkretny problem, którego proste architektury nie mogą rozwiązać. Przedwczesna adopcja traci czas na ceremonie, które się nie zwracają.

### Semantyka dokładnie raz

Dla płatnych uruchomień agentów potrzebujesz "efektywnie dokładnie raz" (dostarczenie co najmniej raz + idempotentny konsument). Inżynieryjne posunięcia:

- **Klucz deduplikacji na uruchomienie.** Dołącz go w każdym wywołaniu efektu ubocznego.
- **Wzorzec outbox.** Efekty uboczne najpierw zapisują do tabeli, potem oddzielny proces je wykonuje. Oba kroki są idempotentne.
- **Transakcje kompensujące.** Gdy efekt uboczny się powiedzie, ale jego zapis śledzenia zawiedzie, zaplanuj kompensację.

To są wzorce inżynierii baz danych, nie specyficzne dla LLM. Podatek LLM polega tylko na tym, że wywołania LLM są wolne; wszystko inne to standardowe systemy rozproszone.

### Wdrożenie tęczowe (rainbow deployment)

System badawczy wieloagentowy Anthropic używa "wdrożeń tęczowych": wiele wersji środowiska uruchomieniowego agenta działa równocześnie, aby długotrwałe agentów nie trzeba było zabijać przy każdym wdrożeniu kodu. Kanarkuj nowe wersje na wycinku ruchu; wycofuj stare wersje, gdy ich agenci zakończą.

To standard dla długotrwałych systemów stanowych; adaptacja z 2026 polega na tym, że agenci mogą żyć godzinami, więc cykle wdrożeniowe muszą to uwzględniać.

### Kanoniczna lista kontrolna dla produkcji

- Trwały stan (punkty kontrolne, migawki lub outbox + odtwarzalny dziennik).
- Idempotentne efekty uboczne.
- Warstwa I/O async dla wywołań LLM.
- Dostarczenie co najmniej raz z deduplikacją.
- Wdrożenie tęczowe/kanarkowe dla stanowych obciążeń pracy.
- Obserwowalność: ślady per-agent, audyt super-kroków, licznik ponowień.

## Build It

`code/main.py` implementuje:

- `CheckpointStore` — dziennik punktów kontrolnych oparty na SQLite z kluczami thread-id. Każdy super-krok dodaje wiersz.
- `run_with_checkpoint(agent, thread_id)` — symuluje awarię w środku działania; drugi proces roboczy wznawia od ostatniego punktu kontrolnego.
- `AgentQueue` — maszyna stanów Bezczynny / Przetwarzanie / Odpowiedź dla każdego agenta z małą kolejką pracy.
- `demo_async_vs_threads()` — uruchamia 500 równoczesnych symulowanych "wywołań LLM" przez asyncio i przez wątki; raportuje czas ścienny i szczytowe użycie pamięci (przybliżone).

Uruchom:

```
python3 code/main.py
```

Oczekiwane wyniki: wznowienie punktu kontrolnego udaje się po symulowanej awarii; wersja async obsługuje 500 równoczesnych wywołań w < 1s; wersja wątkowa zajmuje kilka sekund i używa o rzędy wielkości więcej pamięci na jednostkę równoczesną.

## Use It

`outputs/skill-scaling-advisor.md` doradza w wyborze trwałego wykonywania: FastAPI + Postgres, LangGraph runtime, Temporal, lub własne. Skalibrowany według obciążenia, potrzeb retencji stanu i częstotliwości wdrożeń.

## Ship It

Kanoniczne utwardzanie produkcyjne:

- **Zacznij prosto (zasada Bediego).** FastAPI + Postgres, dopóki nie zmierzysz, że zawodzi.
- **Instrumentuj wszystko przed optymalizacją.** Histogram opóźnień na uruchomienie, czas na krok, liczba ponowień, kategoryzacja błędów.
- **Wzorzec outbox dla efektów ubocznych.** Szczególnie płatności i zewnętrzne wywołania API.
- **Wdrożenia tęczowe.** Nigdy nie zabijaj aktywnych uruchomień agentów podczas wdrożeń.
- **Adoptuj silniki trwałego wykonywania (Temporal / LangGraph / Restate), gdy** napotkasz konkretne problemy: godzinne oczekiwania na człowieka w pętli, koordynacja między regionami, złożone polityki ponawiania/kompensacji.
- **Async dla warstwy I/O.** Wątki tylko dla obliczeniowo intensywnego przetwarzania końcowego.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że wznowienie punktu kontrolnego działa; zmierz różnicę w współbieżności async vs wątki.
2. Zaimplementuj tabelę **outbox**: każde wywołanie narzędzia najpierw zapisuje do outbox, potem oddzielna gorutyna/zadanie wykonuje. Zweryfikuj idempotentność, uruchamiając wywołanie narzędzia dwukrotnie.
3. Zasymuluj **wdrożenie tęczowe**: dwie równoczesne wersje środowiska uruchomieniowego; skieruj połowę nowych thread_id do każdej; potwierdź, że aktywne wątki w starej wersji nie są przerywane.
4. Przeczytaj dokumentację środowiska uruchomieniowego LangGraph (link poniżej). Zidentyfikuj, które funkcje środowiska uruchomieniowego zajęłyby najwięcej czasu do odtworzenia w ręcznie wykonanej wersji FastAPI + Postgres. Czy to powód do adopcji, czy możesz odroczyć?
5. Przeczytaj MegaAgent (arXiv:2408.09955) Sekcję 3. Dwuwarstwowa koordynacja (czat wewnątrz grupy + czat administracyjny między grupami) jest wyraźna. Naszkicuj, jak odwzorowałbyś to na kolejkę komunikatów z dwiema rodzinami kolejek.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|----------------|------------------------|
| Trwałe wykonywanie | "Utrwal stan programu" | Silnik zapisuje stan po każdym super-kroku; odtwarzanie po awarii jest deterministyczne. |
| Super-krok | "Granica transakcyjna" | Jednostka pracy między punktami kontrolnymi. Termin z LangGraph. |
| thread_id | "Identyfikator uruchomienia agenta" | Klucz wiążący punkty kontrolne i logikę wznawiania. |
| Idempotentność | "Bezpieczne do ponowienia" | Powtórzenie efektu ubocznego daje ten sam wynik co jedna próba. |
| Wzorzec outbox | "Oddziel efekty uboczne" | Zapisz intencję w tabeli; oddzielny wykonawca realizuje i oznacza jako zrobione. |
| Dostarczenie co najmniej raz | "Możliwe duplikaty" | Semantyka kolejki komunikatów; klucz deduplikacji czyni konsumenta efektywnie jednorazowym. |
| Wdrożenie tęczowe | "Nakładające się wersje" | Wiele wersji środowiska uruchomieniowego równocześnie podczas długotrwałych obciążeń pracy. |
| Włókno async | "Kooperatywne ustępowanie" | Współbieżność w trybie użytkownika; tania w porównaniu do wątków dla obciążeń I/O. |
| Punkt kontrolny | "Migawka stanu" | Sserializowany stan na granicy super-kroku; klucz do wznowienia. |

## Dalsza lektura

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — projekt środowiska uruchomieniowego LangGraph
- [MegaAgent](https://arxiv.org/abs/2408.09955) — kolejka producent-konsument per-agent; dwuwarstwowa koordynacja przy tysiącach równoczesnych agentów
- [Matrix](https://arxiv.org/abs/2511.21686) — zdecentralizowany framework z kolejkami komunikatów jako podłożem koordynacji
- [Dokumentacja Temporal](https://docs.temporal.io/) — referencyjny silnik przepływu pracy dla trwałego wykonywania
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — lekcje produkcyjne, w tym wdrożenia tęczowe