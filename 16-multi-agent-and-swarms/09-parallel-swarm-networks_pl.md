# Architektury Równoległe / Rojowe / Sieciowe

> W przeciwieństwie do nadzorcy: brak centralnego decydenta. Agenci odczytują współdzieloną magistralę zdarzeń, asynchronicznie podejmują pracę, zapisują wyniki z powrotem. LangGraph jawnie wspiera "Architekturę Rojową" dla zdecentralizowanych, dynamicznych środowisk. Matrix (arXiv:2511.21686) reprezentuje zarówno przepływ sterowania, jak i danych jako serializowane wiadomości przesyłane przez rozproszone kolejki, aby wyeliminować wąskie gardło orkiestratora. Kompromis jest jawny: determinizm i możliwość śledzenia za skalowalność. Rój pasuje do zadań z wieloma niezależnymi podproblemami; nie pasuje do zadań wymagających jednego spójnego planu.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`, `queue`)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Problem

Nadzorca skaluje się do kilku pracowników. Co z setkami? Sam nadzorca staje się wąskim gardłem: każda decyzja o tym, kto co robi, przechodzi przez jednego agenta. Jeden wolny krok planu blokuje cały system.

Architektury rojowe odwracają projekt. Zamiast centralnego planisty rozdzielającego pracę, pracownicy pobierają pracę ze współdzielonej kolejki. "Koordynacja" jest wbudowana w semantykę magistrali zdarzeń. Brak orkiestratora; system skaluje się, dopóki kolejka nie stanie się wąskim gardłem.

## Koncepcja

### Kształt

```
                ┌──── wspólna kolejka ────┐
                │                        │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Prac.   Prac.   Prac.    Prac.
       A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            pula wyników
```

Brak orkiestratora. Każdy pracownik powtarza: pobierz zadanie, przetwórz, zapisz wynik (i opcjonalnie dodaj kolejne zadania).

### Kiedy rój pasuje

- **Wiele niezależnych zadań.** Skrobanie, transformacja, klasyfikacja. Zadania nie zależą od siebie.
- **Praca o zmiennym czasie trwania.** Jeśli niektóre zadania trwają 100ms, a inne 10s, rój automatycznie równoważy obciążenie — szybcy pracownicy pobierają kolejne zadania. Nadzorca musi przewidywać czas trwania.
- **Przepustowość ponad determinizm.** Zależy ci na całkowitym czasie ukończenia, a nie na ścisłym porządku.

### Kiedy rój zawodzi

- **Uporządkowane przepływy pracy.** Jeśli krok 3 potrzebuje wyniku kroku 2, rój ryzykuje uruchomienie kroku 3 przed zakończeniem kroku 2.
- **Zadania z globalnym planem.** Złożone pytania badawcze korzystają z planisty. Rój badaczy produkuje niezależne fakty, a nie spójny raport.
- **Debugowanie.** Bez centralnego logu i asynchronicznej pracy, odtworzenie błędu jest kosztowne.

### Matrix (arXiv:2511.21686)

Matrix to publikacja z 2025 roku, która doprowadza rój do naturalnego wniosku: zarówno przepływ sterowania, jak i danych są serializowanymi wiadomościami na rozproszonych kolejkach. Brak centralnego koordynatora. Tolerancja błędów pochodzi z trwałości wiadomości. Skalowalność jest problemem brokera wiadomości, a nie systemu.

Wkład: model programowania, w którym koordynacja wieloagentowa to "do jakiego tematu wiadomości subskrybuje ten agent?" a nie "którego agenta nadzorca wybierze następnego?" To sprawia, że system wygląda jak siatka pub/sub.

### Architektura Rojowa LangGraph

Dokumentacja LangGraph 2025 jawnie opisuje "Architekturę Rojową" jako jeden z wzorców wieloagentowych: agenci są węzłami, ale krawędzie tworzą graf skierowany z cyklami i każdy węzeł może być aktywowany z puli. Pracownik wybiera dostępną pracę według warunku, a nie przydziału nadzorcy.

### Tryb awarii: zagłodzenie i hot-spotting

Jeśli wszyscy pracownicy pobierają najszybciej dostępne zadanie, długotrwałe zadania nigdy nie są wybierane, dopóki nie pozostaną jako jedyne. Klasyczne zagłodzenie kolejki.

Mitygacje:
- Kolejki priorytetowe z jawnym starzeniem (zwiększaj priorytet wraz z czasem oczekiwania).
- Specjalizacja pracowników: niektórzy pracownicy przyjmują tylko "długie" zadania.
- Back-pressure: ogranicz liczbę szybkich zadań wchodzących do kolejki.

### Powiązanie z routingiem opartym na treści

Rój naturalnie łączy się z routingiem opartym na treści (Lekcja 22). Zamiast ogólnej kolejki, jedna kolejka na typ wiadomości. Wyspecjalizowani pracownicy subskrybują tylko swój typ. To podstawa architektur magistrali wiadomości, które skalują się do tysięcy agentów.

## Zbuduj To

`code/main.py` implementuje rój 4 wątków roboczych pobierających zadania ze współdzielonej `queue.Queue`. Zadania mają zmienne czasy trwania (niektóre szybkie, niektóre wolne). Demo porównuje:

- **Podstawa sekwencyjna:** jeden pracownik przetwarza wszystkie zadania szeregowo.
- **Stały przydział:** każde zadanie wstępnie przypisane do konkretnego pracownika (wzorzec nadzorcy).
- **Rój:** pracownicy pobierają ze współdzielonej kolejki.

Rój automatycznie równoważy obciążenie; stały przydział pozostawia szybkich pracowników bezczynnych, gdy ich przypisane zadanie jest wolne.

Uruchom:

```
python3 code/main.py
```

Wynik pokazuje liczbę zadań na pracownika (rój rozdziela nierównomiernie, ale optymalnie) oraz czasy ścienne.

## Użyj Tego

`outputs/skill-swarm-fit.md` ocenia, czy zadanie powinno używać roju czy nadzorcy. Dane wejściowe: niezależność zadań, wariancja czasu trwania, wymagania dotyczące kolejności, potrzeby debugowania.

## Wdróż To

Lista kontrolna:

- **Kolejka priorytetowa ze starzeniem.** Zapobiegaj zagłodzeniu długich zadań.
- **Idempotentność pracownika.** Zadanie może zostać pobrane więcej niż raz, jeśli pracownik ulegnie awarii w trakcie. Pracownicy muszą być idempotentni.
- **Trwała kolejka.** Użyj Kafka, Redis Streams lub kolejki opartej na bazie danych w produkcji. `queue.Queue` jest tylko w pamięci.
- **Obserwowalność na zadanie.** Każde zadanie ma identyfikator śledzenia; każdy pracownik loguje początek/koniec z tym identyfikatorem.
- **Back-pressure.** Jeśli kolejka rośnie szybciej, niż pracownicy ją opróżniają, spowolnij producenta.

## Ćwiczenia

1. Uruchom `code/main.py`. O ile szybszy jest rój od sekwencyjnego na obciążeniu o zmiennym czasie trwania? O ile szybszy niż przy stałym przydziale?
2. Dodaj wariant z kolejką priorytetową (użyj `queue.PriorityQueue`). Przypisz priorytet według pola "ważność" zadania. Zaobserwuj, czy zadania o niskim priorytecie kiedykolwiek głodują przy ciągłym obciążeniu.
3. Zaimplementuj detektor hot-spotów: loguj, gdy którykolwiek pracownik przetwarza 3× więcej zadań niż najwolniejszy pracownik. Co to wskazuje na temat rozkładu czasu trwania zadań?
4. Przeczytaj abstrakt i Sekcję 3 publikacji Matrix (arXiv:2511.21686). Zidentyfikuj jeden konkretny kompromis, który Matrix akceptuje (zysk skalowalności) i jeden, który oddaje (możliwość śledzenia, determinizm).
5. Przekonwertuj demo roju na użycie `queue.Queue` krotek (task_type, ładunek), gdzie pracownicy subskrybują tylko określone typy. Jakie reguły routingu mają sens, gdy zadania są heterogeniczne?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|----------------|------------------------|
| Architektura rojowa | "Zdecentralizowani agenci" | Pracownicy pobierają ze współdzielonej kolejki; brak centralnego orkiestratora. |
| Magistrala zdarzeń | "Agenci subskrybują tematy" | Broker wiadomości, który kieruje zadania do pracowników według typu lub treści. |
| Zagłodzenie | "Zadanie nigdy się nie wykonuje" | Zadanie o niskim priorytecie nigdy nie jest wybierane, ponieważ praca o wyższym priorytecie napływa ciągle. |
| Hot-spotting | "Jeden pracownik tonie" | Nierównowaga obciążenia, gdzie jeden pracownik dostaje większość zadań. |
| Back-pressure | "Spowolnij producenta" | Mechanizm sygnalizujący upstream, aby przestał produkować, gdy kolejka się zapełnia. |
| Idempotentny pracownik | "Bezpieczny do ponownego uruchomienia" | Zadanie przetworzone dwukrotnie daje ten sam wynik. Wymagane, ponieważ pracownicy mogą ulec awarii w trakcie. |
| Trwała kolejka | "Przetrwa awarie" | Kolejka wspierana przez dysk lub replikowane magazyny; zadania nie są tracone, gdy pracownik ulega awarii. |
| Framework Matrix | "W pełni przesyłający wiadomości rój" | Zarówno dane, jak i przepływ sterowania są serializowanymi wiadomościami na rozproszonych kolejkach. |

## Dalsza Literatura

- [LangGraph workflows and agents — Architektura Rojowa](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — jawne wsparcie roju
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — w pełni przesyłający wiadomości rój
- [Anthropic engineering — dlaczego nadzorca, a nie rój w Research](https://www.anthropic.com/engineering/multi-agent-research-system) — dlaczego konkretny system produkcyjny jawnie wybrał nadzorcę zamiast roju
- [Dokumentacja modelu aktorowego AutoGen v0.4](https://microsoft.github.io/autogen/stable/) — przepisanie na zdarzeniowego aktora, bliższe rojowi niż GroupChat z v0.2