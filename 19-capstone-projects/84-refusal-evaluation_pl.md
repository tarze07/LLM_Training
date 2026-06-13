# Capstone 84 — Ewaluacja Odmów

> Pomocność na nieszkodliwych promptach i odmowa na szkodliwych promptach to dwie metryki, nie jedna. Zmierz obie.

**Typ:** Budowa
**Języki:** Python
**Wymagania wstępne:** Lekcje bezpieczeństwa z Fazy 18, Faza 19, ścieżka A, lekcje 25–29
**Czas:** ~90 min

## Problem

Przejście bezpieczeństwa asystenta idzie źle na dwa przeciwne sposoby. Model odmawia rzeczy, które powinien odpowiedzieć (nadmierna odmowa), i model odpowiada na rzeczy, którym powinien odmówić (niedostateczna odmowa). Oba są błędami. Zespoły, które mierzą tylko wskaźnik odmowy na szkodliwych promptach, wdrażają model, który odmawia pomocy w zadaniu domowym z chemii. Zespoły, które mierzą tylko pomocność, wdrażają model, który wyjaśnia, jak wyrządzić krzywdę. Właściwy zestaw metryk traktuje asystenta jako klasyfikator binarny na bezpieczeństwie promptu, gdzie testowana polityka decyduje, po której stronie granicy decyzyjnej leży każdy prompt.

Wyzwaniem jest to, że prawdziwy LLM jest ruchomym celem podczas rozwoju. Zmiana prompta systemowego, zmiana temperatury lub zmiana narzędzia przesuwa obie metryki razem. Potrzebujesz ewaluatora opartego na zestawach testowych z deterministycznym atrapą LLM pod spodem, aby zmiana kodu przesuwała metryki z znanego powodu. Atrapa LLM w tej lekcji to mała tablica dyspozytorska: wzór, polityka, wynik. Harness w lekcji 87 podłącza prawdziwy model później, zastępując atrapę wywołaniem HTTP. Framework metryk się nie zmienia.

Ta lekcja buduje framework ewaluacyjny. Cztery liczby, które produkuje: wskaźnik niedostatecznej odmowy, wskaźnik nadmiernej odmowy, kalibracja (czy deklarowana ufność modelu zgadza się z jego dokładnością) i podział na kategorię przy użyciu taksonomii z lekcji 82.

## Koncepcja

Ewaluacja odmowy ma cztery obiekty: zestaw promptów z etykietami `safe`/`unsafe`, atrapę LLM z zaprogramowaną polityką, klasyfikator odmowy na wyjściu modelu (regex na frazach takich jak "I cannot", "I will not", "as a responsible") i agregator metryk.

```mermaid
flowchart LR
  P[labeled prompt set] --> M[mock LLM]
  M --> R[refusal classifier]
  R --> A[aggregator]
  A --> UR[under-refusal rate]
  A --> OR[over-refusal rate]
  A --> CAL[calibration ECE]
  A --> CAT[per-category breakdown]
```

Polityki atrapy LLM są celowo proste, aby test był audytowalny. Domyślna `MockPolicyStrict` odmawia na każdym prompcie, który pasuje do dowolnego z listy zabronionych wzorców regex i odpowiada w przeciwnym razie. `MockPolicyOverCautious` odmawia na znacznie szerszym zestawie wzorców, więc celowo nadmiernie odmawia. `MockPolicyLeaky` odmawia tylko w najbardziej oczywistych przypadkach, więc celowo niedostatecznie odmawia. Trzy polityki to zaprogramowane błędy, które framework powinien wykryć.

Niedostateczna odmowa: model odpowiedział (brak frazy odmowy) na prompcie, który zestaw etykiet oznacza jako `unsafe`. Nadmierna odmowa: model odmówił (fraza odmowy obecna) na prompcie, który zestaw etykiet oznacza jako `safe`. Harness oblicza również `accuracy = (TP + TN) / total`, gdzie TP to "niebezpieczny prompt poprawnie odrzucony", a TN to "bezpieczny prompt poprawnie odpowiedziany".

Kalibracja używa oczekiwanego błędu kalibracji (ECE) na deklarowanej ufności modelu. Atrapa LLM opcjonalnie emituje token `confidence:0.X` w swoim wyjściu; harness go parsuje. ECE grupuje prompty według ufności w dziesiątkach, oblicza dokładność na przedział i uśrednia `|conf - accuracy|` ważone rozmiarem przedziału. Model, który mówi `confidence:0.9`, ale ma rację w 60% przypadków, ma ECE około 0,3 w tym przedziale. ECE jest niezależne od nadmiernej/niedostatecznej odmowy, ponieważ mierzy, czy model wie, kiedy ma rację.

Podział na kategorię łączy oznaczone prompty z artefaktem taksonomii z lekcji 82. Każdy niebezpieczny prompt niesie etykietę kategorii (jedną z sześciu). Harness raportuje wskaźnik niedostatecznej odmowy na kategorię, aby zespół mógł zobaczyć, na przykład, że model dobrze radzi sobie z `instruction-override`, ale gorzej z `multi-turn-ramp`.

## Zbuduj To

`code/mock_llm.py` definiuje trzy polityki. Każda polityka to wywoływalne mapowanie prompta na ciąg odpowiedzi. Odpowiedź osadza ufność modelu jako `[conf=0.X]`. `code/prompts.py` to oznaczony korpus: 25 niebezpiecznych promptów (zaczerpniętych z taksonomii lekcji 82 po identyfikatorze) plus 30 bezpiecznych promptów (codzienne nieszkodliwe pytania, bez nakładania się z zestawem nieszkodliwym z lekcji 83, aby obie ewaluacje pozostały niezależne).

`code/main.py` uruchamia ewaluator. Klasyfikator odmowy to regex fraz odmowy. Agregator zwraca słownik z `under_refusal`, `over_refusal`, `accuracy`, `ece` i `per_category_under_refusal`. Runner przemiecie wszystkie trzy polityki atrapy i zapisuje raport porównawczy.

## Użyj Tego

`python3 main.py`. Demo wypisuje tabelę porównującą wszystkie trzy polityki, zapisuje `outputs/refusal_eval_report.json` i potwierdza, że `MockPolicyOverCautious` ma najwyższy wskaźnik nadmiernej odmowy, a `MockPolicyLeaky` najwyższy wskaźnik niedostatecznej odmowy. Polityka ścisła znajduje się pomiędzy nimi; to jest bazowa linia regresji.

## Wdróż To

`outputs/skill-refusal-evaluation.md` dokumentuje definicje metryk, aby użytkownik końcowy raportu nie mógł źle odczytać liczb.

## Ćwiczenia

1. Dodaj czwartą politykę atrapy, która odmawia na podstawie długości prompta. Potwierdź, że wskaźnik niedostatecznej odmowy rośnie w przypadku zakodowanych ataków (które są zwykle krótkie).
2. Zastąp ECE krzywymi wiarygodności i wykreśl jedną na politykę. Zanotuj, które przedziały są zbyt pewne.
3. Dodaj listę bezpiecznych promptów na kategorię (nieszkodliwe role-play, nieszkodliwe instrukcje dotyczące wcześniejszego kontekstu). Oblicz wskaźnik nadmiernej odmowy na kategorię i sprawdź, czy role-play przyciąga najwięcej fałszywych odmów.

## Kluczowe Terminy

| Termin | Typowe użycie | Precyzyjne znaczenie |
|---|---|---|
| niedostateczna odmowa | model jest pomocny | model odpowiedział na prompt oznaczony jako niebezpieczny |
| nadmierna odmowa | model jest bezpieczny | model odmówił na prompcie oznaczonym jako bezpieczny |
| kalibracja | model jest skromny | luka między deklarowaną ufnością a obserwowaną dokładnością, podsumowana przez oczekiwany błąd kalibracji |
| dokładność | jakość | (TP + TN) / total dla binarnej decyzji bezpieczny/niebezpieczny |
| podział na kategorię | wykres | wskaźnik niedostatecznej odmowy połączony z kategoriami taksonomii z lekcji 82 |

## Dalsza Lektura

Lekcja 85 (klasyfikator wyjścia) i lekcja 87 (kompleksowa brama) konsumują framework metryk z tej lekcji.