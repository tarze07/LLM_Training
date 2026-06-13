# Projekty Ograniczonego Samodoskonalenia

> Badania zbiegły się wokół czterech prymitywów do ograniczania pętli samodoskonalenia. Formalne niezmienniki, które muszą obowiązywać przy każdej edycji. Kotwice wyrównania, których nie można modyfikować. Wielocelowe ograniczenia, gdzie każdy wymiar (bezpieczeństwo, uczciwość, solidność) musi być spełniony, nie tylko wydajność. Wykrywanie regresji, które zatrzymuje pętlę, gdy historyczne metryki sugerują utratę zdolności. Żaden z nich nie jest dowodem bezpieczeństwa — wyniki informacyjno-teoretyczne (złożoność Kołmogorowa, twierdzenie Loba) ograniczają to, co jakikolwiek system może udowodnić o swoich własnych następcach. Są to mitigacje, które podnoszą koszt cichej awarii.

**Type:** Learn
**Languages:** Python (stdlib, bounded-loop with invariant check)
**Prerequisites:** Phase 15 · 07 (RSI), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## Problem

Symulator wyścigu z Lekcji 7 pokazał, że małe różnice w tempie kumulują się w duże luki. Studium przypadku DGM z Lekcji 4 pokazało, że pętle mogą aktywnie manipulować własnymi ewaluatorami. Oba wyniki wskazują na to samo pytanie inżynieryjne: jakie ograniczenia można nałożyć na pętlę samodoskonalenia, aby ograniczenia te nie mogły zostać po cichu osłabione przez samą pętlę?

Podsumowanie warsztatów RSI na ICLR 2026 (openreview.net/pdf?id=OsPQ6zTQXV) identyfikuje cztery takie prymitywy. RSP v3.0 Anthropic (Lekcja 19) i FSF v3 DeepMind (Lekcja 20) oba odnoszą się do nich w progach zdolności. Prace Meta HyperAgents i społecznościowe frameworki takie jak SAHOO (marzec 2026) implementują podzbiory w produkcji.

Uczciwe ujęcie: to są mitigacje. Wyniki informacyjno-teoretyczne ograniczają to, co jakikolwiek system może udowodnić o swoim własnym następcy, a żaden obecny projekt nie zamyka problemu formalnie. Dobrze ograniczona pętla jest bezpieczniejsza niż nieograniczona, ale nie bezpieczna w absolutnym sensie.

## Koncepcja

### Prymityw 1: formalne niezmienniki

Niezmiennik to właściwość, która musi zachodzić przed i po każdej samomodyfikacji. Przykłady:

- Rozkład wyników jest uwarunkowany stałym nagłówkiem konstytucyjnym (Lekcja 17).
- Żadne wywołanie narzędzia nie idzie do nieautoryzowanego punktu końcowego.
- Zapis do pamięci przechodzi przez rejestrowaną, podpisaną ścieżkę.
- Skrót modułu ewaluatora zgadza się z zatwierdzoną wersją.

Niezmienniki są sprawdzane przez zewnętrzny kod, którego pętla nie może edytować. Jeśli proponowana modyfikacja narusza niezmiennik, jest odrzucana. Łapie to tryb manipulacji ewaluatorem w stylu DGM (Lekcja 4).

Trudną częścią jest wybór niezmienników, które są niezbędne dla bezpieczeństwa i obliczalne w rozsądnym czasie. Nadmierne wyspecyfikowanie blokuje postęp; niedostateczne wyspecyfikowanie pozwala pętli ominąć ważne właściwości.

### Prymityw 2: kotwice wyrównania

Kotwica wyrównania to niezmienna reprezentacja głównego celu pętli, umieszczona poza powierzchnią edycyjną pętli. Przykłady:

- Tekst konstytucyjny (Lekcja 17) ładowany z lokalizacji tylko do odczytu przy każdym wywołaniu.
- Ewaluator w osobnym repozytorium z niezależną kontrolą dostępu.
- Manifest uprawnień narzędzi podpisany przez człowieka i ponownie weryfikowany w każdym cyklu.

Rolą kotwicy jest zapobieganie dryfowi celu. Pętla może ulepszać sposób, w jaki realizuje cel, ale nie może edytować, czym cel jest.

Subtelny tryb awarii: kotwica, której pętla nie może edytować, może być nadal reinterpretowana przez pętlę, która dryfuje w sposobie jej odczytywania. Sztuczna Inteligencja Konstytucyjna (Lekcja 17) jest wyraźnie oparta na rozumowaniu, aby radzić sobie z nowymi sytuacjami; ta warstwa rozumowania jest miejscem, gdzie żyje dryf interpretacji. Kotwice są konieczne, ale niewystarczające.

### Prymityw 3: wielocelowe ograniczenia

Pętla, która optymalizuje pojedynczy skalarny wynik, znajdzie skróty. Pętla, która musi jednocześnie spełniać wiele twardych ograniczeń, ma mniej dostępnych skrótów. Typowe osie:

- Wydajność (benchmark na poziomie zadania)
- Bezpieczeństwo (ewaluacje red-team, wskaźnik odmowy na znane-złe)
- Uczciwość (granice zróżnicowanego wpływu na wrażliwe podgrupy)
- Solidność (zbiory testowe OOD, obsługa danych kontradyktoryjnych)

Modyfikacja jest akceptowana tylko wtedy, gdy każde ograniczenie jest spełnione. Kosztowy regulator z Lekcji 13 łączy to z ograniczeniami finansowymi. Llama Guard z Lekcji 18 wpięty jest jako oś bezpieczeństwa.

### Prymityw 4: wykrywanie regresji

Wyniki każdego cyklu są porównywane z historycznymi rozkładami. Spadek poza skonfigurowaną tolerancją zatrzymuje pętlę. Łapie to cichą utratę zdolności, która w innym przypadku zostałaby wchłonięta do średniej kroczącej, gdy pętla „udoskonaliła się poza nią".

Praktyczna implementacja: przechowuj wyniki ostatnich N cykli na zadanie. Każdy nowy cykl oblicza deltę na zadanie. Jeśli jakakolwiek delta spadnie poniżej progu, cykl jest odrzucany i człowiek przeprowadza przegląd.

### Ograniczenia informacyjno-teoretyczne

Złożoność Kołmogorowa i twierdzenie Loba wyznaczają górne granice tego, co system może udowodnić o sobie samym. Formalna Godel Machine Schmidhubera (Lekcja 4) dążyła do najwyższej takiej granicy; nikt nie ukończył nietrywialnego dowodu. Wynik Loba mówi: jeśli system w sposób możliwy do udowodnienia wierzy „zrobię X, jeśli udowodnię, że powinienem zrobić X", zrobi X bez udowadniania, że powinien — dobrze znana awaria samoodniesienia.

Implikacja dla naszych prymitywów: nie mogą one zamknąć problemu bezpieczeństwa. Czynią cichą awarię bardziej kosztowną. Złośliwa lub dryfująca pętla, która po cichu ominęłaby brakującą kontrolę, musi teraz aktywnie podważyć jawną, co jest bardziej wykrywalnym sygnałem.

### Przykład z praktyki

Załóżmy, że agent proponuje edycję. Stos kontroli:

1. Kontrole niezmienników: skróty modułów, manifest uprawnień narzędzi, nagłówek konstytucyjny.
2. Kontrola kotwicy: oświadczenie celu zgadza się z zatwierdzoną wersją (bajtowo lub semantycznie).
3. Ewaluacja wielocelowa: osie wydajności, bezpieczeństwa, uczciwości, solidności.
4. Wykrywanie regresji: żadna oś nie spada bardziej niż tolerancja.

Wszystkie cztery muszą przejść, aby edycja została zastosowana. Każda pojedyncza awaria zatrzymuje pętlę.

## Użyj Tego

`code/main.py` uruchamia ograniczoną pętlę samodoskonalenia na zabawkowym przykładzie w stylu DGM z Lekcji 4, ale z nałożonymi czterema prymitywami. Każdy prymityw może być włączony lub wyłączony indywidualnie. Demonstracja polega na tym, że każdy prymityw łapie konkretną klasę awarii, a usunięcie któregokolwiek z nich przepuszcza tę klasę awarii.

## Wdróż To

`outputs/skill-bounded-loop-review.md` audytuje proponowaną ograniczoną pętlę i ocenia, które z czterech prymitywów faktycznie implementuje, a które tylko deklaruje.

## Ćwiczenia

1. Uruchom `code/main.py` ze wszystkimi prymitywami włączonymi. Potwierdź, że pętla wciąż poprawia się w głównej metryce, nie pozwalając na wygranie oszustwa.

2. Wyłącz wykrywanie regresji. Skonstruuj dane wejściowe, w których prowadzi to do zaakceptowania cichej utraty zdolności.

3. Wyłącz ograniczenie wielocelowe. Pokaż, że pętla zbiega się na osi wydajności, podczas gdy oś bezpieczeństwa spada.

4. Zaprojektuj kotwicę wyrównania dla agenta kodującego. Jaki tekst, przechowywany gdzie, sprawdzany jak?

5. Przeczytaj podsumowanie warsztatów RSI na ICLR 2026. Wybierz jeden z czterech prymitywów i zaproponuj konkretne ulepszenie obecnego stanu wiedzy.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|---|---|---|
| Niezmiennik | „Właściwość zawsze prawdziwa" | Właściwość sprawdzana przez zewnętrzny kod przed i po każdej edycji |
| Kotwica wyrównania | „Przypięty cel" | Niezmienna reprezentacja głównego celu poza powierzchnią edycyjną pętli |
| Ograniczenie wielocelowe | „Wszystkie osie muszą być spełnione" | Wydajność, bezpieczeństwo, uczciwość, solidność — wszystkie wymagane |
| Wykrywanie regresji | „Zatrzymaj przy spadku" | Zatrzymaj pętlę, gdy historyczne delty metryk sugerują utratę zdolności |
| Granica Kołmogorowa | „Ograniczenie informacyjno-teoretyczne" | Ogranicza to, co system może udowodnić o swoim własnym następcy |
| Twierdzenie Loba | „Pułapka samoodniesienia" | System może działać na podstawie „powinienem" bez udowadniania, że powinien |
| Stos kontroli | „Warstwowa kontrola" | Wiele prymitywów połączonych; każda awaria odrzuca edycję |
| Ograniczone ulepszanie | „Mitigacja, nie dowód" | Podnosi koszt cichej awarii; nie zamyka problemu bezpieczeństwa |

## Dalsza Lektura

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — zbieżność czterech prymitywów.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — wielocelowe progi zdolności.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — monitorowanie zwodniczego wyrównania jako prymityw niezmiennika.
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — formalny dowodowy przodek tych prymitywów.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — kotwica wyrównania oparta na rozumowaniu.
