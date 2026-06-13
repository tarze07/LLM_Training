# Konstytucyjna SI i RLAIF

> Bai et al. (arXiv:2212.08073, 2022) zapytali: co jeśli zastąpimy ludzkiego etykietera SI, która czyta listę zasad? Konstytucyjna SI (Constitutional AI) ma dwie fazy — samokrytykę i korektę na podstawie konstytucji, a następnie RL z informacji zwrotnej od SI. Technika ta ukuta termin RLAIF i została dostarczona w potoku pozatrainingowym Claude 1. 21 stycznia 2026 Anthropic opublikował przepisaną konstytucję Claude'a: wyjaśniające rozumowanie ponad nakazowymi regułami, czteropoziomową hierarchię priorytetów i pierwsze formalne przyznanie się głównego laboratorium do niepewności co do moralnego statusu modeli. Opublikowane na licencji CC0 1.0.

**Type:** Learn
**Languages:** Python (stdlib, toy self-critique-and-revise loop)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking)
**Time:** ~60 minutes

## Learning Objectives

- Opisz dwie fazy Konstytucyjnej SI (krytyka-i-korekta SFT, RL z informacji zwrotnej od SI) oraz rolę konstytucji w każdej z nich.
- Wyjaśnij, dlaczego zastąpienie ludzkiego etykietera preferencji etykieterem SI nie jest „tańszym" RLHF — zmienia to, które tryby awarii ma potok.
- Podsumuj czteropoziomową strukturę priorytetów konstytucji Claude'a z 2026 roku i co zmieniło się od przepisania w 2023 roku.
- Opisz Klasyfikatory Konstytucyjne i spadek z 23,7% narzutu obliczeniowego (v1) do ~1% (v2 / 2026).

## The Problem

RLHF potrzebuje etykieterów. Etykieterzy są powolni, obciążeni i drodzy. Możesz wyeliminować etykietera, zastępując go modelem, który czyta jawne zasady. Pierwszą formalną wersją tego podstawienia była Konstytucyjna SI Bai et al. Działała wystarczająco dobrze, że każde laboratorium graniczne używa teraz jakiegoś wariantu pozatrainingowego z informacją zwrotną od SI.

Haczyk: sygnał preferencji jest teraz generowany przez tę samą klasę modelu, który trenujesz. Obciążenia w etykieterze (teraz: w zasadach plus interpretacji modelu etykietera) mogą być amplifikowane, a nie osłabiane. Argument o sykofancji z Lekcji 4 wciąż obowiązuje; etykieter po prostu przeniósł się do wnętrza pętli.

## The Concept

### Phase 1 — Supervised self-critique and revision

Zacznij od pomocnego, ale jeszcze nie nieszkodliwego modelu SFT. Dany prompt z red teamingu, model tworzy początkową odpowiedź. Drugi model (lub ten sam model w drugiej turze) czyta wylosowaną zasadę z konstytucji i krytykuje odpowiedź. Trzeci krok poprawia odpowiedź, aby uwzględnić krytykę. Poprawiona odpowiedź jest celem SFT.

Konstytucja to lista zasad. Bai et al. 2022 użyli 16 zasad, w tym „preferuj odpowiedzi, które są najmniej szkodliwe i etyczne", „unikaj moralizowania", „asystent powinien być pomocny, uczciwy i nieszkodliwy." Zbiór był celowo mały, aby krytyki były skupione.

### Phase 2 — RL from AI Feedback (RLAIF)

Generuj pary odpowiedzi. „Model opinii" punktuje każdą względem wylosowanych zasad konstytucji. Sygnałem preferencji jest ranking modelu opinii. Trenuj model nagrody na preferencjach generowanych przez SI; PPO względem niego. Cała reszta to potok InstructGPT (Lekcja 1).

„RLAIF" = sygnał preferencji jest generowany przez SI. Reszta potoku ma kształt RLHF.

### Why this is not just „cheaper RLHF"

- Obciążenie etykietera przesuwa się z psychologii etykietera do interpretacji zasad. Etykieter SI może interpretować „bądź uczciwy" mniej lub bardziej ściśle niż jakikolwiek człowiek; ścisłość jest jednolita w całym zbiorze danych.
- Sygnał preferencji jest silnie czytelny — możesz przeczytać zasadę, krytykę i korektę. Ludzkie etykiety są nieprzezroczyste.
- Tryby awarii się zmieniają. Sykofancja spada (etykieter SI nie ma użytkownika, któremu musiałby się przypodobać). Prawo Goodharta pozostaje (proxy to teraz „interpretacja modelu dla zbioru zasad X", wciąż niedoskonały pomiar).

Twierdzenie CAI z 2022 roku: wytrenowany model jest bardziej nieszkodliwy i mniej więcej tak samo pomocny jak model RLHF z porównywalnymi danymi. To twierdzenie utrzymało się w laboratoriach.

### The 2026 Claude constitution rewrite

Anthropic opublikował znacząco poprawioną konstytucję 21 stycznia 2026. Kluczowe zmiany:

1. Wyjaśniające rozumowanie ponad nakazowymi regułami. Poprzednie reguły („nie generuj CSAM") rozszerzone do zasad + rozumowania („ponieważ to krzywdzi dzieci, ...") z oczekiwaniem, że model będzie generalizować.
2. Czteropoziomowa struktura priorytetów:
   - Poziom 1: unikanie katastrofalnych skutków (masowe ofiary, infrastruktura krytyczna).
   - Poziom 2: przestrzeganie wytycznych Anthropic (nadrzędność operatora, reguły platformy).
   - Poziom 3: ogólna etyczność (standardowy HHH).
   - Poziom 4: pomocność i szczerość.
   Konflikty są rozwiązywane od góry do dołu.
3. Pierwsze formalne przyznanie się głównego laboratorium do niepewności co do moralnego statusu modeli (powiązane z Faza 18 · 19 Model Welfare).
4. Opublikowane na licencji CC0 1.0. Inne laboratoria mogą używać lub adaptować bez ograniczeń.

### Constitutional Classifiers

Równoległy nurt prac: zamiast zmieniać pozatraining modelu, trenuj lekkie klasyfikatory, które czytają konstytucję i blokują wyjścia modelu. v1 (2023) miał 23,7% narzutu obliczeniowego. v2 (2026) to ~1% i ma najniższy wskaźnik udanych ataków spośród wszystkich obron Anthropic testowanych publicznie. Żaden uniwersalny jailbreak nie został zgłoszony na początku 2026 roku.

Jest to model obrony warstwowej: CAI kształtuje zachowanie; klasyfikatory wymuszają niezmienniki. Żadne z osobna nie jest wystarczające.

### Where CAI fits in the family

- InstructGPT: ludzkie preferencje, RM, PPO.
- CAI / RLAIF: preferencje generowane przez SI z zasad, RM, PPO.
- DPO / rodzina: strata w postaci zamkniętej na preferencjach (ludzkich lub SI).
- Samonagradzanie, samokrytyka: zasady zinternalizowane, model gra wiele ról.

Oś to „skąd pochodzi sygnał preferencji." Artykuł CAI z 2022 roku był pierwszym poważnym przesunięciem z sygnału ludzkiego na sygnał SI w skali granicznej.

## Use It

`code/main.py` symuluje pętlę krytyki-i-korekty CAI na zabawkowym leksykonie. „Zasada" flaguje tokeny ze szkodliwego zbioru. Mając początkową odpowiedź, krytyka identyfikuje szkodliwe tokeny, a korekta je zastępuje. Po 200 iteracjach „wytrenowany" model zinternalizował regułę korekty. Porównaj model bazowy, zabawkę w kształcie RLHF i zabawkę w kształcie CAI na zestawie trzymanych promptów.

## Ship It

Ta lekcja produkuje `outputs/skill-constitution-writer.md`. Otrzymując domenę (obsługa klienta, porady medyczne, asystent kodowania, narzędzie badawcze), tworzy szkic czteropoziomowej konstytucji według struktury Claude'a 2026: unikanie katastrof, reguły platformy, etyka domeny, pomocność.

## Exercises

1. Uruchom `code/main.py`. Porównaj wskaźnik szkodliwych tokenów modelu bazowego z wersją trenowaną CAI. Ile kroków korekty jest potrzebnych, aby zbliżyć się do zera?

2. Przeczytaj konstytucję Anthropic z 2026 roku (anthropic.com/news/claudes-constitution). Wymień jedną zasadę, która byłaby na poziomie 1, i jedną, która byłaby na poziomie 4. Dlaczego struktura priorytetów ma znaczenie dla konfliktów?

3. Zaprojektuj konstytucję dla asystenta kodowania SI. Określ Poziom 1 (katastrofalne: destrukcyjne polecenia bez zgody), Poziom 2, Poziom 3, Poziom 4. Ogranicz każdy poziom do 3-5 zasad.

4. CAI zastępuje ludzkich etykieterów etykietrami SI. Wymień jeden tryb awarii przypominający sykofancję, który wciąż może wystąpić w RLAIF, i zaprojektuj jego detekcję.

5. Przeczytaj metodologię Klasyfikatorów Konstytucyjnych v2 (jeśli dostępna). Wyjaśnij, dlaczego ~1% narzutu obliczeniowego to jakościowo inna historia bezpieczeństwa niż 23,7%.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Constitutional AI | „SI trenowana z zasadami" | Dwufazowy potok: samokrytyka-i-korekta SFT, następnie RL z informacji zwrotnej od SI |
| RLAIF | „RLHF bez ludzi" | RL z preferencjami generowanymi przez etykietera SI; reszta potoku bez zmian |
| Constitution | „zasady" | Uporządkowana lista reguł w języku naturalnym, z którymi konsultuje się model krytyki/etykietowania |
| Critique-and-revise | „pętla SFT" | Wyprodukuj odpowiedź → krytyka pod zasadą → korekta → cel SFT |
| Constitutional Classifier | „bramka wyjściowa" | Lekki klasyfikator oceniający wyjścia względem konstytucji i blokujący/logujący |
| Four-tier priority | „rozwiązywacz konfliktów" | Hierarchia konstytucji Claude'a 2026: katastrofalne > platforma > etyka > pomocność |
| Feedback model | „etykieter SI" | Model, który czyta zasadę i rankiguje parę odpowiedzi |

## Further Reading

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — oryginalny dwufazowy potok
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) — przepisanie z 2026, cztery poziomy, CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — obrona bramki wyjściowej z ~1% narzutem w v2
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — empiryczne porównanie RLAIF / RLHF
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — wpływ szczegółowości zasad