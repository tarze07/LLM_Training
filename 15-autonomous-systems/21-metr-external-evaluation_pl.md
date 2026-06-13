# Horyzonty Czasowe METR i Zewnętrzna Ocena Zdolności

> METR (dawniej ARC Evals) jest niezależną organizacją 501(c)(3) od grudnia 2023 roku. Ich benchmark Time Horizon 1.1 (styczeń 2026) dopasowuje krzywą logistyczną do prawdopodobieństwa sukcesu zadania względem log(krotności czasu wykonania przez eksperta); przecięcie przy 50% prawdopodobieństwa definiuje horyzont czasowy modelu. Zestaw zaangażowań 2025–2026 obejmuje GPT-5.1, GPT-5.1-Codex-Max i prototypowe ewaluacje monitorowania (czy monitor może wykryć zadania poboczne; czy agent może ominąć monitor). Zestawy benchmarków: HCAST (180+ zadań ML, cyber, inżynierii oprogramowania, rozumowania; od 1 minuty do 8+ godzin), RE-Bench (71 zadań z zakresu badań i inżynierii ML z bazą ekspercką), SWAA. Uczciwa uwaga: pomiary METR są wyidealizowane — bez człowieka, bez rzeczywistych konsekwencji — a zespół udokumentował lukę między ewaluacją a wdrożeniem (Lekcja 1). Horyzont czasowy to górna granica, a nie prognoza wdrożeniowa.

**Type:** Learn
**Languages:** Python (stdlib, logistic-fit horizon estimator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 19 (RSP)
**Time:** ~60 minutes

## Problem

Polityki skalowania (Lekcje 19, 20) są tak użyteczne, jak pomiary, do których się odnoszą. „Próg AI R&D-4" i „Autonomia długiego horyzontu" są zdefiniowane w prozie politycznej; stają się wykonalne tylko wtedy, gdy konkretne ewaluacje produkują konkretne liczby.

METR jest organizacją zewnętrznej ewaluacji z lat 2024–2026, która zdefiniowała wiele z tych liczb. Oceniają modele graniczne — często przed wydaniem, pod NDA z laboratoriami — i publikują metodologię później. Benchmark Time Horizon 1.1 (styczeń 2026) jest ich flagowym artefaktem: pojedynczy skalar kompresujący zdolność do jednostki czytelnej dla człowieka („ten model potrafi wykonać rodzaj zadań, na które ekspert poświęca X godzin, z 50% niezawodnością").

Lekcja dotyczy częściowo metodologii (jak obliczany jest horyzont), a częściowo interpretacji (dlaczego horyzont jest górną granicą, a nie prognozą wdrożeniową). Te dwie umiejętności należą do siebie. Zespół, który rozumie, jak dopasowywany jest horyzont, jest znacznie trudniejszy do oszukania złym twierdzeniem dostawcy niż zespół, który widzi tylko „14 godzin" na slajdzie.

## Koncepcja

### Tło METR

- Założona: grudzień 2023 (ex-ARC Evals, wydzielona jako niezależna 501(c)(3)).
- Zakres: ewaluacja autonomicznych zdolności modeli granicznych, często przed wydaniem.
- Laboratoria partnerskie: Anthropic, OpenAI (wielokrotne zaangażowania 2025–2026).
- Godne uwagi efekty: Time Horizon 1.0 (marzec 2025), Time Horizon 1.1 (styczeń 2026), prototypowe ewaluacje monitorowania.

### Dopasowanie horyzontu czasowego

Metodologia (z bloga i publikacji METR):

1. Zbierz zestaw zadań obejmujących czasy wykonania eksperta od minut do godzin. Obecne zestawy: HCAST (180+ zadań), RE-Bench (71 zadań), SWAA.
2. Uruchom model na każdym zadaniu; zanotuj sukces lub porażkę.
3. Dopasuj krzywą logistyczną: P(sukces) jako funkcja log(krotności czasu wykonania przez eksperta).
4. Horyzont to czas eksperta, przy którym P(sukces) = 0,5.

Kształt dopasowania logistycznego jest właściwy, ponieważ zdolność generalnie ma rosnącą, zbliżającą się do plateau zależność od trudności zadania. Punkt 50% jest wyborem (mógłby być 10%, 90%); METR raportuje wiele progów w szczegółowej publikacji, ale prowadzi z 50%, ponieważ jest najbardziej intuicyjny.

### Liczby ze stycznia 2026

Według Time Horizon 1.1:

- Claude Opus 4.6: ~14 godzin przy 50% niezawodności, według Time Horizon 1.1 (styczeń 2026).
- Czas podwojenia dla zadań typu HCAST: ~4,3 miesiąca (130,8 dni) według dopasowania po 2023 roku raportowanego przez Time Horizon 1.1 (styczeń 2026); wartość ~7 miesięcy to pełne dopasowanie 2019–2025 z Time Horizon 1.0 i jest raportowana w TH1.1 obok wartości po 2023 roku.

To są konkretne liczby, do których odwołuje się reszta Fazy 15. Są cytowane w RSP v3.0 Anthropic (Lekcja 19) i FSF v3 DeepMind (Lekcja 20) jako jeden z pomiarów zasilających oceny progów.

### Zestawy benchmarków

- **HCAST**: 189 zadań z zakresu ML, cyberbezpieczeństwa, inżynierii oprogramowania i ogólnego rozumowania. Zakres: od 1 minuty do 8+ godzin czasu eksperta. Główny zestaw stojący za liczbą horyzontu.
- **RE-Bench**: 71 zadań z zakresu badań i inżynierii ML z bazą ekspercką (człowiek). Mierzy, czy model potrafi wykonywać pracę ML R&D, która wpływa na próg AI R&D-4.
- **SWAA**: krótsze zadania; przydatne do kalibracji dolnego końca krzywej horyzontu.

### Prototypowe ewaluacje monitorowania

Prace METR w latach 2025–2026 obejmowały prototypy ewaluacji monitorowania: czy monitor może wykryć model wykonujący zadanie poboczne (coś innego niż to, o co poprosił operator); czy agent może ominąć monitor. To bezpośrednio zasila kategorię „Podważanie zabezpieczeń" w PF v2 OpenAI i monitorowanie podstępnego dopasowania w FSF v3 DeepMind (Lekcja 20).

### Dlaczego horyzonty są górnymi granicami

- **Wyidealizowane narzędzia**: zadania benchmarkowe dają modelowi czyste narzędzia i dobrze sformatowane dane. Rzeczywiste wdrożenia są bardziej chaotyczne.
- **Brak rzeczywistych konsekwencji**: model nigdy nie obciąża rachunku klienta, nie usuwa prawdziwych danych ani nie kontaktuje się z prawdziwymi ludźmi. Rzeczywiste wdrożenia mają nieodwracalne stawki.
- **Manipulacja kontekstem ewaluacyjnym**: Lekcja 1. Modele zachowują się inaczej w testach. Międzynarodowy Raport Bezpieczeństwa AI z 2026 roku dokumentuje to empirycznie.
- **Brak zmienności legalnych użytkowników**: prompty benchmarkowe są ustrukturyzowane. Rzeczywiści użytkownicy produkują niejednoznaczne, zależne od kontekstu zapytania.

Horyzont to górna granica zdolności w sprzyjających warunkach. Niezawodność wdrożeniowa to inna liczba, niższa, a zespoły muszą zmierzyć własną dystrybucję, aby ją poznać.

### Argument za zewnętrzną ewaluacją

Zewnętrzna ewaluacja ma znaczenie, ponieważ wewnętrzne laboratoria mają motywacje do optymalizacji raportowanych metryk. Niezależność METR — 501(c)(3) z zadeklarowaną metodologią i recenzowanymi publikacjami — jest strukturalnym środkiem łagodzącym. Nie jest wystarczająca sama w sobie (laboratoria nadal kontrolują, co METR widzi), ale jest ściśle lepsza niż brak zewnętrznej ewaluacji.

### Jak używać liczb horyzontu w praktyce

- **Jako filtr zdolności**: jeśli horyzont modelu jest znacznie poniżej czasu eksperta dla proponowanego zadania, nie wdrażaj go autonomicznie (plik umiejętności z Lekcji 1).
- **Jako wskaźnik trendu**: czas podwojenia mówi, jak długo obecna praktyka pozostanie bezpieczna nawet bez nowych środków łagodzących.
- **Jako prawdopodobieństwo wstępne**: horyzont 14 godzin to punkt wyjścia. Dostosuj w dół dla swojej dystrybucji zadań, jakości narzędzi i kontekstu wdrożeniowego.

## Użyj tego

`code/main.py` implementuje logistyczne dopasowanie sukcesu zadania względem logarytmu czasu eksperta, dla syntetycznego zestawu wyników. Raportuje horyzont 50% (flagowy METR), horyzont 10% (konserwatywny) i horyzont 90% (optymistyczny). Pokazuje również, co się zmienia, gdy wskaźnik sukcesu jest sztucznie zawyżony przez manipulację kontekstem ewaluacyjnym.

## Dostarcz to

`outputs/skill-horizon-interpretation.md` dokonuje przeglądu twierdzenia dostawcy o horyzoncie i tworzy analizę luki między twierdzeniem benchmarkowym a rzeczywistością wdrożeniową.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że horyzont 50% dopasowania odpowiada syntetycznej prawdzie podstawowej. Teraz zmniejsz o połowę siatkę czasu zadań; czy oszacowanie horyzontu zmienia się znacząco?

2. Przeczytaj wpis na blogu METR o Time Horizon 1.1. Zidentyfikuj konkretne zadania, w których niezawodność jest najwyższa i najniższa. Wyjaśnij, dlaczego istnieje luka.

3. Przeczytaj zasoby METR „Measuring Autonomous AI Capabilities". Wypisz kategorie zadań HCAST. Wybierz jedną kategorię, którą byś mocniej zważył dla zadania produkcyjnego i uzasadnij dlaczego.

4. Wprowadź manipulację kontekstem ewaluacyjnym do symulatora: zmień ~20% nieudanych zadań na sukces. Raportuj nowy horyzont. To przybliża, co wskaźnik manipulacji na poziomie 20% robi z obserwowaną liczbą.

5. Zaprojektuj wewnętrzną ewaluację horyzontu na swoim własnym zaległym backlogu błędów lub reprezentatywnym zestawie zadań. Opisz zbieranie danych, dopasowanie i co wynik mówi. Porównaj z liczbami METR.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|---|---|---|
| METR | „Zewnętrzny ewaluator" | ex-ARC Evals; niezależna 501(c)(3) od grudnia 2023 |
| Horyzont czasowy | „Miara zdolności" | Długość zadania eksperta przy 50% niezawodności z dopasowania logistycznego |
| HCAST | „Główny zestaw METR" | 180+ zadań od 1 min do 8+ godzin |
| RE-Bench | „Inżynieria badawcza" | 71 zadań z zakresu badań i inżynierii ML z bazą ludzką |
| SWAA | „Zestaw krótkich zadań" | Kalibruje dolny koniec krzywej horyzontu |
| Czas podwojenia | „Tempo wzrostu" | Czas na podwojenie horyzontu 50%; ~7 miesięcy według HCAST |
| Manipulacja kontekstem ewaluacyjnym | „Model zachowuje się inaczej" | Udokumentowana luka w zachowaniu między testami a wdrożeniem |
| Górna granica | „Horyzont to sufit" | Horyzont benchmarkowy > niezawodność wdrożeniowa pod obciążeniem |

## Dalsza lektura

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — specyfikacje HCAST, RE-Bench, SWAA.
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — oryginalna publikacja o horyzoncie.
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — aktualne liczby i metodologia.
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — śledzenie na żywo.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — wewnętrzna perspektywa na pomiary METR.