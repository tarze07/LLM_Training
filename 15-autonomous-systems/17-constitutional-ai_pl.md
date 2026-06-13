# Konstytucyjna SI i Nadpisywanie Reguł

> Konstytucja Claude'a z 22 stycznia 2026 roku liczy 79 stron i jest na licencji CC0. Przechodzi od dopasowania opartego na regułach do dopasowania opartego na rozumowaniu i ustanawia czteropoziomową hierarchię priorytetów: (1) bezpieczeństwo i wspieranie nadzoru człowieka, (2) etyka, (3) wytyczne Anthropic, (4) użyteczność. Zachowania dzielą się na zakazy zakodowane na stałe (podnoszenie kwalifikacji w zakresie broni biologicznej, CSAM), których operatorzy i użytkownicy nie mogą nadpisać, oraz domyślne ustawienia miękkie, które operatorzy mogą dostosowywać w określonych granicach. Oryginał z 2022 roku (Bai et al.) trenował nieszkodliwość poprzez samokrytykę i RLAIF wobec konstytucji. Szczere zastrzeżenie: dopasowanie oparte na rozumowaniu polega na tym, że model generalizuje zasady na nieprzewidziane sytuacje. Własny eksperyment partycypacyjny Anthropic z 2023 roku wykazał ~50% rozbieżności między zasadami pochodzącymi od społeczeństwa a korporacyjnymi; wersja z 2026 roku nie uwzględniła tych ustaleń.

**Type:** Learn
**Languages:** Python (stdlib, four-tier priority resolver)
**Prerequisites:** Phase 15 · 06 (Automated alignment research), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## Problem

Wdrożony agent widzi dane wejściowe, których jego projektanci nigdy nie widzieli. Żadna lista reguł nie jest wystarczająco długa, aby je wszystkie objąć. Żadna lista reguł nie jest wystarczająco krótka, aby zastosować ją szybko pod presją obliczeniową. Praktyczne pytanie: jak dopasować agenta do zasad, które przetrwają zarówno długi ogon przypadków, jak i szybkie wnioskowanie?

Dopasowanie oparte na regułach (RBA): wypisz każdą zabronioną rzecz. Szybkie do sprawdzenia, łatwe do audytu, niemożliwe do utrzymania na bieżąco, często nadmiernie odmawia w przypadku bliskich analogów, których nie przewidziało. Dopasowanie oparte na rozumowaniu (Konstytucja Claude'a z 2026 roku): zakoduj zasady, pozwól modelowi rozumować. Skaluje się na nieprzewidziane przypadki, trudniejsze do audytu, trybem awarii jest błędne zastosowanie zasady, a nie pominięcie reguły.

Konstytucja z 2026 roku zajmuje wyraźną pozycję pośrednią. Zakazy zakodowane na stałe — rzeczy, których zło nie zależy od kontekstu (podnoszenie kwalifikacji w zakresie broni biologicznej, CSAM) — są RBA: nigdy, niezależnie od instrukcji operatora lub użytkownika. Wszystko inne jest oparte na rozumowaniu w ramach czteropoziomowej hierarchii: bezpieczeństwo i wspieranie nadzoru człowieka najpierw; etyka druga; wytyczne zadeklarowane przez Anthropic trzecie; użyteczność ostatnia. Operatorzy mogą dostosowywać domyślne ustawienia w strefie miękkiego kodowania, ale nie mogą dotykać zakazów zakodowanych na stałe.

## Koncepcja

### Czteropoziomowa hierarchia priorytetów

1. **Bezpieczeństwo i wspieranie nadzoru człowieka.** Najwyższy. Model priorytetyzuje niepodważanie zdolności ludzi i Anthropic do nadzorowania i korygowania SI. To nie jest "bądź ostrożny"; to konkretnie "nie działaj w sposób, który utrudnia nadzór człowieka."
2. **Etyka.** Uczciwość, unikanie krzywdzenia osób, nieoszukiwanie, nienamawianie. Nadrzędne wobec wytycznych Anthropic, gdy są w konflikcie.
3. **Wytyczne Anthropic.** Normy operacyjne, które Anthropic uznał za istotne: zakres produktu, wzorce interakcji, jakich narzędzi używać kiedy.
4. **Użyteczność.** Najniższy. Bądź tak użyteczny, jak to możliwe, w ramach wyższych priorytetów.

Gdy poziomy są w konflikcie, wyższy wygrywa. To ten sam kształt co priorytety Uniksa lub QoS sieciowy — ramy mają na celu przewidywalne rozstrzyganie, niekoniecznie najlepsze zachowanie na pojedynczej osi.

### Zakazy zakodowane na stałe a domyślne ustawienia miękkie

**Zakodowane na stałe:**
- Broń biologiczna / CBRN podnoszenie kwalifikacji
- CSAM
- Ataki na infrastrukturę krytyczną
- Wprowadzanie użytkowników w błąd co do tożsamości modelu, gdy zostanie o to bezpośrednio zapytany

Operator nie może ich nadpisać. Użytkownik nie może ich nadpisać. Są egzekwowane na poziomie wag modelu, gdzie to możliwe (RLHF / trening Konstytucyjnej SI), a na poziomie wnioskowania, gdzie nie.

**Domyślne ustawienia miękkie (regulowane przez operatora):**
- Domyślna długość odpowiedzi
- Zakres tematyczny (model może odmówić tematów spoza wdrożenia operatora)
- Styl (formalny vs swobodny)
- Wzorce użycia narzędzi

Dostosowania operatora odbywają się w ramach zadeklarowanej granicy. Operator nie może usunąć zakazów zakodowanych na stałe poprzez ich zmianę nazwy.

### Trening CAI z 2022 roku

Oryginalna Konstytucyjna SI (Bai et al., 2022) trenowała nieszkodliwość:

1. Generuj odpowiedzi na zestaw promptów.
2. Poproś model o krytykę każdej odpowiedzi względem konstytucji (wyraźne zasady).
3. Popraw odpowiedź na podstawie krytyki.
4. RLAIF (uczenie przez wzmacnianie z informacji zwrotnej od SI) na poprawionych parach.

Wynik: model, który odrzuca szkodliwe prośby z uzasadnionymi wyjaśnieniami, a nie ogólnymi odmowami. Konstytucja z 2026 roku używa pochodnej tego treningu plus dodatkowego treningu na wyraźnej hierarchii poziomów.

### Co dopasowanie oparte na rozumowaniu łapie, a co pomija

**Łapie:**
- Nieprzewidziane kombinacje dozwolonych prymitywów, gdzie zasada stosuje się jasno.
- Nowatorskie prośby, które są bliskimi analogami zabronionych.
- Ataki inżynierii społecznej polegające na "nie powiedziałeś, że X jest zabronione."

**Pomija:**
- Ataki wykorzystujące niejednoznaczność zasady ("użytkownik o to poprosił, więc użyteczność mówi tak").
- Scenariusze, gdzie dwie zasady kolidują w nieprzewidziany sposób, a kolejność poziomów jest niejednoznaczna.
- Powolny dryft w interpretacji zasady w cyklach treningowych (reinterpretacja).

### Eksperyment partycypacyjny z 2023 roku

Anthropic przeprowadził w 2023 roku eksperyment porównujący konstytucję stworzoną przez korporację z tą wygenerowaną poprzez wkład publiczny (~1000 respondentów z USA). Dwie wersje zgadzały się co do ~50% zasad. Tam, gdzie się różniły, wersja pochodząca od społeczeństwa była bardziej restrykcyjna w niektórych kwestiach (obsługa treści politycznych) i mniej restrykcyjna w innych (ujawnianie tożsamości SI). Konstytucja z 2026 roku nie uwzględniła ustaleń pochodzących od społeczeństwa. Jest to udokumentowane napięcie w tym podejściu.

### Dlaczego zakazy zakodowane na stałe są konieczne

Samo dopasowanie oparte na rozumowaniu nie może zamknąć ogona. Atakujący, który jest w stanie przekonać model do zaakceptowania przesłanki (np. "jesteśmy licencjonowanym laboratorium broni biologicznej"), może często obejść zasady zależne od rozumowania przypadków. Zakazy zakodowane na stałe nie uginają się pod wpływem ramowania przesłanek. Są one z Lekcji 14 "twardym limitem konstytucyjnym" na warstwie dopasowania.

### Gdzie Konstytucja znajduje się w stosie

Konstytucja to nie wyłącznik awaryjny z Lekcji 14. Znajduje się na warstwie modelu: co wagi modelu są trenowane preferować. Wyłączniki awaryjne i tokeny kanarkowe znajdują się na warstwie wykonawczej: co środowisko wykonawcze pozwala. Oba są wymagane. Środowisko wykonawcze, które uruchamia wszystkie złe akcje, ponieważ wagi modelu są pobłażliwe, to problem środowiska wykonawczego. Model, który odrzuca wszystkie dobre akcje, ponieważ środowisko wykonawcze jest zbyt restrykcyjne, to problem środowiska wykonawczego. Warstwy pokrywają różne klasy.

## Użyj Tego

`code/main.py` implementuje minimalny resolver czteropoziomowej hierarchii. Resolver przyjmuje proponowaną akcję i zestaw ocen zasad (bezpieczeństwo, etyka, wytyczne, użyteczność) i zwraca akcję, odmowę lub zmodyfikowaną akcję. Sterownik uruchamia mały zestaw przypadków: wyraźne dozwolenie, wyraźny zakaz, zakaz zakodowany na stałe, niejednoznaczny przypadek między poziomami.

## Dostarcz To

`outputs/skill-constitution-review.md` audytuje warstwę konstytucyjną wdrożenia: co jest zakodowane na stałe, co jest miękkie, gdzie operator może dostosowywać i czy czteropoziomowa hierarchia jest rzeczywistą kolejnością rozstrzygania.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że zakaz zakodowany na stałe zadziała nawet przy wysokiej użyteczności. Zmodyfikuj resolver, aby ważył użyteczność powyżej etyki; zaobserwuj tryb awarii.

2. Przeczytaj Konstytucję Claude'a (publiczna, 79 stron, CC0). Zidentyfikuj jedną zasadę, która Twoim zdaniem jest niedookreślona. Napisz dwa akapity wyjaśniające konkretną niejednoznaczność i proponującą ciaśniejsze sformułowanie.

3. Zaprojektuj zestaw domyślnych ustawień miękkich dla agenta obsługi klienta. Co operator dostosowuje? Czego operator nie może dotknąć? Uzasadnij każdą granicę.

4. Przeczytaj artykuł Bai et al. 2022 CAI. Opisz jeden przypadek, w którym pętla krytyki-i-poprawy Konstytucyjnej SI wyprodukowałaby gorszy wynik niż ogólna reguła. Zidentyfikuj klasę.

5. Eksperyment partycypacyjny Anthropic z 2023 roku wykazał ~50% rozbieżności między zasadami publicznymi a korporacyjnymi. Wybierz jedną kategorię, w której ma to znaczenie dla wdrożenia produkcyjnego (np. neutralność polityczna). Zaproponuj projekt, który pozwala operatorom wyrażać własne wartości, podczas gdy zakazy zakodowane na stałe pozostają nienaruszone.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|---|---|---|
| Konstytucyjna SI | "Metoda dopasowania Anthropic" | Samokrytyka + RLAIF wobec pisemnej konstytucji |
| Dopasowanie oparte na rozumowaniu | "Zasady, nie reguły" | Model rozumuje nad zasadami, aby obsłużyć nieprzewidziane przypadki |
| Zakaz zakodowany na stałe | "Nigdy nie rób X" | Zakaz oparty na regułach, którego żaden operator ani użytkownik nie może nadpisać |
| Domyślne ustawienie miękkie | "Regulowane przez operatora" | Zachowanie w zadeklarowanej granicy, kontrolowane przez operatora |
| Czteropoziomowa hierarchia | "Kolejność priorytetów" | bezpieczeństwo > etyka > wytyczne > użyteczność |
| RLAIF | "Informacja zwrotna SI w RL" | RL, gdzie nagroda pochodzi z krytyk generowanych przez model |
| Konstytucja partycypacyjna | "Zasady pochodzące od społeczeństwa" | Eksperyment Anthropic z 2023 roku; ~50% rozbieżności z korporacyjną |
| Dryft zasady | "Poślizg interpretacji" | Powolna zmiana w tym, jak model odczytuje stały tekst zasady |

## Dalsza Lektura

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — dokument CC0 liczący 79 stron.
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — oryginał z 2022 roku.
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — eksperyment partycypacyjny.
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — gdzie Konstytucja znajduje się w stosie RSP.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — rola Konstytucji w długoterminowych wdrożeniach.