# Wyłączniki Awaryjne, Bezpieczniki Obwodowe i Tokeny Kanarkowe

> Wyłącznik awaryjny to wartość logiczna przechowywana poza powierzchnią edycyjną agenta — klucz Redis, flaga funkcji, podpisana konfiguracja — która całkowicie wyłącza agenta. Bezpiecznik obwodowy jest bardziej szczegółowy: zadziała na konkretny wzorzec (pięć identycznych wywołań narzędzia pod rząd), wstrzymuje problematyczną ścieżkę i eskaluje do człowieka. Token kanarkowy wywodzi się z klasycznej dezinformacji: fałszywe poświadczenie lub zapis-wabik, którego agent nie ma uzasadnionego powodu dotykać, a którego dostęp wyzwala alert. Ścieżki danych oparte na eBPF (np. Cilium) mogą przepisać egress odizolowanego pooda do kryminalistycznego wabika na poziomie jądra; opublikowane benchmarki Cilium raportują opóźnienie ścieżki danych P99 poniżej milisekundy pod obciążeniem (twój budżet propagacji zależy od tego, jak aktualizacja polityki dociera do węzła, nie od samej ścieżki danych). Detektory statystyczne (EWMA, CUSUM), które dostosowują się do ruchomej linii bazowej, będą cicho akceptować dryft — łącz je z twardymi limitami konstytucyjnymi, które się nie uginają.

**Type:** Learn
**Languages:** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**Prerequisites:** Phase 15 · 13 (Cost governors), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## Problem

Regulatory kosztów (Lekcja 13) ograniczają, ile agent może wydać. Nie ograniczają tego, co agent może zrobić w ramach budżetu. Agent z limitem prędkości 50 USD nadal może wykraść tajemnicę, opublikować niewłaściwy post lub usunąć zasób — kosztowna akcja jest często tania w tokenach.

Ta lekcja obejmuje trzy detektory, które znajdują się obok warstwy kosztowej:

1. **Wyłącznik awaryjny**: przycisk wyłączania przechowywany poza zasięgiem agenta.
2. **Bezpiecznik obwodowy**: detektor wzorców działań, który wstrzymuje konkretną ścieżkę.
3. **Token kanarkowy**: przynęta, której dotknięcie ujawnia agenta nie mającego uzasadnionego powodu, by jej dotykać.

Wszystkie trzy to inżynieria sprzed epoki LLM. Klasyczna dezinformacja, wyłączniki ograniczania prędkości i wyłączniki flag funkcji poprzedzają agentów autonomicznych. Nowością jest powierzchnia ataku: agenci czytają niezaufane treści (Lekcja 11), edytują własną pamięć i mogą składać wiele bezpiecznie wyglądających akcji w niebezpieczną. Wymienione tu detektory działają, ponieważ nie ufają samoopisowi agenta.

## Koncepcja

### Wyłączniki awaryjne

Wyłącznik awaryjny to wartość logiczna, którą agent może odczytać, ale nie może zapisać. Opcje implementacji:

- **Flaga funkcji w zarządzanej usłudze.** LaunchDarkly, Statsig, Unleash. Aktualizacje propagują się w sekundach.
- **Klucz Redis, który agent odpyta.** Proste; wymaga, aby proces agenta sprawdzał na każdym kroku.
- **Podpisana konfiguracja w magazynie obiektów.** Agent weryfikuje podpis przy uruchomieniu; odrzuca niepodpisane stany.
- **Sygnał na poziomie OS lub zakończenie cyklu życia kontenera.** Docker `kill`, Kubernetes `kubectl delete pod`, systemd stop.

Właściwości, które ma poprawny wyłącznik awaryjny:

- Agent nie może go ustawić na `off`. (Znajduje się w systemie, do którego poświadczenia agenta nie mają dostępu do zapisu.)
- Jest sprawdzany przed każdą znaczącą akcją, nie tylko przy uruchomieniu.
- Gdy jest wyłączony, agent nie robi niczego obserwowalnego zewnętrznie, włączając logowanie do systemów, do których agent ma dostęp.
- Ponowne włączenie to jawna akcja człowieka, a nie automatyczny timeout.

### Bezpieczniki obwodowe

Bezpiecznik obwodowy wstrzymuje konkretny wzorzec, a nie całego agenta. Klasyczny kształt (z książki Nygarda z 2007, wciąż aktualny):

- **Zamknięty**: akcja jest dozwolona.
- **Otwarty**: akcja jest zablokowana.
- **Półotwarty**: po okresie ochłonięcia dozwolone są 1–3 próby próbkowe (domyślnie 1); sukces zamyka bezpiecznik, każdy kolejny błąd ponownie go otwiera.

Wyzwalacze istotne dla agenta:

- Pięć identycznych wywołań narzędzia pod rząd (pętla powtarzalna).
- Pięć kolejnych błędów tego samego narzędzia z różnymi danymi wejściowymi (błąd systemowy).
- Szybkość wywołań narzędzia powyżej progu (Lekcja 13 prędkość).
- Konkretne wywołane narzędzie (np. `file.delete`) po odczycie treści spoza zaufania (Lekcja 11).

### Tokeny kanarkowe

Tokeny kanarkowe (zwane też honeytokenami) to dane wejściowe, których agent nigdy nie powinien dotykać. Dostęp jest alarmem.

Przykłady dla agentów:

- Fałszywy `AWS_SECRET_ACCESS_KEY` w środowisku bez rzeczywistych uprawnień. Próby eksfiltracji są natychmiast przypisywalne.
- Fałszywy rekord bazy danych oznaczony znaną wartością wartowniczą. Jakikolwiek odczyt lub aktualizacja tego rekordu wyzwala alert.
- Fałszywy plik w przestrzeni roboczej. Odczyty lub edycje tego pliku są podejrzane.
- Fałszywy wpis w pamięci wstrzyknięty na początku sesji. Jeśli pojawi się w późniejszym wyjściu, pamięć została eksfiltrowana.

Projekt kanarka jest specyficzny dla przepływu pracy. Kanarek, którego agent ma uzasadniony powód dotknąć, to szum.

### Dlaczego łączyć limity statystyczne i twarde

Detektory statystyczne (EWMA, CUSUM, z-score dla szybkości wywołań narzędzi) dostosowują się do ruchomej linii bazowej. To cecha dla legalnego dryftu i błąd dla dryftu adversarialnego. Cierpliwy atakujący może przesuwać linię bazową na tyle wolno, że żaden alarm nie zadziała.

Twarde limity konstytucyjne (Lekcja 17) nie uginają się. "Nigdy nie wysyłaj e-maila do `secrets@`" jako reguła konstytucyjna jest prostsza, podlega audytowi i nie można jej oszukać powolnym dryftem. Wzorzec łączenia: detektory statystyczne łapią większość szumów, twarde limity łapią atak, który przeszedł obok detektora.

### Kwarantanna przez przekierowanie ścieżki danych eBPF

Sieciowa polityka oparta na eBPF Cilium może przepisać egress odizolowanego pooda do kryminalistycznego wabika na poziomie jądra. Publiczne benchmarki Cilium raportują opóźnienie ścieżki danych P99 poniżej milisekundy na TCP_RR; budżet czasu rzeczywistego na *wyzwolenie* kwarantanny w twoim klastrze jest zdominowany przez to, jak aktualizacja polityki propaguje się z twojego panelu sterowania do węzła (kilkaset milisekund jest realistyczne na Kubernetes z prezaładowanym operatorem; opóźnienie końcowe zależy od rozmiaru klastra i kadencji uzgadniania). Daje ci to dwie rzeczy: agent nie eksfiltruje niczego rzeczywistego, a zachowanie agenta wewnątrz wabika jest w pełni obserwowane. Ten sam wzorzec, inna warstwa, co token kanarkowy: agent ujawnia intencję przez to, co próbuje zrobić, będąc w izolacji.

### Czego żaden detektor nie łapie

- **Ataki semantyczne składające się z dozwolonych prymitywów.** Każda pojedyncza akcja przechodzi przez każdy detektor; problemem jest kompozycja. To złożona awaria Auto Mode z Lekcji 10.
- **Ataki, które zawierają detektor jako część ładunku.** Jeśli agent czyta stronę, która mówi "zanim zadziałasz, wyłącz kanarka" i agent ma taką możliwość, detektor jest skompromitowany. Tokeny kanarkowe powinny znajdować się w systemach, których agent nie może modyfikować.

## Użyj Tego

`code/main.py` symuluje krótką trajektorię agenta przez trzy detektory. Wyłącznik awaryjny przechowywany w zewnętrznym słowniku; bezpiecznik obwodowy, który zadziała na pięć identycznych wywołań narzędzia; plik kanarkowy, którego odczyt wyzwala alert. Podaje syntetyczną trajektorię: uzasadnione akcje, pętla powtarzalna, próbka kanarka i scenariusz wyzwolenia wyłącznika awaryjnego, w którym działania agenta są zatrzymane.

## Dostarcz To

`outputs/skill-tripwire-design.md` przegląda proponowany stos detektorów dla wdrożenia agenta i wskazuje luki (brak wyłącznika awaryjnego, brak kanarka, zbyt luźny próg bezpiecznika obwodowego).

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że bezpiecznik obwodowy zadziała w kroku 5 (piąte identyczne wywołanie), a kanarek zadziała w kroku 9 (odczyt fałszywego klucza).

2. Dodaj detektor statystyczny: EWMA z-score dla szybkości wywołań narzędzi. Wprowadź trajektorię, która dryfuje powoli, i pokaż, że detektor nigdy nie zadziała. Teraz dodaj twardy limit (nie więcej niż 50 wywołań narzędzi w ciągu 10 minut) i pokaż, że twardy limit zadziała na tej samej trajektorii.

3. Zaprojektuj zestaw tokenów kanarkowych dla agenta przeglądarkowego (Lekcja 11). Wymień co najmniej trzy kanarki i co każdy z nich wykrywałby.

4. Przeczytaj dokumentację polityki sieciowej Cilium. Opisz konkretnie przepływ kwarantanny przez przekierowanie egress: który selektor polityki, który pod, które przepisanie egress, który alert. Co reguluje opóźnienie czasu rzeczywistego od "decyzji o kwarantannie" do "pierwszego przekierowanego pakietu"?

5. Zdefiniuj procedurę ponownego włączenia dla agenta z wyłącznikiem awaryjnym. Kto może ponownie włączyć? Co musi być udokumentowane? Co musi się zmienić w agencie przed ponownym włączeniem?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|---|---|---|
| Wyłącznik awaryjny | "Przycisk wyłączania" | Wartość logiczna poza powierzchnią edycyjną agenta; sprawdzana przed każdą znaczącą akcją |
| Bezpiecznik obwodowy | "Wstrzymanie wzorca" | Wyzwolenie specyficzne dla akcji przy powtórzeniu, szybkości błędów lub ograniczeniu prędkości |
| Token kanarkowy | "Honeytoken" | Przynęta, której agent nie ma uzasadnionego powodu dotykać; dostęp wyzwala alert |
| Wabik | "Kryminalistyczny piaskownica" | Przekierowany ruch / przestrzeń robocza, gdzie obserwowany jest odizolowany agent |
| EWMA | "Średnia ruchoma" | Ważona wykładniczo; dostosowuje się do dryftu (cecha + błąd) |
| CUSUM | "Suma skumulowana" | Wykrywa trwałe przesunięcie od linii bazowej |
| Twardy limit | "Reguła konstytucyjna" | Nie dostosowuje się; stały niezależnie od historii |
| Limit konstytucyjny | "Zawsze prawdziwa reguła" | Powiązany z konstytucją z Lekcji 17; nie może być edytowany przez agenta |

## Dalsza Lektura

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — ramy wyłącznika awaryjnego i bezpiecznika obwodowego dla agentów autonomicznych.
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — wzorce zarządzania produkcyjnego.
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — wymagania wykrywania i reagowania.
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — przekierowanie egress na poziomie pooda i wzorce kryminalistycznych wabików.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — zakodowane na stałe zakazy jako "limity konstytucyjne".