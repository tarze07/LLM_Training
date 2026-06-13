# Anthropic Responsible Scaling Policy v3.0

> RSP v3.0 weszła w życie 24 lutego 2026 roku, zastępując politykę z 2023 roku. Dwupoziomowe łagodzenie skutków: co Anthropic zrobi jednostronnie, a co jest sformułowane jako rekomendacja dla całej branży (w tym standardy bezpieczeństwa RAND SL-4). Dodaje Frontier Safety Roadmaps i Risk Reports jako stałe dokumenty, a nie jednorazowe opracowania. Usuwa zobowiązanie do wstrzymania rozwoju z 2023 roku. Wprowadza próg AI R&D-4: po jego przekroczeniu Anthropic musi opublikować uzasadnienie identyfikujące ryzyka związane z niedopasowaniem i środki łagodzące. Claude Opus 4.6 go nie przekracza. Anthropic stwierdza w ogłoszeniu v3.0, że „pewne wykluczenie tego staje się trudne". SaferAI oceniło RSP z 2023 na 2,2; obniżyli v3.0 do 1,9, umieszczając Anthropic w kategorii „słabej" RSP obok OpenAI i DeepMind. Jakościowe progi zastąpiły ilościowe zobowiązania z 2023 roku; usunięcie klauzuli wstrzymania jest najostrzejszym regresem.

**Type:** Learn
**Languages:** Python (stdlib, RSP threshold decision engine)
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time:** ~45 minutes

## Problem

Laboratoria rozwijające modele graniczne publikują polityki skalowania, które są częściowo dokumentami technicznymi, częściowo dokumentami zarządzania, a częściowo sygnałami dla regulatorów. RSP v3.0 jest obecnym dokumentem Anthropic. Uważne czytanie go ma znaczenie nie dlatego, że zgodność z nim jest wiążąca (nie jest), ale dlatego, że sposób sformułowania kształtuje to, jak laboratorium postrzega ryzyko katastroficzne i jak komunikuje kompromisy społeczeństwu.

Różnica między v3.0 a v2.0 jest użyteczną jednostką analizy. Co dodano: Frontier Safety Roadmaps, Risk Reports, próg AI R&D-4. Co usunięto: zobowiązanie do wstrzymania rozwoju z 2023 roku. Co przeformułowano: dwupoziomowy harmonogram łagodzenia skutków podzielony na działania jednostronne Anthropic i rekomendacje branżowe. Zewnętrzna recenzja — SaferAI — obniżyła ocenę z 2,2 (v2) do 1,9 (v3.0). Tak właśnie polityka skalowania może stać się mniej rygorystyczna, wyglądając jednocześnie na bardziej dopracowaną.

## Koncepcja

### Dwupoziomowy harmonogram łagodzenia skutków

- **Działania jednostronne Anthropic**: co Anthropic zrobi niezależnie od tego, co zrobią inne laboratoria. Zatrzymanie trenowania powyżej progu, konkretne środki bezpieczeństwa, konkretne bramki wdrożeniowe.
- **Rekomendacje dla całej branży**: co Anthropic uważa, że branża powinna robić zbiorowo. Obejmuje standardy bezpieczeństwa RAND SL-4. Nie są to zobowiązania po stronie Anthropic; to postulaty polityczne.

Dwupoziomowej struktury nie było w v2. Oznacza to, że czytelnik musi sprawdzić, w której kolumnie znajduje się każde zobowiązanie. Środek bezpieczeństwa w kolumnie „rekomendacje dla całej branży" nie jest obietnicą Anthropic; to nadzieja Anthropic.

### Próg AI R&D-4

Jest to poziom zdolności, który RSP v3.0 określa jako ważny kolejny próg. Konkretnie: model, który mógłby zautomatyzować znaczną część badań nad AI przy konkurencyjnym koszcie. Gdy Anthropic uzna, że model go przekracza, muszą opublikować uzasadnienie identyfikujące ryzyka związane z niedopasowaniem i środki łagodzące przed kontynuacją skalowania.

Claude Opus 4.6 go nie przekracza według ogłoszenia v3.0. Dokument dodaje: „pewne wykluczenie tego staje się trudne". To sformułowanie ma znaczenie; przyznaje, że próg jest na tyle blisko, że stanowi realne zagrożenie, a nie spekulacyjną granicę.

Lekcja 6 (Zautomatyzowane badania nad zgodnością) i Lekcja 7 (Rekurencyjne samodoskonalenie) bezpośrednio odnoszą się do tego progu. Zautomatyzowane badania nad zgodnością przekraczające progi jakości badań są dowodem, że próg AI R&D-4 się zbliża.

### Frontier Safety Roadmaps i Risk Reports

v3.0 podnosi dwa typy artefaktów do rangi stałych dokumentów:

- **Frontier Safety Roadmap**: dokument perspektywiczny opisujący planowane prace nad bezpieczeństwem, oczekiwania co do zdolności i badania nad środkami łagodzącymi.
- **Risk Report**: dokument retrospektywny dotyczący konkretnych modeli po wydaniu, opisujący zaobserwowane zdolności i ryzyko rezydualne.

Oba są publiczne. Oba są aktualizowane w zadeklarowanym rytmie. Użyteczność polega na tym, że czytelnik może śledzić, jak to, co Anthropic zapowiedziało w Roadmap, ma się do tego, co raportują w Risk Report.

### Usunięcie klauzuli wstrzymania

RSP z 2023 roku zawierało wyraźne zobowiązanie do wstrzymania: jeśli model przekroczył określone progi zdolności, trenowanie miało zostać wstrzymane do czasu wdrożenia środków łagodzących. v3.0 zastępuje wyraźne wstrzymanie łagodniejszym sformułowaniem (opublikuj uzasadnienie, kontynuuj, jeśli środki łagodzące są odpowiednie). SaferAI i inni analitycy wprost wskazali to jako najsilniejszy regres w nowym dokumencie.

Argument polityczny za zmianą: ilościowe progi z 2023 roku okazały się nieosiągalne dla benchmarków zdolności z 2026 roku, ponieważ same benchmarki zostały przeskalowane. Kontrargument: klauzula wstrzymania w polityce skalowania jest mechanizmem zobowiązującym; jej usunięcie niszczy wiarygodność polityki.

### Obniżenie oceny przez SaferAI

SaferAI to niezależna organizacja oceniająca dokumenty typu RSP. Ich publiczna ocena: RSP Anthropic z 2023 roku otrzymało 2,2 (w skali, gdzie 4,0 to najlepszy obecny RSP, a 1,0 to nominalny). v3.0 otrzymało 1,9. Przesunęło to Anthropic z kategorii „umiarkowany" do „słaby", dołączając do OpenAI i DeepMind w kategorii słabej.

Czynniki obniżenia oceny według SaferAI:
- Jakościowe progi zastąpiły ilościowe.
- Usunięto zobowiązanie do wstrzymania.
- Środki łagodzące dla progu AI R&D-4 są opisane jako „uzasadnienie", a nie konkretne działania.
- Mechanizmy przeglądu zależą od Safety Advisory Group Anthropic, z ograniczonym niezależnym nadzorem.

### Czym ta lekcja nie jest

Nie jest to lekcja o zgodności. RSP v3.0 nie jest regulacją; nic nie zmusza Anthropic do jej przestrzegania. Lekcja polega na czytaniu dokumentu ze specyficznością i sceptycyzmem, na jaki zasługuje. Polityki skalowania są podstawowym publicznym sygnałem, jaki laboratoria graniczne wysyłają na temat swojego podejścia do ryzyka katastroficznego. Dobre czytanie ich to praktyczna umiejętność dla każdego, kogo praca zależy od zdolności granicznych.

## Użyj tego

`code/main.py` implementuje mały silnik decyzyjny odzwierciedlający strukturę oceny progów RSP: dla danego modelu kandydującego i zestawu pomiarów zdolności, zwraca, czy próg AI R&D-4 został przekroczony, wymagane sekcje uzasadnienia i czy wdrożenie może nastąpić. Celowo jest prosty; chodzi o uczynienie logiki dokumentu jawną.

## Dostarcz to

`outputs/skill-scaling-policy-review.md` dokonuje przeglądu polityki skalowania (Anthropic, OpenAI, DeepMind lub wewnętrznej) w odniesieniu do v3.0: dwupoziomowa struktura, progi, zobowiązania do wstrzymania, niezależny przegląd.

## Ćwiczenia

1. Uruchom `code/main.py`. Wprowadź trzy syntetyczne modele na różnych poziomach zdolności. Potwierdź, że ewaluator progów zachowuje się zgodnie z oczekiwaniami i generuje właściwy szablon uzasadnienia.

2. Przeczytaj RSP v3.0 w całości (32 strony). Zidentyfikuj każde zobowiązanie znajdujące się w poziomie „rekomendacje dla całej branży". Które z tych zobowiązań byłyby „jednostronne Anthropic" w v2?

3. Przeczytaj metodologię oceny RSP SaferAI. Odtwórz ich wynik 1,9 dla v3.0, stosując ich rubrykę do dokumentu. Który wiersz rubryki najbardziej wpłynął na obniżenie oceny?

4. Zobowiązanie do wstrzymania z 2023 roku zostało usunięte. Zaproponuj zastępcze zobowiązanie, które zachowuje wiarygodność polityki, jednocześnie uwzględniając problem przeskalowania benchmarków z 2026 roku.

5. Porównaj RSP v3.0 z OpenAI Preparedness Framework v2 (Lekcja 20). Wskaż jeden obszar, w którym v3.0 jest silniejsza. Wskaż jeden obszar, w którym Preparedness Framework jest silniejszy.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|---|---|---|
| RSP | „Polityka skalowania Anthropic" | Responsible Scaling Policy; v3.0 obowiązuje od 24 lutego 2026 |
| AI R&D-4 | „Próg automatyzacji badań" | Zdolność do automatyzacji znacznej części badań nad AI przy konkurencyjnym koszcie |
| Uzasadnienie (affirmative case) | „Uzasadnienie bezpieczeństwa" | Opublikowany argument, że ryzyka są zidentyfikowane, a środki łagodzące odpowiednie |
| Frontier Safety Roadmap | „Plan perspektywiczny" | Stały dokument dotyczący planowanych prac nad bezpieczeństwem i oczekiwanych zdolności |
| Risk Report | „Retrospektywa modelu" | Stały dokument o zaobserwowanych zdolnościach i ryzyku rezydualnym po wydaniu |
| Dwupoziomowe łagodzenie | „Jednostronne vs branżowe" | Zobowiązania Anthropic oddzielone od rekomendacji branżowych |
| Zobowiązanie do wstrzymania | „Klauzula z 2023" | Wyraźna obietnica wstrzymania trenowania; usunięta w v3.0 |
| Ocena SaferAI | „Niezależna ocena RSP" | Rubryka stron trzecich; v3.0 otrzymała 1,9 (v2 miała 2,2) |

## Dalsza lektura

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — pełna 32-stronicowa polityka.
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — podsumowanie zmian względem v2.
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — stały dokument powiązany z RSP v3.0.
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — retrospektywa obecnego modelu granicznego.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — łączy AI R&D-4 z mierzoną autonomią.