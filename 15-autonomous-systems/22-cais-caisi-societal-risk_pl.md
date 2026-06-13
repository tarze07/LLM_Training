# CAIS, CAISI i Ryzyko w Skali Społecznej

> Center for AI Safety (CAIS, San Francisco, założone w 2022 roku przez Hendrycksa i Zhanga) publikuje czteroelementowe ramy ryzyka — złośliwe użycie, wyścigi AI, ryzyka organizacyjne, zbuntowane AI — oraz majowe oświadczenie z 2023 roku o ryzyku wyginięcia podpisane przez setki profesorów i liderów firm. Publikacje CAIS z 2026 roku: AI Dashboard do ewaluacji modeli granicznych, Remote Labor Index (ze Scale AI), Superintelligence Strategy Paper, newsletter AI Frontiers. Odrębny podmiot: NIST Center for AI Standards and Innovation (CAISI) — dobrowolne porozumienia i niejawne ewaluacje zdolności skierowane do rządu USA, skoncentrowane na ryzyku cyber, biologicznym i chemicznej broni. CAIS wskazuje ryzyko organizacyjne jako jedno z czterech głównych ryzyk: kultura bezpieczeństwa, rygorystyczne audyty, wielowarstwowe zabezpieczenia i bezpieczeństwo informacji są fundamentalne, ale rutynowo poświęcane na rzecz szybkości wdrożenia. California SB-53, jeśli podpisana, byłaby pierwszą stanową regulacją ryzyka katastroficznego w USA.

**Type:** Learn
**Languages:** Python (stdlib, four-risk inventory and mitigation matcher)
**Prerequisites:** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**Time:** ~45 minutes

## Problem

Lekcje 19 i 20 omówiły wewnętrzne polityki skalowania laboratoriów. Lekcja 21 omówiła niezależną ewaluację zdolności. Ta lekcja obejmuje trzecią perspektywę: organizacje społeczeństwa obywatelskiego i rządowe, które kształtują publiczną dyskusję i regulacyjną linię bazową dla katastroficznego ryzyka AI.

Dwa odrębne podmioty mają znaczenie. CAIS to non-profitowa organizacja badawcza publikująca ramy myślenia o ryzyku AI i koordynująca publiczne oświadczenia. CAISI to rządowe centrum USA w ramach NIST, które prowadzi dobrowolne porozumienia z laboratoriami i niejawne ewaluacje zdolności. Nazwy są podobne; misje nie pokrywają się. Praktyk powinien znać oba.

Praktyczna treść: czteroelementowe ramy ryzyka CAIS są najczęściej cytowaną taksonomią ryzyka w skali społecznej w literaturze. Kultura bezpieczeństwa i ryzyko organizacyjne są jednym z tych czterech elementów, i to ten, który jest najbardziej bezpośrednio pod kontrolą praktyka. SB-53 (Kalifornia) byłaby pierwszą stanową regulacją ryzyka katastroficznego w USA, jeśli podpisana; sposób sformułowania ustawy ma znaczenie, ponieważ regulacje stanowe historycznie prowadziły do działań federalnych w amerykańskiej polityce technologicznej.

## Koncepcja

### CAIS — Center for AI Safety

- Założone: 2022 w San Francisco, przez Dana Hendrycksa i współpracowników (nazwisko „Zhang" odnosi się do wczesnego współpracownika, nie obecnego współzałożyciela; zobacz stronę CAIS po obecne kierownictwo).
- Status: non-profit 501(c)(3).
- Godne uwagi osiągnięcie z 2023 roku: oświadczenie o ryzyku wyginięcia, współpodpisane przez setki badaczy i dyrektorów generalnych. Stwierdzało: „Łagodzenie ryzyka wyginięcia z AI powinno być globalnym priorytetem obok innych ryzyk w skali społecznej, takich jak pandemie i wojna nuklearna."
- Publikacje z 2026 roku: AI Dashboard do ewaluacji modeli granicznych, Remote Labor Index (razem ze Scale AI), Superintelligence Strategy Paper, newsletter AI Frontiers.

### Czteroelementowe ramy ryzyka

Ramy CAIS grupują katastroficzne ryzyko AI w cztery główne kategorie:

1. **Złośliwe użycie**: zły aktor używa AI do wyrządzenia szkody (synteza broni biologicznej, dezinformacja, cyberataki).
2. **Wyścigi AI**: presja konkurencyjna między laboratoriami, firmami lub narodami popycha wdrożenie poza punkt, w którym jest bezpieczne.
3. **Ryzyka organizacyjne**: wewnętrzna dynamika laboratorium (braki w kulturze bezpieczeństwa, niewystarczający audyt, niedofinansowane bezpieczeństwo) prowadzi do złego wdrożenia.
4. **Zbuntowane AI**: wystarczająco zdolne AI dąży do celów sprzecznych z dobrostanem człowieka.

To nie jest jedyna taksonomia; jest najczęściej cytowana. Kategorie nie wykluczają się wzajemnie — zbuntowane AI wyprodukowane przez organizację, która poświęciła audyt na rzecz szybkości w wyścigu, to wszystkie cztery.

### Gdzie znajduje się ryzyko organizacyjne

Z czterech kategorii ryzyko organizacyjne jest najbardziej wykonalne dla praktyków. Kultura bezpieczeństwa laboratorium, rygorystyczność audytu, warstwowość obrony i bezpieczeństwo informacji decydują, czy ich model jest wdrażany z kontrolami z Lekcji 10–18 faktycznie wdrożonymi, czy też te kontrole są pozycjami na liście kontrolnej, których nikt nie zweryfikował.

Konkretne dźwignie ryzyka organizacyjnego:

- **Kultura bezpieczeństwa**: czy członkowie zespołu czują się w stanie zgłosić obawę bez ryzyka dla kariery? Badania CAIS pokazują, że jest to silny predyktor pozostałych dźwigni.
- **Rygorystyczne audyty**: zewnętrzne i wewnętrzne. Audyty tylko wewnętrzne produkują optymistyczne raporty.
- **Wielowarstwowe zabezpieczenia**: żadna pojedyncza warstwa nie jest wystarczająca (przewodni motyw Fazy 15).
- **Bezpieczeństwo informacji**: wyciek wag modelu, wyciek danych ewaluacyjnych, wyciek technik omijania monitorów. RAND SL-4 z Lekcji 19 to konkretny standard.

### CAISI — Center for AI Standards and Innovation

- Działa w ramach NIST.
- Prowadzi dobrowolne porozumienia z laboratoriami granicznymi.
- Publikuje niejawne ewaluacje zdolności skoncentrowane na ryzyku cyber, biologicznym i chemicznej broni.
- Odrębny od CAIS; akronimy są zbieżne; sprawdź URL (nist.gov), aby potwierdzić, który czytasz.

Rola CAISI to publiczny, rządowy odpowiednik prywatnych zaangażowań METR z laboratoriami (Lekcja 21). Raporty CAISI są niejawne; raporty METR są często objęte NDA. Praktyk czytający oba otrzymuje pełniejszy obraz.

### California SB-53

Ustawa Senatu Kalifornii (sesja 2025–2026) adresuje katastroficzne ryzyko ze strony modeli granicznych. Kluczowe postanowienia w wersji roboczej:

- Konkretne progi zdolności uruchamiające obowiązki na poziomie stanowym.
- Ochrona sygnalistów dla pracowników laboratoriów AI.
- Wymogi raportowania incydentów dla katastroficznych awarii.

Jeśli podpisana, byłaby pierwszą stanową regulacją ryzyka katastroficznego w USA. Niezależnie od statusu podpisania, sposób sformułowania ustawy kształtuje podejście innych stanowych legislatur. Praktycy w Kalifornii powinni śledzić status ustawy; praktycy gdzie indziej powinni ją przeczytać, aby zrozumieć, jak prawdopodobnie będzie wyglądać amerykańska regulacja stanowa.

### Ryzyko w skali społecznej nie jest problemem jednowarstwowym

Przewodni motyw Fazy 15 — obrona w głąb — stosuje się również do warstwy społecznej. Żadna pojedyncza organizacja, regulacja ani ramy nie zamkną ryzyka katastroficznego. Ekosystem działa tylko wtedy, gdy:

- Laboratoria publikują polityki skalowania (Lekcje 19, 20).
- Zewnętrzni ewaluatorzy produkują pomiary (Lekcja 21).
- Społeczeństwo obywatelskie śledzi i nagłaśnia (CAIS).
- Rząd prowadzi programy dobrowolne i podstawową regulację (CAISI, SB-53).
- Praktycy budują wielowarstwowe kontrole (Lekcje 10–18).

To jest końcowa synteza dla tej fazy: każda poprzednia lekcja jest jedną warstwą w stosie, którego kompletność ma większe znaczenie niż siła którejkolwiek pojedynczej warstwy.

## Użyj tego

`code/main.py` implementuje małe narzędzie do inwentaryzacji ryzyka. Dla proponowanego wdrożenia oznacza je względem czterech kategorii ryzyka i zwraca listę kontrolną środków łagodzących. Jest pomocą do czytania ram, a nie substytutem ludzkiego osądu.

## Dostarcz to

`outputs/skill-societal-risk-review.md` dokonuje przeglądu wdrożenia pod kątem postawy wobec ryzyka w skali społecznej: których z czterech kategorii dotyka, jakie środki łagodzące są wdrożone, jaka jest ekspozycja na ryzyko organizacyjne.

## Ćwiczenia

1. Uruchom `code/main.py`. Wprowadź trzy syntetyczne wdrożenia na różnych skalach. Potwierdź, że oznaczenia czterech kategorii ryzyka odpowiadają twoim oczekiwaniom; zidentyfikuj jeden przypadek, w którym narzędzie oznacza zbyt mało lub zbyt wiele.

2. Przeczytaj w całości publikację CAIS o czterech kategoriach ryzyka. Wybierz jedną kategorię ryzyka i napisz dwa akapity o tym, co uważasz za najważniejsze wydarzenie 2026 roku w tej kategorii.

3. Przeczytaj aktualny projekt California SB-53. Zidentyfikuj jeden przepis, który twoim zdaniem wzmacnia postawę wobec ryzyka katastroficznego, i jeden, który ją osłabia. Uzasadnij oba.

4. Wybierz znane ci produkcyjne wdrożenie AI (twoje lub opublikowane). Oceń je względem pod-dźwigni ryzyka organizacyjnego: kultura bezpieczeństwa, rygorystyczność audytu, wielowarstwowe zabezpieczenia, bezpieczeństwo informacji. Która jest najsłabsza? Ile kosztowałoby doprowadzenie jej do normy?

5. Naszkicuj wersję czteroelementowych ram ryzyka na 2028 rok, która odzwierciedla jeden rok dodatkowych zdolności i jeden rok dodatkowego doświadczenia wdrożeniowego. Co byś dodał, usunął lub przegrupował?

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|---|---|---|
| CAIS | „Center for AI Safety" | Non-profit; czteroelementowe ramy ryzyka; oświadczenie o wyginięciu z 2023 |
| CAISI | „Rządowe bezpieczeństwo AI USA" | Centrum NIST; dobrowolne porozumienia; niejawne ewaluacje |
| Czteroelementowe ramy ryzyka | „Taksonomia CAIS" | złośliwe użycie, wyścigi AI, ryzyka organizacyjne, zbuntowane AI |
| Złośliwe użycie | „Zły aktor używa AI" | Broń biologiczna, dezinformacja, cyberataki |
| Wyścigi AI | „Presja konkurencyjna" | Laboratoria/firmy/narody popychają wdrożenie poza bezpieczeństwo |
| Ryzyko organizacyjne | „Wewnętrzna awaria laboratorium" | Kultura bezpieczeństwa, audyt, zabezpieczenia, bezpieczeństwo informacji |
| Zbuntowane AI | „Niedopasowany agent" | Zdolne AI dążące do celów sprzecznych z dobrostanem człowieka |
| California SB-53 | „Regulacja stanowa" | Ustawa 2025–2026; pierwsza stanowa regulacja ryzyka katastroficznego w USA, jeśli podpisana |

## Dalsza lektura

- [Center for AI Safety](https://safe.ai/) — instytucjonalna siedziba czteroelementowych ram ryzyka.
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — publikacja o czterech kategoriach ryzyka.
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — krótkie wspólne oświadczenie.
- [NIST CAISI](https://www.nist.gov/caisi) — rządowe centrum standardów i innowacji AI.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — łączy zobowiązania na poziomie laboratorium z ramami w skali społecznej.