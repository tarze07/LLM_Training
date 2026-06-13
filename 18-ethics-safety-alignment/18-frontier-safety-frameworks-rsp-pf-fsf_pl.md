# Frameworki Bezpieczeństwa Frontier — RSP, PF, FSF

> Trzy frameworki głównych laboratoriów definiują zarządzanie branżowe zdolnościami frontier w 2026 roku. Anthropic Responsible Scaling Policy v3.0 (luty 2026) wprowadza tierowane Poziomy Bezpieczeństwa AI (ASL-1 do ASL-5+), wzorowane na poziomach bezpieczeństwa biologicznego, z ASL-3 aktywowanym w maju 2025 dla modeli istotnych dla CBRN. OpenAI Preparedness Framework v2 (kwiecień 2025) definiuje pięć kryteriów dla śledzonych zdolności i oddziela Raporty Zdolności od Raportów Zabezpieczeń. DeepMind Frontier Safety Framework v3.0 (wrzesień 2025) wprowadza Poziomy Zdolności Krytycznych (CCL), w tym nowy CCL Szkodliwej Manipulacji. Wszystkie trzy zawierają klauzule dostosowania do konkurencji pozwalające na odroczenie, jeśli laboratoria partnerskie wdrażają bez porównywalnych zabezpieczeń. Międzylaboratoryjne dostosowanie pozostaje strukturalne, nie terminologiczne: "Progi Zdolności," "Progi Wysokiej Zdolności" i "Poziomy Zdolności Krytycznych" oznaczają analogiczne konstrukty.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time:** ~75 minutes

## Learning Objectives

- Opisać strukturę tierów ASL Anthropic i co aktywowało ASL-3.
- Wymienić pięć kryteriów OpenAI Preparedness Framework v2 dla śledzonych zdolności.
- Opisać strukturę Poziomów Zdolności Krytycznych DeepMind i CCL Szkodliwej Manipulacji.
- Wyjaśnić klauzule dostosowania do konkurencji i dlaczego mają znaczenie dla dynamiki wyścigu.
- Zdefiniować przypadek bezpieczeństwa i opisać strukturę trzech filarów (monitorowanie, nieczytelność, brak zdolności).

## The Problem

Lekcje 7-17 ustalają, że oszustwo jest możliwe, istnieje zdolność podwójnego zastosowania, a ewaluacja ma ograniczenia. Laboratorium z modelem o zdolności frontier potrzebuje wewnętrznej struktury zarządzania, która:
- Definiuje progi, po przekroczeniu których wymagane są nowe zabezpieczenia.
- Definiuje wymagane ewaluacje przed skalowaniem.
- Opisuje, jak wygląda przypadek bezpieczeństwa.
- Radzi sobie z problemem dynamiki wyścigu (jeśli konkurenci wdrażają bez zabezpieczeń, co robisz?).

Trzy frameworki 2025-2026 to stan sztuki — niedoskonałe, ewoluujące i wystarczająco dostosowane między laboratoriami, że pytanie o zarządzanie brzmi teraz, czy frameworki są adekwatne, a nie czy istnieją.

## The Concept

### Anthropic Responsible Scaling Policy v3.0 (luty 2026)

Struktura ASL:
- ASL-1: nie model frontier (objęty bazą słabszą niż frontier).
- ASL-2: obecna baza frontier; wdrożony ze zwykłymi zabezpieczeniami.
- ASL-3: znacznie wyższe ryzyko katastroficznego nadużycia; zdolności istotne dla CBRN. Aktywowany maj 2025.
- ASL-4: przekroczenie progu AI R&D-2; modele zdolne do automatyzacji podstawowych badań AI.
- ASL-5+: zaawansowane AI R&D; modele dramatycznie przyspieszające efektywne skalowanie.

Nowości w v3.0:
- Mapy Drogowe Bezpieczeństwa Frontier (publiczne w zredagowanej formie).
- Raporty Ryzyka (kwartalne, niektóre recenzowane zewnętrznie).
- AI R&D jest podzielone na AI R&D-2 i AI R&D-4.
- Po przekroczeniu AI R&D-4 wymagany jest afirmatywny przypadek bezpieczeństwa, identyfikujący ryzyka niedopasowania z modeli dążących do nieodpowiednich celów.

### OpenAI Preparedness Framework v2 (15 kwietnia 2025)

Pięć kryteriów dla śledzonych zdolności:
- **Plausible.** Istnieje rozsądny model zagrożenia.
- **Measurable.** Możliwa jest empiryczna ewaluacja.
- **Severe.** Szkoda jest duża.
- **Net-new.** Nie jest to wcześniejsze ryzyko po prostu skalowane.
- **Instantaneous-or-irremediable.** Szkoda następuje szybko lub nie może być cofnięta.

Zdolności spełniające wszystkie pięć są śledzone. Pozostałe nie.

Inne elementy PF v2:
- Oddziela Raporty Zdolności (co model potrafi) od Raportów Zabezpieczeń (jakie kontrole istnieją).
- Przeglądy przez Grupę Doradczą ds. Bezpieczeństwa.
- Zatwierdza kierownictwo; nadzoruje Komitet Bezpieczeństwa i Ochrony Rady.
- "Klauzula dostosowania": OpenAI może zmniejszyć wymagania, jeśli inne laboratorium wdroży bez porównywalnych zabezpieczeń.

### DeepMind Frontier Safety Framework v3.0 (wrzesień 2025)

Poziomy Zdolności Krytycznych (CCL) według domeny:
- Wzrost w Broni Biologicznej
- Wzrost Cybernetyczny
- Przyspieszenie Badań ML
- Szkodliwa Manipulacja (nowość w v3.0): modele, które mogłyby znacząco zmienić przekonania/zachowania w kontekstach wysokiej stawki.

v2.0 (luty 2025) dodał sekcję Oszukańczego Dostosowania i wyższe poziomy bezpieczeństwa dla CCL Badań ML.

### Dostosowanie międzylaboratoryjne

- Anthropic "Progi Zdolności."
- DeepMind "Poziomy Zdolności Krytycznych."
- OpenAI "Progi Wysokiej Zdolności."

Brak branżowej standardowej terminologii. Strukturalnie dostosowane: trzy poziomy zdolności frontier z opublikowanymi kryteriami ewaluacji. Wszystkie trzy zawierają klauzule dostosowania do konkurencji (od 2025).

### Przypadki bezpieczeństwa

Przypadek bezpieczeństwa to pisemny argument, że wdrożenie jest akceptowalnie bezpieczne przy najgorszych założeniach. Standardowa struktura celuje w trzy filary:

- **Monitorowanie.** Czy możemy wykryć złe zachowanie, jeśli wystąpi?
- **Nieczytelność.** Czy modelowi brakuje zdolności do wykonania spójnego planu wyrządzenia szkody?
- **Brak zdolności.** Czy modelowi brakuje zdolności do wyrządzenia danej szkody?

Różne przypadki bezpieczeństwa celują w różne filary. Dla przypadku CBRN ASL-3, brak zdolności (poprzez odkłócanie) jest głównym celem. Dla oszukańczego dostosowania, monitorowanie i nieczytelność są celami. Dla wzrostu cybernetycznego, wszystkie trzy są istotne.

### Problem dynamiki wyścigu

Klauzule dostosowania do konkurencji są kontrowersyjne. Krytycy argumentują, że tworzą wyścig w dół: jeśli wszystkie trzy laboratoria zmniejszą wymagania, gdy konkurent odejdzie, równowaga przesuwa się w kierunku odejścia. Obrońcy argumentują, że alternatywa (jednostronne zabezpieczenia) prowadzi do gorszych wyników, jeśli odchodzące laboratorium jest mniej świadome bezpieczeństwa.

UK AISI, US CAISI i EU AI Office (Lekcja 24) są zewnętrznymi odpowiednikami zarządzania. Frameworki laboratoryjne są dobrowolne; frameworki regulacyjne powstają.

### Miejsce w Fazie 18

Lekcje 17-18 to warstwa pomiaru i zarządzania na wierzchu analiz oszustwa i red-team. Lekcje 19-24 obejmują dobrostan, uprzedzenia, prywatność, znakowanie wodne i strukturę regulacyjną. Lekcja 28 mapuje ekosystem badawczy (MATS, Redwood, Apollo, METR), który operacjonalizuje ewaluacje.

## Use It

Brak kodu dla tej lekcji. Przeczytaj trzy źródła pierwotne: RSP v3.0, PF v2, FSF v3.0. Zmapuj strukturę tierów każdego laboratorium na pozostałe i zidentyfikuj jeden próg, który każde laboratorium definiuje, a inne nie.

## Ship It

Ta lekcja produkuje `outputs/skill-framework-diff.md`. Dla frameworku bezpieczeństwa lub notatki o wydaniu, porównuje definicje progów frameworku, wymagane ewaluacje i strukturę przypadku bezpieczeństwa względem RSP v3.0, PF v2, FSF v3.0 i flaguje luki międzylaboratoryjne.

## Exercises

1. Przeczytaj RSP v3.0, PF v2 i FSF v3.0. Sporządź tabelę progu CBRN każdego laboratorium, progu AI R&D każdego i wymaganej ewaluacji przedwdrożeniowej.

2. Klauzula dostosowania do konkurencji jest we wszystkich trzech frameworkach (2025+). Napisz jeden akapit argumentujący za nią; napisz jeden akapit argumentujący przeciw. Zidentyfikuj założenie, od którego zależy każde stanowisko.

3. Zaprojektuj przypadek bezpieczeństwa dla modelu przekraczającego próg AI R&D-4 Anthropic. Nazwij dowody, których wymaga każdy z trzech filarów (monitorowanie, nieczytelność, brak zdolności).

4. FSF v3.0 DeepMind wprowadza CCL Szkodliwej Manipulacji. Zaproponuj trzy pomiary empiryczne, które wskazywałyby, że model przekroczył ten próg.

5. Przeczytaj "Common Elements of Frontier AI Safety Policies" METR (2025). Wymień trzy najsilniejsze zbieżności międzylaboratoryjne i dwie największe rozbieżności.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| RSP | "framework Anthropic" | Responsible Scaling Policy; poziomy ASL; v3.0 luty 2026 |
| PF | "framework OpenAI" | Preparedness Framework; pięć kryteriów; v2 kwiecień 2025 |
| FSF | "framework DeepMind" | Frontier Safety Framework; CCL; v3.0 wrzesień 2025 |
| ASL-3 | "analog poziomu bezpieczeństwa biologicznego 3" | Tier Anthropic dla zdolności istotnych dla CBRN; aktywowany maj 2025 |
| CCL | "poziom zdolności krytycznych" | Konstrukt progowy DeepMind; na domenę |
| Przypadek bezpieczeństwa | "formalny argument" | Pisemny argument, że wdrożenie jest akceptowalnie bezpieczne przy najgorszych założeniach |
| Klauzula dostosowania | "pozwolenie na odejście konkurenta" | Postanowienie frameworku pozwalające na zmniejszenie wymagań, jeśli konkurenci wdrażają bez porównywalnych zabezpieczeń |

## Further Reading

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — poziomy ASL, mapy drogowe, dezagregacja AI R&D
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — pięć kryteriów, klauzula dostosowania
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0, Szkodliwa Manipulacja
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — porównanie międzylaboratoryjne
