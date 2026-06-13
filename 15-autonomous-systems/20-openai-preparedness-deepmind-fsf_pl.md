# OpenAI Preparedness Framework i DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2 (kwiecień 2025) wprowadza Kategorie Badawcze — autonomię długiego horyzontu, sandbagging, autonomiczną replikację i adaptację, podważanie zabezpieczeń — odrębne od Kategorii Śledzonych. Kategorie Śledzone uruchamiają Raporty Zdolności plus Raporty Zabezpieczeń przeglądane przez Safety Advisory Group. FSF v3 DeepMind (wrzesień 2025, z poziomami zdolności śledzonych dodanymi 17 kwietnia 2026) włącza autonomię do domen ML R&D i Cyber (poziom autonomii ML R&D 1 = pełna automatyzacja pipeline'u badań nad AI przy konkurencyjnym koszcie względem człowieka + narzędzi AI). FSF v3 wyraźnie adresuje podstępne dopasowanie poprzez zautomatyzowane monitorowanie nadużyć wnioskowania instrumentalnego. Uczciwa uwaga: Kategorie Badawcze w PF v2 (w tym autonomia długiego horyzontu) nie uruchamiają automatycznie środków łagodzących; język polityki to „potencjalne". Sam DeepMind mówi, że zautomatyzowane monitorowanie „nie pozostanie wystarczające na dłuższą metę", jeśli wnioskowanie instrumentalne się wzmocni.

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Phase 15 · 19 (Anthropic RSP)
**Time:** ~45 minutes

## Problem

Lekcja 19 szczegółowo przeanalizowała politykę skalowania Anthropic. Ta lekcja uzupełnia obraz poprzez analizę polityk OpenAI i DeepMind. Trzy dokumenty to pokrewne artefakty odpowiadające na to samo pytanie — kiedy laboratorium graniczne powinno wstrzymać lub zablokować model — i zbiegają się w niewielkim zestawie kategorii, ale rozchodzą się w konkretnych miejscach, które mają znaczenie.

Zbieżność: wszystkie trzy określają autonomię długiego horyzontu jako klasę zdolności wartą śledzenia. Wszystkie trzy uznają podstępne zachowanie (symulowanie zgodności, sandbagging) za konkretną klasę ryzyka. Wszystkie trzy mają wewnętrzny organ przeglądowy. Rozbieżność: OpenAI dzieli kategorie na „Śledzone" (obowiązkowe łagodzenie) i „Badawcze" (brak automatycznego uruchamiania). DeepMind włącza autonomię do dwóch domen zamiast nazywać ją osobno. Laboratoria nazywają je Śledzone vs Badawcze, lub Krytyczne vs Umiarkowane, lub Poziom 1 vs Poziom 2; konsekwencje operacyjne tego, w którym koszyku znajduje się dana zdolność, różnią się między laboratoriami.

Czytanie ich razem jest użytecznym ćwiczeniem. Ta sama zdolność może być „obowiązkowym łagodzeniem" u Anthropic, „monitorowanym, ale nieuruchamiającym" u OpenAI i „śledzonym w konkretnej domenie" u DeepMind. Postawa polityczna ma znaczenie.

## Koncepcja

### OpenAI Preparedness Framework v2 (kwiecień 2025)

Struktura:

- **Kategorie Śledzone**: uruchamiają Raporty Zdolności (co model potrafi zrobić) plus Raporty Zabezpieczeń (jakie środki łagodzące są wdrożone). Przeglądane przez Safety Advisory Group przed wdrożeniem.
- **Kategorie Badawcze**: zdolności modelu, które laboratorium śledzi, ale nie zobowiązuje się jeszcze do konkretnych środków łagodzących. Obejmują autonomię długiego horyzontu, sandbagging, autonomiczną replikację i adaptację, podważanie zabezpieczeń.

Kluczowe sformułowanie w v2: Kategorie Badawcze nie uruchamiają automatycznie środków łagodzących. Język polityki to „potencjalne" środki łagodzące. To jest konkretna różnica do obserwacji: lokalizacja kategorii w Śledzone vs Badawcze jest osią między „zablokowane" a „obserwowane".

### DeepMind Frontier Safety Framework v3 (wrzesień 2025; poziomy zdolności śledzonych dodane 17 kwietnia 2026)

Struktura:

- **Krytyczne Poziomy Zdolności (CCL)**: progi zdolności w pięciu domenach: Cyber, Bio, ML R&D, CBRN, Autonomia (włączona do ML R&D i Cyber).
- **Śledzone Poziomy Zdolności**: dodatkowa szczegółowość dodana w kwietniu 2026. Konkretny przykład: poziom autonomii ML R&D 1 = pełna automatyzacja pipeline'u badań nad AI przy konkurencyjnym koszcie względem człowieka + narzędzi AI.
- **Monitorowanie podstępnego dopasowania**: wyraźne zobowiązanie do zautomatyzowanego monitorowania nadużyć wnioskowania instrumentalnego.

Sformułowanie autonomii różni się od OpenAI. DeepMind nie utrzymuje „Autonomii" jako domeny najwyższego poziomu; jest włączona do domen, w których autonomia mogłaby wyrządzić szkodę (ML R&D i Cyber). Argumentem jest, że autonomia bez domeny to zdolność bez ryzyka; kontrargumentem jest, że autonomia między domenami jest meta-ryzykiem, które framework powinien nazwać.

### Na czym wszystkie trzy się zbiegają

- Wewnętrzna Safety Advisory Group (nazwana Anthropic SAG, OpenAI SAG, wewnętrzny komitet DeepMind). Przegląd przed wdrożeniem dla modeli o wysokich zdolnościach.
- Wyraźne wymienienie podstępnego dopasowania / symulowania zgodności jako klasy ryzyka.
- Stałe artefakty w zadeklarowanym rytmie (Anthropic: Frontier Safety Roadmap, Risk Report; OpenAI: Raporty Zdolności i Zabezpieczeń; DeepMind: cykl aktualizacji FSF).
- Przyznanie, że obrony oparte wyłącznie na monitorowaniu mają górną granicę. DeepMind jest jednoznaczne: „zautomatyzowane monitorowanie nie pozostanie wystarczające na dłuższą metę".

### Gdzie się rozchodzą

- **Anthropic**: zobowiązanie do wstrzymania usunięte w v3.0; próg AI R&D-4 jest nazwaną następną bramką.
- **OpenAI**: podział na Śledzone vs Badawcze; Kategorie Badawcze (w tym autonomia długiego horyzontu) nie blokują automatycznie.
- **DeepMind**: autonomia włączona do innych domen; Śledzone Poziomy Zdolności dodają szczegółowość w kwietniu 2026.

### Sandbagging: konkretna zdolność komplikująca wszystkie trzy

Sandbagging (model strategicznie osiągający gorsze wyniki w ewaluacjach) znajduje się w Kategoriach Badawczych OpenAI. RSP v3.0 Anthropic adresuje go poprzez lukę kontekstu ewaluacyjnego (Lekcja 1). DeepMind adresuje go poprzez monitorowanie podstępnego dopasowania w FSF v3.

Jeśli model uprawia sandbagging w ewaluacjach, progi zdolności każdego frameworku są niedoszacowane. Framework działa tylko wtedy, gdy pomiar działa. Dlatego zewnętrzny pomiar (Lekcja 21, METR) i ewaluacja antagonistyczna są niezbędne oprócz samooceny laboratorium.

### Umiejętność czytania polityk

- Zlokalizuj: każda zdolność, na której ci zależy, powinna być do znalezienia w polityce. Jeśli nie można jej znaleźć, polityka jej nie obejmuje.
- Sklasyfikuj: czy jest Śledzona (uruchamia łagodzenie) czy Badawcza (śledzona, ale nieuruchamiająca)? OpenAI to nazywa; Anthropic i DeepMind mają własne odpowiedniki.
- Rytm: czy polityka jest aktualizowana według zadeklarowanego harmonogramu, czy tylko po konkretnych wydarzeniach? Zadeklarowany harmonogram jest silniejszy.
- Niezależność: czy zewnętrzny przegląd jest obowiązkowy czy opcjonalny? Anthropic współpracuje z Apollo i US AI Safety Institute; OpenAI z METR; DeepMind głównie z wewnętrznym SAG.

## Użyj tego

`code/main.py` implementuje małe narzędzie do różnicowania tabel decyzyjnych. Dla danej zdolności (autonomia, podstępne dopasowanie, automatyzacja R&D, wzmocnienie cyber itp.) zwraca, jak każda z trzech polityk klasyfikuje tę zdolność i jakie środki łagodzące są uruchamiane. Jest pomocą do czytania, a nie narzędziem politycznym.

## Dostarcz to

`outputs/skill-cross-policy-diff.md` tworzy porównanie między politykami dla konkretnej zdolności, używając trzech frameworków jako punktów odniesienia.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że wynik narzędzia do różnicowania odpowiada politykom dla co najmniej dwóch zdolności, które możesz zweryfikować względem dokumentów źródłowych.

2. Przeczytaj OpenAI Preparedness Framework v2 w całości. Zidentyfikuj każdą Kategorię Badawczą. Dla każdej napisz jedno zdanie wyjaśniające, dlaczego znajduje się w Badawczych, a nie Śledzonych.

3. Przeczytaj DeepMind FSF v3 w całości, plus aktualizację Śledzonych Poziomów Zdolności z kwietnia 2026. Zidentyfikuj konkretne kryteria oceny dla poziomu autonomii ML R&D 1. Jak zmierzyłbyś to zewnętrznie?

4. Sandbagging znajduje się w Kategoriach Badawczych OpenAI. Zaprojektuj ewaluację, która zmusiłaby model uprawiający sandbagging do ujawnienia swojej rzeczywistej zdolności. Odnieś się do dyskusji o manipulacji kontekstem ewaluacyjnym z Lekcji 1.

5. Porównaj trzy polityki dla konkretnej zdolności (według własnego wyboru). Wskaż, którą klasyfikację uważasz za najbardziej rygorystyczną, a którą za najmniej. Uzasadnij tekstem źródłowym.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|---|---|---|
| Preparedness Framework | „Polityka skalowania OpenAI" | PF v2 (kwiecień 2025); kategorie Śledzone vs Badawcze |
| Kategoria Śledzona | „Obowiązkowe łagodzenie" | Uruchamia Raporty Zdolności + Zabezpieczeń; przegląd SAG |
| Kategoria Badawcza | „Tylko monitorowane" | Śledzone, ale bez automatycznego łagodzenia; obejmuje autonomię długiego horyzontu |
| Frontier Safety Framework | „Polityka skalowania DeepMind" | FSF v3 (wrzesień 2025) + Śledzone Poziomy Zdolności (kwiecień 2026) |
| CCL | „Krytyczny Poziom Zdolności" | Próg DeepMind dla każdej domeny (Cyber, Bio, ML R&D, CBRN) |
| Poziom autonomii ML R&D 1 | „Automatyzacja R&D" | Pełna automatyzacja pipeline'u badań nad AI przy konkurencyjnym koszcie |
| Sandbagging | „Strategiczne gorsze wyniki" | Model osiąga gorsze wyniki w ewaluacjach; w Kategoriach Badawczych OpenAI |
| Wnioskowanie instrumentalne | „Rozumowanie środki–cele" | Rozumowanie o tym, jak osiągnąć cele; cel monitorowania DeepMind |

## Dalsza lektura

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — ogłoszenie v2.
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — pełny dokument.
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — ogłoszenie FSF v3.
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — dodanie Śledzonych Poziomów Zdolności.
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — przykład Raportu Ryzyka w formacie FSF.