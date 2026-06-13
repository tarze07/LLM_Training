# LLaVA-OneVision: Pojedynczy Obraz, Wiele Obrazów, Wideo w Jednym Modelu

> Przed LLaVA-OneVision (Li et al., sierpień 2024) świat otwartych VLM miał oddzielne linie: LLaVA-1.5 dla pojedynczych obrazów, modele wieloobrazowe takie jak Mantis i VILA, modele wideo takie jak Video-LLaVA i Video-LLaMA. Każdy wygrywał swój benchmark i zawodził w innych. LLaVA-OneVision argumentował, że jeden program nauczania może wyszkolić jeden model do dominacji we wszystkich trzech scenariuszach, a że efekty transferu zadań (umiejętności z pojedynczego obrazu eksportowane do wideo, rozumowanie wieloobrazowe eksportowane do pojedynczego obrazu) biją sumę specjalistów. Przepis jest zwodniczo prosty: budżet tokenów wizualnych, który pozostaje stały we wszystkich scenariuszach, plus jawny program nauczania przechodzący od pojedynczego obrazu do OneVision (wiele obrazów) do wideo. Ta lekcja czyta budżet, program nauczania i emergentne zachowania.

**Type:** Build
**Languages:** Python (stdlib, solver budżetu tokenów + planista programu nauczania)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**Time:** ~180 minutes

## Learning Objectives

- Zaprojektować budżet tokenów wizualnych, który pozostaje stały dla wejść pojedynczego obrazu, wielu obrazów i wideo.
- Ułożyć program treningowy, który przenosi umiejętności z pojedynczego obrazu do wideo bez katastroficznego zapominania.
- Wyjaśnić, dlaczego jeden model bije specjalistów przy tej samej liczbie parametrów, gdy program nauczania jest zrobiony prawidłowo.
- Wymienić trzy emergentne zdolności raportowane przez LLaVA-OneVision: rozumowanie wielokamerowe, promptowanie znacznikami zestawu, agent zrzutów ekranu iPhone.

## The Problem

Obraz, wiele obrazów i wideo każdy obciążają model inaczej.

Pojedynczy obraz chce tokenów wysokiej rozdzielczości (AnyRes, ~2880 tokenów wizualnych), aby wychwycić OCR i drobne szczegóły. Budżet na próbkę: jeden obraz, 2880 tokenów.

Wiele obrazów chce kilku obrazów w umiarkowanej rozdzielczości (~576 tokenów każdy), aby rozumowanie między obrazami mieściło się w kontekście. Budżet na próbkę: 4-8 obrazów, 576 każdy, 2300-4600 tokenów.

Wideo chce wielu klatek w niskiej rozdzielczości (~196 tokenów na klatkę po pooling'u), aby uchwycić dynamikę czasową. Budżet na próbkę: 8-32 klatek, 196 każda, 1600-6200 tokenów.

Jeśli trenujesz osobne modele, wybierasz jeden budżet. Jeśli trenujesz jeden model, potrzebujesz, aby budżet skalował się sensownie między scenariuszami bez wysadzania kontekstu.

Przed OneVision domyślną odpowiedzią było "trenuj jeden scenariusz, ignoruj pozostałe." Video-LLaVA przerobił wideo na model obrazu z dodatkowymi etapami treningu. LLaVA-NeXT dodał obsługę wielu obrazów z kafelkowaniem. Żaden nie obsługiwał wszystkich trzech czysto.

## The Concept

### Budżet tokenów OneVision

LLaVA-OneVision wybiera ujednolicony budżet tokenów wizualnych wynoszący około 3000-4000 tokenów na próbkę, alokowany inaczej w zależności od scenariusza:

- Pojedynczy obraz: AnyRes-9 (3x3 kafelki + miniatura), każdy kafelek przy 384 z 729 paczkami, agresywny dwuliniowy pooling 2x2 → 182 na kafelek. Razem: 9 * 182 + 182 = 1820 tokenów. Lub AnyRes-4 przy 729-na-kafelek = 2916 + 729.
- Wiele obrazów: każdy obraz w umiarkowanej rozdzielczości (384, bez kafelkowania), 729 tokenów bez poolingu. Budżet 6 obrazów → 4374 tokenów.
- Wideo: 32 klatki przy rozdzielczości 384 z agresywnym 3x3 dwuliniowym poolem → 81 tokenów na klatkę. Razem: 32 * 81 = 2592 tokenów.

Alokacja utrzymuje w przybliżeniu stałą całkowitą liczbę tokenów. LLM nigdy nie widzi partii, która wysadza jego kontekst. Enkoder produkuje inną geometrię na scenariusz, ale LLM konsumuje ten sam budżet.

### Trzyetapowy program nauczania

LLaVA-OneVision trenuje w trzech etapach:

1. SFT pojedynczego obrazu (etap SI). Wszystkie dane to pojedynczy-obraz-plus-tekst. Trenuj na wejściu AnyRes o wysokiej rozdzielczości. To uczy percepcji, OCR i drobnoziarnistego rozumienia. Używa danych LLaVA-NeXT plus danych OneVision specyficznych dla pojedynczego obrazu.
2. SFT OneVision (etap OV). Mieszanka pojedynczy obraz + wiele obrazów + wideo (równomiernie próbkowane klatki). Trenuj na ujednoliconym budżecie tokenów. To uczy model obsługi heterogenicznych kształtów partii. Bez resetowania wag — kontynuuje z etapu SI.
3. Transfer zadań (etap TT). Kontynuuj z docelową mieszanką zadań, zazwyczaj cięższą na wiele obrazów lub wideo w zależności od produktu. Opcjonalne dostrojenie do wdrożenia.

Kluczowe: kolejność programu nauczania ma znaczenie. Trenowanie najpierw wideo lub wielu obrazów daje gorszą wydajność obrazu niż najpierw pojedynczy obraz, nawet z tymi samymi danymi. Praca abluje to jawnie.

### Dlaczego program nauczania działa

Trenowanie na pojedynczym obrazie buduje bazę percepcyjną. Tokeny paczek niosą drobnoziarniste cechy wizualne; LLM uczy się integrować je z tekstem. Wiele obrazów i wideo wprowadzają wyzwania strukturalne (który obraz jest który, co wydarzyło się najpierw), które są trudne do nauczenia bez silnej bazy percepcyjnej.

Jeśli trenujesz wszystkie scenariusze od zera razem, model niedopasowuje percepcję (ograniczone dane pojedynczego obrazu na partię) i nadpasowuje strukturę (dużo danych wielu obrazów / wideo). Wynik: model, który podąża za wzorcami rozumowania między obrazami, ale jest wizualnie płytki.

Uporządkowanie programu nauczania daje siłę percepcyjną z etapu SI, a następnie rozumowanie kompozycyjne/czasowe z etapu OV, bez utraty żadnego.

### Emergentne umiejętności między-scenariuszowe

Praca LLaVA-OneVision raportuje trzy emergentne zdolności:

1. Rozumowanie wielokamerowe. Trenowany na wielu obrazach + wideo osobno; podczas wnioskowania poproszony o rozumowanie sceny jazdy z wielu kamer. Model poprawnie integruje widoki, mimo że nigdy nie widział tego dokładnego formatu w treningu.
2. Promptowanie znacznikami zestawu. Użytkownik adnotuje obiekty na obrazie ponumerowanymi znacznikami; model rozumuje "co robi znacznik 3 w relacji do znacznika 7." Trenowany ani na znacznikach, ani na adnotacjach; nauczył się z kombinacji ugruntowania przestrzennego + odniesienia do wielu obrazów.
3. Agent zrzutów ekranu iPhone. Użytkownik dostarcza zrzut ekranu iPhone i prosi o zaplanowanie następnego kliknięcia. Trenowany na zrzutach UI, wideo przepływów pracy użytkownika i parach przed/po wielu obrazów. Generalizuje do przypadku użycia agenta.

To nie są trenowane zadania; emergują ze struktury kompozycyjnej programu nauczania.

### Pooling tokenów wizualnych

Budżet tokenów wymaga poolingu. OneVision używa dwuliniowej interpolacji na 2D siatce paczek: 24x24 = 576 paczek staje się 12x12 = 144 (czynnik 2x) lub 8x8 = 64 (czynnik 3x). Pooling jest wykonywany w przestrzeni siatki paczek, a nie przestrzeni tokenów, aby zachować lokalność.

Wybór czynnika poolingu na scenariusz jest sam w sobie hiperparametrem. Mniej poolingu = więcej tokenów = bogatsza reprezentacja. Więcej poolingu = mniej tokenów = więcej klatek / obrazów się mieści.

### LLaVA-OneVision-1.5

Kontynuacja z 2025 (LLaVA-OneVision-1.5, arXiv 2509.23661) jest "w pełni otwarta" w danych treningowych, wagach modelu i kodzie. Dorównuje zastrzeżonej luce na niektórych benchmarkach i demokratyzuje przepis. Ten sam program nauczania, więcej danych, lepszy bazowy LLM. Żadnych zmian w architekturze.

### Kontrast z Qwen2.5-VL

Qwen2.5-VL (Lekcja 12.09) dokonuje innych wyborów. Używa M-RoPE i dynamicznego FPS zamiast stałego poolingu. Jego budżet skaluje się z wejściem — minutowe wideo używa więcej tokenów niż 5-sekundowe wideo. LLaVA-OneVision ustala budżet i skaluje pooling. Oba działają; wymieniają konfigurowalność na przewidywalność.

## Use It

`code/main.py` to planista programu nauczania i budżetu dla VLM w stylu OneVision. Mając budżet tokenów na próbkę i docelową mieszankę scenariuszy (np. 40% pojedynczy obraz, 30% wiele obrazów, 30% wideo),:

- Alokuje rozdzielczość, czynnik poolingu i klatki na scenariusz.
- Sprawdza, czy każdy scenariusz mieści się w wspólnym budżecie.
- Raportuje oczekiwaną liczbę tokenów, FLOP-y LLM i które scenariusze są niedotokenizowane.
- Wypisuje harmonogram treningu etap po etapie.

Użyj go do planowania dostrojenia OneVision lub do sanityzacji kosztu na żądanie wdrożenia VLM.

## Ship It

Ta lekcja produkuje `outputs/skill-onevision-budget-planner.md`. Mając docelową dystrybucję zadań i budżet na próbkę, emituje czynnik AnyRes, pooling na klatkę, liczbę klatek wideo i wagi etapów programu nauczania. Użyj tego, gdy trenujesz lub dostrajasz ujednolicony VLM dla wielu scenariuszy.

## Exercises

1. Twój produkt obsługuje 80% pojedynczy obraz, 10% wiele obrazów (2-4 obrazy), 10% wideo (8-16 klatek). Zaprojektuj budżet tokenów. Gdzie umieściłbyś dodatkowy budżet zaoszczędzony na nie robieniu ciężkiego wiele obrazów?

2. Przeczytaj LLaVA-OneVision Sekcja 4.3 (emergentne zdolności). Zaproponuj czwartą emergentną umiejętność, którą program nauczania prawdopodobnie odblokowałby, ale praca nie raportowała.

3. Zamień kolejność programu nauczania — trenuj najpierw wiele obrazów, potem pojedynczy obraz, potem wideo. Przewidź, które benchmarki ulegną degradacji i dlaczego.

4. Praca raportuje benchmarki wideo trenowane na tylko 8 klatkach na próbkę. Czy to generalizuje do 30-sekundowych filmów podczas wnioskowania? Co psuje się pierwsze — budżet tokenów czy rozumowanie czasowe?

5. Dwuliniowy pooling paczek 24x24 do 12x12 to 4x redukcja na wymiar. Zaimplementuj pooling w stdlib Python i zweryfikuj, że średnia po każdym bloku 2x2 pasuje do wyjścia dwuliniowego.

## Key Terms

| Term | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| OneVision scenario | "Pojedynczy obraz, wiele obrazów lub wideo" | Jeden z trzech kształtów wejściowych obsługiwanych przez ujednolicony VLM; budżet pozostaje stały między nimi |
| Token budget | "Ile tokenów na próbkę" | Całkowita liczba tokenów wizualnych, które LLM widzi na próbkę treningową / wnioskowania, zazwyczaj 3000-4000 |
| Curriculum | "Kolejność treningu" | Uporządkowanie etapów (pojedynczy obraz → wiele obrazów → wideo) wybrane dla emergentnego transferu |
| Bilinear pooling | "Zmniejszanie tokenów" | Stosowanie dwuliniowej interpolacji do siatki paczek (2D) w celu zmniejszenia liczby tokenów przy zachowaniu lokalności |
| Emergent skill | "Nietrenowane, nadal działa" | Zdolność pojawiająca się podczas wnioskowania bez pasujących danych treningowych, dzięki kompozycji programu nauczania |
| AnyRes-k | "Konfiguracja k-kafelków" | k pod-kafelków o stałej rozdzielczości plus jedna miniatura, typowe k ∈ {4, 9} |
| Task transfer | "Generalizacja między-scenariuszowa" | Umiejętności nauczone na pojedynczym obrazie, które stosują się do wideo (i odwrotnie) poprzez wspólny szkielet |

## Further Reading

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)