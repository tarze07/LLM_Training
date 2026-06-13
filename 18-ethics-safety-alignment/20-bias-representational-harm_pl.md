# Uprzedzenia i Szkoda Reprezentacyjna w LLM

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Computational Linguistics 2024, arXiv:2309.00770). Fundamentalny przegląd z 2024 roku rozróżniający szkody reprezentacyjne (stereotypy, wymazywanie) od szkód alokacyjnych (nierówna dystrybucja zasobów) i kategoryzujący metryki ewaluacji jako oparte na osadzeniach, prawdopodobieństwie lub wygenerowanym tekście. Empiria 2024-2025: An et al. (PNAS Nexus, marzec 2025) mierzą międzynarodowe uprzedzenie płeć x rasa w GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet, Llama 3-70B w automatycznej ocenie CV dla 20 stanowisk podstawowych. WinoIdentity (COLM 2025, arXiv:2508.07111) wprowadza ewaluację uczciwości opartą na niepewności dla tożsamości międzynarodowych. Yu i Ananiadou 2025 identyfikują neurony płci w warstwach MLP; Ahsan i Wallace 2025 używają SAE do ujawnienia klinicznego uprzedzenia rasowego; Zhou et al. 2024 (UniBias) manipuluje głowami uwagi dla odchylania. Meta-krytyka (arXiv:2508.11067): 10-letnia literatura w nieproporcjonalnym stopniu skupia się na uprzedzeniu binarnej płci.

**Type:** Build
**Languages:** Python (stdlib, zabawkowa sonda uprzedzenia oparta na osadzeniach)
**Prerequisites:** Phase 05 (word embeddings), Phase 18 · 01 (instruction following)
**Time:** ~60 minutes

## Learning Objectives

- Zdefiniować szkodę reprezentacyjną vs alokacyjną i podać jeden przykład każdej we wdrożeniu LLM.
- Wymienić trzy kategorie metryk ewaluacji z Gallegos et al. 2024 i opisać jedną metrykę z każdej.
- Opisać międzynarodowość i dlaczego pomiar uczciwości oparty na niepewności WinoIdentity wypełnia luki w ewaluacji uprzedzenia jednoosiowego.
- Opisać dwa podejścia mechanistycznej interpretowalności do uprzedzenia (neurony płci, cechy SAE, manipulacja głowami uwagi).

## The Problem

Poprzednie lekcje obejmują zamierzoną szkodę (jailbreak, knowanie) i zarządzanie bezpieczeństwem. Uprzedzenie to szkoda, która pojawia się bez zamiaru — z rozkładów danych treningowych, z ramowania promptu, z nagromadzonych wyborów projektowych. Mierzenie i redukowanie go to odrębne wyzwanie metodologiczne od odporności adwersarialnej.

## The Concept

### Reprezentacyjna vs alokacyjna

- **Szkoda reprezentacyjna.** Stereotypy, wymazywanie, poniżające portrety. LLM, który przedstawia pielęgniarki jako wyłącznie kobiety, powoduje szkodę reprezentacyjną.
- **Szkoda alokacyjna.** Nierówne wyniki materialne. LLM, który systematycznie niżej ocenia CV czarnoskórych kandydatów, powoduje szkodę alokacyjną.

To nie to samo. Model może być "reprezentacyjnie nieuprzedzony" (produkuje zróżnicowane portrety) będąc "alokacyjnie uprzedzonym" (dokonuje nierównych rekomendacji). Ewaluacje muszą mierzyć oba.

### Trzy kategorie metryk ewaluacji (Gallegos et al. 2024)

- **Oparte na osadzeniach.** Testy w stylu WEAT na osadzeniach przed RLHF. Mierzą statystyczne skojarzenia między terminami tożsamości a terminami atrybutów. Ograniczone: mierzą reprezentację, nie zachowanie.
- **Oparte na prawdopodobieństwie.** Log-prawdopodobieństwo dokończeń potwierdzających stereotyp vs łamiących stereotyp. Pomiar po stronie dekodera. Oddaje część behawioralnego uprzedzenia.
- **Oparte na wygenerowanym tekście.** Pomiar w zadaniu końcowym na wygenerowanym tekście. Ocena CV, pisanie rekomendacji, dialog. Najbardziej ekologicznie trafne; najtrudniejsze do odtworzenia.

### Międzynarodowość

Ewaluacja uprzedzenia na "płci" pomija uprzedzenie, które uaktywnia się tylko na parach (płeć, rasa). An et al. 2025 znajdują, że GPT-4o karze czarnoskóre kobiety w ocenie CV bardziej niż czarnoskórych mężczyzn i bardziej niż białe kobiety osobno. Ewaluacja jednoosiowa nie może tego uchwycić.

WinoIdentity (COLM 2025) wprowadza międzynarodową uczciwość opartą na niepewności. Mierzy, czy niepewność modelu co do wyników różni się między międzynarodowymi krotkami tożsamości — nie tylko punktową predykcję. To łapie przypadki, w których model jest równie błędny we wszystkich grupach, ale bardziej niepewny dla niektórych, co produkuje różne zachowania alokacyjne downstream.

### Podejścia mechanistyczne

Prace nad interpretowalnością 2024-2025 otwierają uprzedzenie na interwencję mechaniczną:

- **Neurony płci (Yu i Ananiadou 2025).** Konkretne neurony MLP korelują z zachowaniami specyficznymi dla płci. Ablacja tych neuronów zmniejsza metryki luki płciowej przy ograniczonym koszcie zdolności.
- **Kliniczne uprzedzenie rasowe przez SAE (Ahsan i Wallace 2025).** Cechy autoenkodera rzadkiego rozkładają wewnętrzną reprezentację na interpretowalne wymiary; cechy skorelowane z rasą mogą być zidentyfikowane i stłumione.
- **UniBias (Zhou et al. 2024).** Manipulacja głowami uwagi dla odchylania zerowym strzałem. Konkretne głowy wzmacniają wrażliwość na klasę tożsamości; zerowanie lub przeważanie tych głów redukuje uprzedzenie bez dostrajania.

### Meta-krytyka

10-letni przegląd literatury (arXiv:2508.11067, 2025) stwierdza, że pole w nieproporcjonalnym stopniu skupia się na uprzedzeniu binarnej płci. Inne osie — niepełnosprawność, religia, status migracyjny, tożsamość wielojęzyczna — otrzymują znacznie mniej uwagi. Meta-krytyka argumentuje, że wąskie skupienie może szkodzić zmarginalizowanym grupom przez zaniedbanie: model dobrze odchylony na binarnej płci może być mocno uprzedzony na wymiarach, których nikt nie sprawdził.

### Miejsce w Fazie 18

Lekcje 20-21 obejmują formalnie uprzedzenie i uczciwość. Lekcja 22 obejmuje prywatność. Lekcja 23 obejmuje znakowanie wodne. To warstwa szkody dla użytkownika uzupełniająca wcześniejszą warstwę oszustwa/bezpieczeństwa.

## Use It

`code/main.py` buduje zabawkową sondę uprzedzenia opartą na osadzeniach: mierzy odległość w stylu WEAT między terminami tożsamości a terminami atrybutów w prostej reprezentacji współwystępowania. Możesz wstrzyknąć uprzedzenie i zaobserwować wyzwolenie metryki; zastosować prostą operację odchylania i zaobserwować częściowe przywrócenie.

## Ship It

Ta lekcja produkuje `outputs/skill-bias-eval.md`. Dla karty modelu lub twierdzenia o uczciwości, audytuje ewaluację w trzech kategoriach metryk (osadzenia, prawdopodobieństwo, wygenerowany tekst), pokrycie międzynarodowości i mechanizm każdej interwencji odchylania.

## Exercises

1. Uruchom `code/main.py`. Raportuj wyniki uprzedzenia w stylu WEAT przed i po kroku odchylania. Wyjaśnij, dlaczego metryka nie spada do zera.

2. Rozszerz sondę o test międzynarodowy: (płeć, rasa) x (kariera, rodzina). Raportuj wyniki uprzedzenia międzyosiowego.

3. Przeczytaj An et al. 2025 (PNAS Nexus). Zidentyfikuj dwa efekty międzynarodowe, które raportują, a które ewaluacja jednoosiowa płci by pominęła.

4. Yu i Ananiadou 2025 identyfikują neurony płci. Naszkicuj eksperyment falsyfikacyjny, który odróżniłby "te neurony powodują uprzedzenie płciowe" od "te neurony korelują z uprzedzeniem płciowym."

5. Meta-krytyka argumentuje, że pole skupia się zbyt wąsko na binarnej płci. Wybierz jedną niedostatecznie zbadaną oś i opisz protokół pomiaru szkody reprezentacyjnej dla niej.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Szkoda reprezentacyjna | "stereotypy / wymazywanie" | Stronnicze przedstawienie grupy |
| Szkoda alokacyjna | "nierówne decyzje" | Stronniczy wynik materialny dla grupy |
| WEAT | "test osadzeń" | Word Embedding Association Test; sonda uprzedzenia oparta na współwystępowaniu |
| Międzynarodowość | "połączone efekty tożsamości" | Uprzedzenie pojawiające się na przecięciu wielu osi tożsamości |
| Neurony płci | "neurony uprzedzenia MLP" | Konkretne neurony, których aktywacje korelują z zachowaniem specyficznym dla płci |
| Cecha SAE | "interpretowalny wymiar" | Cecha zidentyfikowana przez rzadki autoenkoder; użyteczna do mechanistycznej analizy uprzedzenia |
| UniBias | "odchylanie głów uwagi" | Odchylanie zerowym strzałem przez przeważanie głów uwagi |

## Further Reading

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — kanoniczny przegląd
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — pięciomodelowe badanie międzynarodowe
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — nowy benchmark
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) — odchylanie zerowym strzałem
