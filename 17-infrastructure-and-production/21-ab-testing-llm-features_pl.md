# Testy A/B Funkcji LLM — GrowthBook, Statsig i Problem "Wibracji"

> Tradycyjne testy A/B nie były zaprojektowane dla niedeterministycznych LLM. Kluczowe rozróżnienie: ewaluacje odpowiadają "czy model może wykonać zadanie?" Testy A/B odpowiadają "czy użytkownicy się tym przejmują?" Oba są wymagane; wdrażanie na podstawie wibracji się skończyło. Co testować w 2026: inżynieria promptów (sformułowanie), wybór modelu (GPT-4 vs GPT-3.5 vs OSS; dokładność vs koszt vs latencja), parametry generacji (temperatura, top-p). Rzeczywiste przypadki: wariant modelu nagrody czatbota dostarczył +70% długości rozmowy i +30% retencji; eksperymenty z tematami wiadomości Nextdoor dostarczyły +1% CTR po dopracowaniu funkcji nagrody; Khan Academy Khanmigo iterowało na osi latencja-vs-dokładnośc-matematyczna. Podział platform: **Statsig** (przejęty przez OpenAI za $1.1B we wrześniu 2025) — testowanie sekwencyjne, CUPED, wszystko w jednym. **GrowthBook** — open-source, natywny dla magazynów danych, silniki Bayesowskie + Frequentystowskie + Sekwencyjne, CUPED, kontrole SRM, korekty Benjamini-Hochberg i Bonferroni. Wybierasz na podstawie preferencji SQL w magazynie danych i tego, czy "przejęte przez OpenAI" ma znaczenie dla Twojej organizacji.

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 minutes

## Learning Objectives

- Odróżnić ewaluacje ("czy model może wykonać zadanie") od testów A/B ("czy użytkownicy się tym przejmują").
- Wymienić trzy testowalne osie (prompt, model, parametry) i wybrać metrykę dla każdej.
- Wyjaśnić CUPED, testowanie sekwencyjne i korekty porównań wielokrotnych Benjamini-Hochberg.
- Wybrać Statsig lub GrowthBook na podstawie postawy SQL w magazynie danych i stanowiska wobec przejęcia korporacyjnego.

## The Problem

Ręcznie dostroiłeś system prompt. Wydaje się lepszy. Wdrażasz go. Konwersja zmienia się o szum. Obwiniasz metrykę. Albo wdrożyłeś nowy model i konwersja się nie ruszyła — czy model się pogorszył, czy zmiana była zbyt mała do wykrycia? Nie wiesz, ponieważ wdrożyłeś bez testu A/B.

Ewaluacje odpowiadają, czy model może wykonać zadanie na oznakowanym zestawie. Nie odpowiadają, czy użytkownicy preferują wynik. Tylko kontrolowany eksperyment online odpowiada na to pytanie, i tylko jeśli eksperyment ma wystarczającą moc, kontroluje niedeterminizm i koryguje porównania wielokrotne.

## The Concept

### Ewaluacje vs testy A/B

**Ewaluacje** — offline, oznakowany zestaw, sędzia (rubryka lub LLM-jako-sędzia lub człowiek). Odpowiedź: "Czy wynik jest poprawny / pomocny / bezpieczny na tej ustalonej dystrybucji?"

**Test A/B** — online, żywi użytkownicy, randomizowani. Odpowiedź: "Czy nowy wariant przesuwa metrykę na poziomie użytkownika, która ma znaczenie?"

Oba wymagane. Ewaluacje wychwytują regresje przed ekspozycją; A/B potwierdza wpływ na produkt po.

### Co testować

1. **Inżynieria promptów** — sformułowanie, struktura system promptu, przykłady. Metryka: sukces zadania, retencja użytkowników, koszt/żądanie.
2. **Wybór modelu** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS. Metryka: dokładność (zadania) + koszt/żądanie + latencja P99. Wielocelowa.
3. **Parametry generacji** — temperatura, top-p, max_tokens. Metryka: specyficzna dla zadania (różnorodność wyjścia vs determinizm).

### CUPED — redukcja wariancji

Controlled-experiments Using Pre-Experiment Data. Regresja wariancji przedokresowej przed porównaniem okresu po. Typowa redukcja wariancji: 30-70%. Efektywna wielkość próbki wzrasta za darmo.

Implementacja: zarówno Statsig, jak i GrowthBook implementują.

### Testowanie sekwencyjne

Klasyczne A/B zakłada ustaloną wielkość próbki. Testy sekwencyjne ("zajrzyj-i-zdecyduj") kontrolują wskaźnik fałszywie pozytywnych przy wielokrotnych zaglądaniach. Zawsze-ważne procedury sekwencyjne (mSPRT, sekwencje ufności Howarda) pozwalają zatrzymać się wcześnie na wyraźnych zwycięzcach.

### Korekty porównań wielokrotnych

Uruchomienie 20 testów A/B przy 95% ufności produkuje jeden fałszywie pozytywny przypadkowo. Korekta Bonferroni zaostrza α na test; Benjamini-Hochberg kontroluje wskaźnik fałszywych odkryć. GrowthBook implementuje obie.

### SRM — niedopasowanie proporcji próbki

Hash przypisania randomizuje użytkowników do wariantów. Jeśli podział 50/50 daje 47/53, coś jest zepsute — kontrola SRM to flaguje. Obie platformy implementują.

### Statsig vs GrowthBook

**Statsig**:
- Przejęty przez OpenAI za $1.1B (wrzesień 2025). Hostowany, SaaS.
- Testowanie sekwencyjne, CUPED, populacje wstrzymane.
- Wszystko w jednym: flagi funkcji + eksperymentacja + obserwowalność.
- Najlepsze dopasowanie: zespół chce już pakietowego produktu, nie przejmuje się własnością OpenAI.

**GrowthBook**:
- Open-source (MIT); natywny dla magazynów danych (czyta bezpośrednio z Snowflake/BigQuery/Redshift).
- Wiele silników: Bayesowski, Frequentystowski, Sekwencyjny.
- CUPED, SRM, korekty Bonferroni, BH.
- Samodzielne hostowanie lub zarządzana chmura.
- Najlepsze dopasowanie: sklep z SQL w magazynie danych, zespół danych kontroluje warstwę metryk, chce OSS.

### Niedeterminizm komplikuje moc

Ten sam prompt produkuje różne wyjścia. Tradycyjne obliczenia mocy zakładają IID obserwacje. Przy niedeterminizmie LLM efektywna wielkość próbki jest niższa niż nominalna. Pomnóż wymaganą wielkość próbki przez ~1.3-1.5x jako margines bezpieczeństwa.

### Rzeczywiste wyniki przypadków

- Wariant modelu nagrody czatbota: +70% długości rozmowy, +30% retencji.
- Tematy wiadomości Nextdoor: +1% CTR po dopracowaniu funkcji nagrody.
- Khan Academy Khanmigo: iteracyjny kompromis latencja-vs-dokładnośc-matematyczna.

### Wzorzec antywzorcowy: wdrażanie na wibracjach

Każdy starszy inżynier może wymienić funkcję, która została wdrożona, bo "wydaje się lepsza" bez A/B. Większość z nich pogorszyła metryki produktu, których zespół nie zauważył przez miesiące. A/B jest funkcją wymuszającą.

### Liczby, które warto zapamiętać

- Statsig przejęty przez OpenAI: $1.1B, wrzesień 2025.
- GrowthBook: open-source MIT; Bayesowski + Frequentystowski + Sekwencyjny.
- Redukcja wariancji CUPED: 30-70%.
- Niedeterminizm LLM → +30-50% bufora wielkości próbki.

## Use It

`code/main.py` symuluje sekwencyjny test A/B z ustalonymi i sekwencyjnymi granicami. Pokazuje, jak sekwencyjny pozwala zatrzymać się wcześnie.

## Ship It

Ta lekcja produkuje `outputs/skill-ab-plan.md`. Na podstawie zmiany funkcji, obciążenia i bazowego, wybiera platformę, bramki i wielkość próbki.

## Exercises

1. Uruchom `code/main.py`. Dla oczekiwanego wzrostu 5% z bazową konwersją 3%, jaka wielkość próbki do 80% mocy?
2. Wybierz Statsig lub GrowthBook dla klienta on-prem z regulowanej opieki zdrowotnej.
3. Zaprojektuj test A/B porównujący GPT-4 vs GPT-3.5 na koszcie na rozwiązane zgłoszenie. Jaka jest podstawowa metryka, metryka zabezpieczająca, drugorzędna?
4. Twoje kanarkowe przechodzi, ale A/B pokazuje -1.2% konwersji. Czy wdrażasz? Napisz kryteria eskalacji.
5. Zastosuj CUPED do okresu przed z 60% wariancji okresu po. Oblicz wzrost efektywnej wielkości próbki.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Eval | "offline test" | Labeled-set evaluation of model capability |
| A/B test | "experiment" | Live randomized comparison on users |
| CUPED | "variance reduction" | Pre-period regression to reduce variance |
| Sequential test | "peek-ok test" | Always-valid procedure allowing early stop |
| Multiple comparison | "the family error" | Running many tests inflates false positives |
| Bonferroni | "tight correction" | Divide α by number of tests |
| Benjamini-Hochberg | "BH FDR" | False-discovery-rate control, less conservative |
| SRM | "bad split" | Sample ratio mismatch; assignment bug |
| Statsig | "OpenAI owned" | Commercial all-in-one, acquired 2025 |
| GrowthBook | "the OSS one" | MIT warehouse-native platform |
| mSPRT | "sequential probability ratio test" | Classical sequential procedure |

## Further Reading

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)