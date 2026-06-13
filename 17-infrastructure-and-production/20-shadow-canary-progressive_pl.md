# Ruch Cieniowania, Wdrożenie Kanarkowe i Wdrożenie Progresywne dla LLM

> Wdrożenia LLM łączą najtrudniejsze części wdrażania oprogramowania: brak testów jednostkowych, rozproszone tryby awarii, opóźnione sygnały. Sekwencja to (1) tryb cienia — duplikuj żądania produkcyjne do modelu kandydata, loguj, porównuj z zerowym wpływem na użytkownika; wychwytuje oczywiste problemy z dystrybucją, ale nie jest gwarancją jakości; (2) wdrożenie kanarkowe — progresywne przesunięcie ruchu 10% → 25% → 50% → 75% → 100% z bramkami na każdym kroku; śledź percentyle latencji, koszt/żądanie, wskaźnik błędów/odmów, dystrybucję długości wyjścia, wskaźnik opinii użytkowników; (3) testy A/B dla wyraźnych alternatyw po potwierdzeniu stabilności. Niedeterminizm jest nieusuwalny — do 15% zmienności dokładności między uruchomieniami z identycznymi wejściami z powodu niełączności FP GPU plus zmienności rozmiaru partii. Koszt jest zmienną, nie stałą — model o 20% lepszy może być 3x droższy na wywołanie. Szybkość wycofania jest decydująca: jeśli wycofanie wymaga ponownego wdrożenia, jesteś za wolny. Polityka żyje w config/flags; model żyje w rejestrze z przypiętymi skrótami; wycofanie = flipnięcie polityki + przywrócenie progu + przypięcie starego modelu w sekundach.

**Type:** Learn
**Languages:** Python (stdlib, toy canary-progression simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**Time:** ~60 minutes

## Learning Objectives

- Odróżnić tryb cienia (porównanie bez wpływu), kanarkowe (progresywny ruch na żywo) i A/B (porównanie po potwierdzeniu stabilności).
- Wymienić pięć metryk kanarkowych specyficznych dla LLM (latencja, koszt/żądanie, błędy/odmowy, dystrybucja długości wyjścia, opinie użytkowników).
- Wyjaśnić, dlaczego niedeterminizm LLM (do 15%) zmienia znaczenie "stabilności" we wdrożeniu.
- Zaprojektować ścieżkę wycofania trwającą sekundy (flipnięcie polityki), a nie godziny (ponowne wdrożenie).

## The Problem

Wdrażasz nowy model. Ewaluacje offline pokazują 3% wzrost dokładności. Włączasz go w produkcji. W ciągu 24 godzin koszt wzrasta o 40%, kciuki w dół użytkowników wzrastają o 8%, trzy zgłoszenia klientów raportują "dziwne odpowiedzi". Wycofujesz się. Ponowne wdrożenie zajmuje 3 godziny. Twój weekend jest zrujnowany.

Każda część tego była do uniknięcia. Tryb cienia wychwyciłby 40% skok kosztów, zanim zobaczył go jakikolwiek użytkownik. Kanarkowe zatrzymałoby się na 10%, gdy kciuki w dół się poruszyły. Wycofanie przez flagę polityki zajęłoby 30 sekund. Dyscyplina wypełnia lukę między "ewaluacje offline wyglądają dobrze" a "rzeczywiści użytkownicy są zadowoleni."

## The Concept

### Tryb cienia

Kandydat otrzymuje te same żądania co produkcja; wyniki są logowane, nie zwracane użytkownikom. Zerowy wpływ na użytkownika. Loguj:

- Zawartość wyjścia (różnica względem produkcji).
- Liczby tokenów (delta kosztu).
- Latencję.
- Odmowy i błędy.

Wychwytuje: eksplozje kosztów, regresje długości, oczywiste zmiany w odmowach, twarde błędy. NIE wychwytuje: delty jakości, którą użytkownicy by zauważyli. Cień to test dymny, nie test jakości.

### Wdrożenie kanarkowe

Progresywne przesunięcie ruchu z bramkami. Typowa progresja: 1% → 10% → 25% → 50% → 75% → 100%. Bramkuj na 5 metrykach na każdym kroku:

1. **Percentyle latencji** — P50, P95, P99. Naruszenie: kanarkowe ma P99 > 1.5x bazowej.
2. **Koszt na żądanie** — mieszany $. Naruszenie: >20% powyżej bazowej.
3. **Wskaźnik błędów / odmów** — 5xx plus jawne odmowy. Naruszenie: 2x bazowa.
4. **Dystrybucja długości wyjścia** — średnia + P99. Naruszenie: przesunięcie dystrybucji.
5. **Wskaźnik opinii użytkowników** — kciuki w dół / zgłoszenia. Naruszenie: 1.5x bazowa.

### Niedeterminizm to nowa wariancja

Identyczne wejścia produkują nieidentyczne wyjścia. Powody:

- Niełączność FP GPU (kolejność redukcji zmiennoprzecinkowej różni się w zależności od partii).
- Zmienność rozmiaru partii (ten sam prompt w partii 128 vs partii 16).
- Próbkowanie (temperatura > 0).

Zmierzono: do 15% zmienności dokładności między uruchomieniami na identycznych zestawach ewaluacyjnych. "Stabilność" we wdrożeniu oznacza, że metryki mieszczą się w oczekiwanej wariancji, nie są identyczne z bazową. Ustaw bramki powyżej poziomu szumu.

### Koszt jest zmienną

Model o 20% lepszy może być 3x droższy na wywołanie. Koszt/żądanie jest jedną z pięciu bramek. Wdrożenie "lepszego" modelu, który psuje jednostkową ekonomikę, jest przypadkiem wycofania.

### Wycofanie jest bronią

- Flaga polityki (system flag funkcji): flipnij procent w konfiguracji; zajmuje sekundy.
- Przypięcie modelu (skrót rejestru): przypięty model nie aktualizuje się automatycznie.
- Wycofanie = przywróć flagę + ustaw przypięty skrót na poprzedni. Sekundy, nie godziny.

Jeśli Twój stos wymaga ponownego wdrożenia do wycofania, napraw to przed wdrożeniem.

### Narzędzia

**Argo Rollouts** / **Flagger** — kontrolery wdrożeń progresywnych Kubernetes. Integracja z ważonym routingiem Istio/Linkerd.

**Ważony routing Istio** — podział ruchu na poziomie siatki usług.

**KServe / Seldon Core** — serwowanie modeli z wbudowanym kanarkowym.

**Flagi funkcji** — LaunchDarkly, Flagsmith, Unleash. Flipnięcie na poziomie polityki, bez ponownego wdrożenia.

### Kadencja metryk

Bramki kanarkowe sprawdzają co 5-15 minut w zależności od wolumenu ruchu. 1% ruchu z 10 req/min daje 50-150 punktów danych na okno — wystarczająco dla latencji, ale zaszumione dla opinii użytkowników. 10% daje ~10x więcej. Progresje powinny pauzować wystarczająco długo, aby zgromadzić wystarczającą liczbę próbek na każdym kroku.

### Krok A/B jest opcjonalny

Jeśli nowy model jest wyraźnie różny (inne zachowanie, inna krzywa kosztów, inny ton), przetestuj go A/B przy 50% po przejściu kanarkowego. Jeśli to tylko ulepszona wersja, przejdź do 100%, gdy bramki kanarkowe przejdą.

### Liczby, które warto zapamiętać

- Progresja kanarkowa: 1% → 10% → 25% → 50% → 75% → 100%.
- Sufit niedeterminizmu: do 15% wariancji między uruchomieniami na identycznych wejściach.
- Pięć metryk kanarkowych: latencja, koszt, błędy/odmowy, długość wyjścia, opinie użytkowników.
- Bramka kosztowa: >20% powyżej bazowej to naruszenie.
- Wycofanie: sekundy, nie godziny.

## Use It

`code/main.py` symuluje wdrożenie kanarkowe z wstrzykniętymi regresjami. Raportuje, na którym etapie wdrożenie się zatrzymuje i która bramka została uruchomiona.

## Ship It

Ta lekcja produkuje `outputs/skill-rollout-runbook.md`. Na podstawie modelu kandydata, bazowego i tolerancji ryzyka, projektuje plan cień→kanarkowe→100%.

## Exercises

1. Uruchom `code/main.py`. Wstrzyknij 25% regresję kosztów. Na którym etapie zatrzymuje się kanarkowe?
2. Twój nowy model ma 3% wzrost dokładności offline, ale koszt/żądanie jest +18%. Czy to do wdrożenia? Zależy od polityki — napisz obie ścieżki.
3. Zaprojektuj wycofanie trwające poniżej 60 sekund od końca do końca. Wymień wymaganą infrastrukturę.
4. Niedeterminizm pokazuje ±7% na Twojej ewaluacji. Ustaw bramki kanarkowe tak, aby nie generować fałszywych alarmów. Jakich mnożników używasz?
5. Tryb cienia wychwytuje 40% skok kosztów przed kanarkowym. Napisz regułę alarmu, która uruchamia się w cieniu.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Shadow mode | "duplicate to new" | Zero-impact send-to-candidate for logging |
| Canary | "progressive traffic" | Gradual user-exposed rollout with gates |
| Gates | "rollout checks" | Metric thresholds that block progression |
| Non-determinism | "LLM variance" | Irreducible run-to-run differences |
| Policy flag | "flag flip rollback" | Config-level rollback, seconds not hours |
| Model pin | "registry digest" | Immutable reference to a model version |
| Argo Rollouts | "K8s progressive" | Kubernetes-native canary/rollback controller |
| KServe | "inference K8s" | Model serving with canary primitives |
| Istio weighted | "mesh split" | Service-mesh traffic splitter |

## Further Reading

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)