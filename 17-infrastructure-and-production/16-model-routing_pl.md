# Routing modeli jako prymityw redukcji kosztów

> Dynamiczny broker ocenia każde żądanie (typ zadania, długość tokenów, podobieństwo embeddingu, pewność) i wysyła proste zapytania do taniego modelu, eskalując złożone do modelu granicznego. Nazywane też kaskadowaniem modeli. Studia przypadków produkcyjnych pokazują 20-60% redukcji kosztów przy zachowanej jakości we wdrożeniach w USA/UK/UE; 30% poprawa wydajności routingu w SaaS o dużym wolumenie przekłada się na sześciocyfrowe roczne oszczędności. Kontekst 2026: ceny inferencji LLM spadły ~10x rocznie — token klasy GPT-4 spadł z $20/M do ~$0.40/M od końca 2022 do 2026. Większość spadku to lepsze stosy serwujące (Phase 17 · 04-09), nie sprzęt. Routing to sposób na zamianę tego spadku cen na marżę bez regresji produktu. Tryb awarii to dryf taniego modelu: route kieruje 40% do słabszego modelu, jakość spada o 3-5% w zadaniach wymagających rozumowania, nikt tego nie zauważa przez kwartał. Ograniczaj trasy za pomocą metryk jakości online, nie tylko offline zestawów ewaluacyjnych.

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 minutes

## Learning Objectives

- Wyjaśnić kaskadowanie modeli: najpierw tani z kontrolą pewności, eskalacja przy niskiej pewności.
- Wymienić cztery sygnały routingu (klasyfikacja zadania, długość promptu, podobieństwo embeddingu do znanego trudnego zestawu, pewność własna z pierwszego przejścia).
- Obliczyć oczekiwany mieszany koszt przy docelowym podziale routingu i tolerancji utraty jakości.
- Wymienić metrykę monitorowania dryfu (bramka jakości online), która wychwytuje pełzanie taniego modelu.

## The Problem

Twój serwis kosztuje $80k/miesiąc na GPT-5. Analityka pokazuje, że 70% zapytań jest prostych: "Która godzina w Paryżu?", "Przeformułuj to zdanie." Model klasy Haiku radzi sobie z nimi doskonale za 3% kosztu. 30% potrzebuje rozumowania GPT-5 — kodowania, matematyki, planowania wieloetapowego.

Jeśli skierujesz 70% do taniego modelu, a 30% do drogiego, Twój rachunek spada ~65% przy tej samej jakości produktu. To jest routing. Sztuką jest zbudowanie brokera bez pogorszenia jakości.

## The Concept

### Cztery sygnały routingu

1. **Klasyfikacja zadania**: proste/złożone/kodowanie/matematyka/czat. Może to być klasyfikator oparty na regułach, mały LLM (klasa Haiku za $0.25/M) lub podobieństwo embeddingu do oznakowanych kubełków. Wynik: route = tani / zrównoważony / graniczny.

2. **Długość promptu**: prompty >4K tokenów często potrzebują modelu granicznego dla spójności. Prompty <500 tokenów zwykle nie.

3. **Podobieństwo embeddingu do znanego trudnego zestawu**: jeśli zapytanie jest bliskie (cosine > 0.88) znanemu trudnemu kubełkowi, eskaluj bezpośrednio do modelu granicznego.

4. **Pewność własna z pierwszego przejścia**: wyślij do taniego; jeśli log-proby modelu pokazują niską pewność LUB odmawia LUB generuje język wymijający, powtórz na modelu granicznym. Dodaje P95 latencji na ~10% ruchu, ale oszczędza 50%+ na pozostałych 90%.

### Trzy wzorce

**Pre-route** (klasyfikator z przodu): ~5-10ms dodanej latencji; najszybszy ogólnie.

**Kaskada** (najpierw tani, eskalacja przy niskiej pewności): ~1.2x mediana latencji (tanie uruchomienie plus weryfikacja), ~2x przy eskalacji. Najlepsze minimum jakości.

**Route zespołowy** (uruchom tani i graniczny równolegle dla próbki, wybór przez model nagrody): najwyższa jakość, najwyższy koszt; używaj tylko do krytycznych testów A/B.

### Implementacja

Bramki AI (Phase 17 · 19) udostępniają routing. LiteLLM ma konfigurację `router` z fallbackiem i routingiem kosztowym. Portkey ma zabezpieczenia + routing. Kong AI Gateway ma routing oparty na wtyczkach. Rynek modeli OpenRouter udostępnia API rekomendacji.

Open-source: RouteLLM (LMSYS), Not Diamond (komercyjny), Prompt Mule.

### Krzywa cenowa 2026

| Klasa modelu | Koniec 2022 | 2026 | Zmiana |
|-------------|-----------|------|--------|
| Jakość GPT-4 | ~$20/M | ~$0.40/M | 50x taniej |
| Graniczny (GPT-5, Claude 4) | — | ~$3-10/M | nowy poziom |

Większość poprawy to wydajność serwowania — podstawowe lekcje z Phase 17 · 04-09 przełożyły się na spadki kosztów po stronie dostawców. Routing pozwala przechwycić te zyski na poziomie aplikacji, zamiast czekać, aż wszyscy użytkownicy migrują do taniego poziomu.

### Dryf to realne ryzyko

Twój route kieruje 40% do taniego modelu. Przez sześć miesięcy dystrybucja zadań się zmienia (użytkownicy stają się bardziej zaawansowani, zadają dłuższe pytania). Router tego nie zauważa, ponieważ jego klasyfikator został wytrenowany na danych z Q1. Jakość spada po cichu. Nikt nie narzeka wystarczająco głośno. Dowiadujesz się o tym, gdy przegrywasz w benchmarku konkurencji.

Ograniczaj trasy za pomocą metryk jakości online:

- Kciuk w górę / kciuk w dół użytkownika na trasę.
- Automatyczny sędzia LLM na wstrzymanej próbce (5%) na trasę.
- Wskaźnik eskalacji: jeśli kaskada przekierowuje wyżej >30%, tani model jest nadużywany.
- Wskaźnik odmowy na trasę.

### Liczby, które warto zapamiętać

- Oszczędności routingu 2026 przy zachowanej jakości: 20-60% w studiach przypadków.
- Spadek cen LLM 2022-2026: ~10x rocznie.
- GPT-4 2022 vs 2026: ~$20/M → ~$0.40/M.
- Wpływ kaskady na latencję: ~1.2x mediana, ~2x eskalacja (~10% ruchu).

## Use It

`code/main.py` symuluje pre-route, kaskadę i route zespołowy na mieszanym obciążeniu. Raportuje mieszany koszt, utratę jakości i wskaźnik eskalacji.

## Ship It

Ta lekcja produkuje `outputs/skill-router-plan.md`. Na podstawie obciążenia i budżetu jakości wybiera wzorzec routingu i sygnały.

## Exercises

1. Uruchom `code/main.py`. Przy jakim progu dokładności kaskada pokonuje pre-route?
2. Twoja baza użytkowników to 30% enterprise (złożone zapytania), 70% darmowy poziom (proste). Zaprojektuj podział routingu. Jaka metryka online go bramkuje?
3. Route obniża jakość o 2%, ale oszczędza 40%. Czy to do wdrożenia? Zależy od produktu — uzasadnij obie strony.
4. Zaimplementuj kontrolę pewności używając logprobów z API OpenAI/Anthropic. Jaki próg ustawiasz na początek?
5. Przez sześć miesięcy wskaźnik eskalacji rośnie z 8% do 22%. Zdiagnozuj trzy przyczyny i poprawkę dla każdej.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## Further Reading

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives.