# SRE dla AI — Reagowanie na Incydenty Wieloagentowe, Runbooki, Wykrywanie Predykcyjne

> SRE AI używa LLM opartych na danych infrastruktury (logi, runbooki, topologia usług) poprzez RAG do automatyzacji faz dochodzenia, dokumentacji i koordynacji. Wzorzec architektoniczny 2026 to orkiestracja wieloagentowa — wyspecjalizowani agenci (logi, metryki, runbooki) koordynowani przez nadzorcę; AI proponuje hipotezy i zapytania, ludzie zatwierdzają decyzje wymagające osądu. Datadog Bits AI i Azure SRE Agent dostarczają to jako zarządzane produkty. Runbooki ewoluują: NeuBird Hawkeye używa ewaluacji adwersarialnej (dwa modele analizują ten sam incydent; zgodność = pewność, niezgodność = niepewność); pamięć operacyjna utrzymuje się między zmianami zespołu. Auto-remediacja pozostaje ostrożna: AI sugeruje, ludzie zatwierdzają. W pełni autonomiczne działanie jest wąskie (restart poda, wycofanie konkretnego wdrożenia) z ścisłymi zabezpieczeniami — każdy sprzedający "ustaw i zapomnij" przesadza. Pojawiająca się granica: przewidywanie przed incydentem. Badania MIT raportują, że LLM wytrenowany na historycznych logach + temperaturach GPU + wzorcach błędów API przewidział 89% awarii 10-15 min wcześniej. Prognoza: 95% enterprise LLM będzie miało automatyczny failover do końca 2026.

**Type:** Learn
**Languages:** Python (stdlib, toy multi-agent incident triage simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60 minutes

## Learning Objectives

- Narysować diagram wieloagentowej architektury SRE AI: nadzorca + wyspecjalizowani agenci (logi, metryki, runbooki) + bramka zatwierdzenia ludzkiego.
- Wyjaśnić, dlaczego auto-remediacja jest wąska (restart poda, wycofanie wdrożenia), a nie szeroka (przebudowa architektury usługi).
- Wymienić wzorzec ewaluacji adwersarialnej (NeuBird Hawkeye): dwa modele zgadzają się = pewność; nie zgadzają się = eskalacja.
- Przytoczyć wynik MIT 89% wczesnego wykrywania i ograniczenie operacyjne: przewidywania bez działań to tylko dashboardy.

## The Problem

Inżynier dyżurny dostaje zgłoszenie o 3 nad ranem. "Wysoki wskaźnik błędów w kasie." Sprawdza Datadog, Loki, trzy runbooki, log wdrożeń. 30 minut później zdaje sobie sprawę, że przyczyną źródłową jest OOM vLLM od skoku KV cache. Restartuje poda; błąd znika.

W 2026 pierwsze 20 minut tego dochodzenia jest automatyzowalne. Grupowanie logów według usługi, korelowanie z ostatnimi wdrożeniami, dopasowywanie do runbooków — wszystko to RAG + użycie narzędzi. Nadzorowany agent może wykonać wstępny triaż i przedstawić hipotezę, zanim człowiek otworzy Datadog.

W pełni autonomiczna remediacja to inny problem. Restart poda: bezpieczne. Skalowanie puli GPU: bezpieczne, jeśli polityka pozwala. Przebudowa architektury usługi: absolutnie nie. Dyscypliną jest wytyczenie wąskiej linii.

## The Concept

### Architektura wieloagentowa

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Nadzorca dzieli incydent na podzapytania. Wyspecjalizowani agenci mają dostęp do narzędzi (wyszukiwanie logów, PromQL, wyszukiwanie dokumentów). Nadzorca syntetyzuje, przedstawia hipotezę + dowody człowiekowi. Człowiek zatwierdza lub przekierowuje.

### Zakres auto-remediacji

**Bezpieczne (wąskie)**: restart poda, wycofanie konkretnego wdrożenia, skalowanie puli w ramach pre-zatwierdzonych granic, włączenie pre-zatwierdzonej flagi funkcji.

**Niebezpieczne (szerokie)**: zmiana topologii usługi, modyfikacja limitów zasobów, wdrożenie nowego kodu, zmiana IAM, modyfikacja baz danych.

Każdy sprzedający "ustaw i zapomnij" przesadza. Bezpieczny zbiór rośnie wraz z dojrzewaniem SRE AI, ale granica jest realna.

### Ewaluacja adwersarialna (NeuBird Hawkeye)

Dwa modele niezależnie analizują ten sam incydent. Jeśli zgadzają się co do przyczyny źródłowej, pewność jest wysoka. Jeśli się nie zgadzają, eskaluj do człowieka z widocznymi obiema hipotezami. Prosty wzorzec, skuteczny filtr przeciwko halucynowanym przyczynom źródłowym.

### Pamięć operacyjna

Rotacja zespołu to ciche zabójstwo tradycyjnego SRE — wiedza plemienna odchodzi. SRE AI przechowuje runbooki + analizy powdrożeniowe w bazie wektorowej; agenci pobierają przy każdym nowym incydencie. Gdy dołączają nowi inżynierowie, AI ma pełną historię.

### Przewidywanie przed incydentem

Badania MIT 2025: LLM wytrenowany na historycznych logach, temperaturach GPU i wzorcach błędów API przewidział 89% awarii 10-15 minut przed ich wystąpieniem na zestawie testowym.

Sprawdzenie rzeczywistości: przewidywania bez działań to dashboardy. Pytanie operacyjne brzmi "gdy przewidujemy, co robimy?" Przedwczesne opróżnienie? Pager? Auto-skala? Odpowiedź jest specyficzna dla polityki.

### Produkty w 2026

- **Datadog Bits AI** — zarządzany copilot SRE w Datadog.
- **Azure SRE Agent** — natywny dla Azure.
- **NeuBird Hawkeye** — ewaluacja adwersarialna + pamięć operacyjna.
- **PagerDuty AIOps** — triaż + deduplikacja.
- **Incident.io Autopilot** — dowódca incydentu + koordynacja.

### Runbooki jako kod

Runbooki ewoluują ze stron Confluence do wersjonowanego markdown ze strukturalnymi sekcjami (objaw, hipoteza, zweryfikuj, działaj). Strukturalne runbooki zasilają lepsze wyszukiwanie RAG. Rozpocznij każde wdrożenie AI-SRE od przekształcenia niestrukturalnych runbooków w strukturalne.

### Liczby, które warto zapamiętać

- Wczesne wykrywanie MIT: 89% awarii, 10-15 min czasu wyprzedzenia.
- Triaż wieloagentowy: nadzorca + (logi, metryki, runbooki) + człowiek.
- Bezpieczny zestaw auto-remediacji: restart poda, wycofanie wdrożenia, skalowanie w granicach.
- Ewaluacja adwersarialna: dwa modele niezależnie; zgodność = pewność.

## Use It

`code/main.py` symuluje triaż wieloagentowy: agent logów znajduje błąd, agent metryk znajduje skok CPU, agent runbooków dopasowuje do znanego problemu. Nadzorca rankinguje hipotezy.

## Ship It

Ta lekcja produkuje `outputs/skill-ai-sre-plan.md`. Na podstawie obecnego dyżuru, wolumenu incydentów i dojrzałości zespołu, projektuje wdrożenie SRE AI.

## Exercises

1. Uruchom `code/main.py`. Co jeśli agent logów i metryk się nie zgadzają? Jak nadzorca rozwiązuje konflikt?
2. Zdefiniuj trzy "bezpieczne" działania auto-remediacji dla swojej usługi. Uzasadnij każde.
3. Napisz strukturalny szablon runbooka: sekcje, wymagane pola, polecenia weryfikacyjne.
4. Wykrywanie predykcyjne uruchamia się z 12-minutowym wyprzedzeniem. Jaka jest Twoja polityka — pager, przedwczesne opróżnienie, czy oba?
5. Uzasadnij, czy 3-osobowy zespół powinien przyjąć SRE AI w 2026 czy czekać. Rozważ dojrzałość, wolumen, ryzyko.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| AI SRE | "agent for on-call" | LLM-backed incident investigation + coordination |
| Supervisor agent | "the orchestrator" | Top-level agent breaking incidents into sub-queries |
| Specialized agent | "domain agent" | Sub-agent with tool access (logs, metrics, runbooks) |
| Auto-remediation | "AI fixes it" | Narrow pre-approved action; NOT broad re-architecture |
| Operational memory | "vector runbooks" | Post-mortems + runbooks in vector DB for RAG |
| Adversarial eval | "two-model check" | Independent analyses; agreement = confidence |
| NeuBird Hawkeye | "the adversarial one" | Product with adversarial-eval + memory pattern |
| Bits AI | "Datadog's SRE agent" | Datadog-managed AI SRE |
| Pre-incident prediction | "early detection" | 10-15 min lead time on outage prediction |

## Further Reading

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)