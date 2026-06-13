# Inżynieria Chaosu dla Produkcji LLM

> Inżynieria chaosu dla LLM to własna dyscyplina w 2026. Wymagania wstępne przed uruchomieniem eksperymentów w produkcji: zdefiniowane SLI/SLO, obserwowalność (ślady + metryki + logi), automatyczne wycofanie, runbooki, dyżur. Architektura ma cztery płaszczyzny: sterowania (harmonogram eksperymentów), celu (usługi, infrastruktura, magazyny danych), bezpieczeństwa (zabezpieczenia + przerwanie + filtry ruchu), obserwowalności (metryki + ślady + logi), sprzężenia zwrotnego (do dostosowań SLO). Zabezpieczenia są obowiązkowe: alarmy tempa wypalania wstrzymują eksperymenty, jeśli dzienne zużycie budżetu błędów > 2x oczekiwanego; okna wyciszania + korelacja ID śladu deduplikują szum alarmów. Kadencja: cotygodniowe małe kanarkowe + przegląd SLO; comiesięczne game day + analiza powdrożeniowa; kwartalny audyt odporności międzyzespołowy + mapowanie zależności. Eksperymenty specyficzne dla LLM: przeciążenie pamięci, awarie sieci, awarie dostawców, zniekształcone prompty, burze eksmisji KV cache. Narzędzia: Harness Chaos Engineering (rekomendacje pochodzące z LLM, skalowanie w dół promienia rażenia, integracja z narzędziami MCP); LitmusChaos (CNCF); Chaos Mesh (CNCF natywny Kubernetes).

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- Wymienić pięć wymagań wstępnych inżynierii chaosu (SLI/SLO, obserwowalność, wycofanie, runbooki, dyżur) i wyjaśnić, dlaczego pominięcie któregokolwiek psuje praktykę.
- Narysować diagram czterech płaszczyzn (sterowania, celu, bezpieczeństwa, obserwowalności) i pętli sprzężenia zwrotnego do SLO.
- Wymienić pięć eksperymentów specyficznych dla LLM (przeciążenie pamięci, awaria sieci, awaria dostawcy, zniekształcony prompt, burza eksmisji KV).
- Wybrać narzędzie — Harness, LitmusChaos, Chaos Mesh — w zależności od stosu.

## The Problem

Testowanie chaosu w tradycyjnych stosach jest ugruntowane. Stosy LLM dodają nowe tryby awarii. 4K-tokenowy prompt ze znakiem trucizny blokuje tokenizator na 12 sekund. Dostawca wyższy 429; Twoja bramka ponawia; Twoja usługa dostaje OOM od wzmocnionej ponowniami współbieżności. Burza eksmisji KV cache przy obciążeniu skokowym powoduje kaskady ponownego prefillu, które nasycają obliczenia.

Żaden z tych nie pojawia się w testach jednostkowych. Inżynieria chaosu to sposób, w jaki odkrywasz je, zanim zrobią to użytkownicy.

## The Concept

### Wymagania wstępne

Nie uruchamiaj chaosu w produkcji bez:

1. **SLI/SLO** — zdefiniowane wskaźniki i cele poziomu usług.
2. **Obserwowalność** — ślady, metryki, logi, podłączone do dashboardów.
3. **Automatyczne wycofanie** — Phase 17 · 20 wycofanie przez flagę polityki.
4. **Runbooki** — strukturalne, Phase 17 · 23.
5. **Dyżur** — ktoś do reagowania.

Brak któregokolwiek oznacza, że chaos staje się prawdziwym incydentem.

### Cztery płaszczyzny + sprzężenie zwrotne

**Płaszczyzna sterowania** — harmonogram eksperymentów (workflow Litmus, harmonogram Chaos Mesh, interfejs Harness).

**Płaszczyzna celu** — usługi, pody, węzły, load balancery, magazyny danych.

**Płaszczyzna bezpieczeństwa** — wyłącznik awaryjny, okna wyciszania, limity promienia rażenia, bramki budżetu błędów.

**Płaszczyzna obserwowalności** — normalne metryki + korelacja ID śladu do odróżnienia awarii wywołanych chaosem od naturalnych.

**Pętla sprzężenia zwrotnego** — ustalenia zasilają dostosowanie SLO, aktualizacje runbooków, poprawki kodu.

### Zabezpieczenia są obowiązkowe

- **Alarm tempa wypalania**: wstrzymaj eksperyment, jeśli dzienne zużycie budżetu błędów przekracza 2x oczekiwanego.
- **Okna wyciszania**: wycisz alerty nieeksperymentalne w promieniu rażenia podczas eksperymentu.
- **Korelacja ID śladu**: wszystkie błędy wywołane eksperymentem niosą tag, aby dyżurny mógł deduplikować.

### Pięć eksperymentów specyficznych dla LLM

1. **Przeciążenie pamięci** — wymuś burzę wywłaszczeń KV cache, wysyłając żądania długiego kontekstu z dużą współbieżnością. Obserwuj: czy usługa łagodnie odrzuca czy ulega awarii?

2. **Awaria sieci** — odetnij łączność między bramką inferencyjną a dostawcą. Obserwuj: czy fallback uruchamia się w ramach SLA? (Phase 17 · 19)

3. **Symulacja awarii dostawcy** — 100% 429 z OpenAI. Obserwuj: czy routing przełącza się na Anthropic? (Phase 17 · 16, 19)

4. **Zniekształcony prompt** — wstrzyknij ładunek blokujący tokenizator (np. głęboko zagnieżdżony unicode, ogromny punkt kodowy UTF-8). Obserwuj: czy pojedyncze żądanie blokuje workera?

5. **Burza eksmisji KV** — wymuś eksmisję przez nasycenie budżetu bloków vLLM. Obserwuj: czy LMCache odzyskuje, czy usługa ulega degradacji?

### Kadencja

- **Cotygodniowe** — małe eksperymenty kanarkowe w stagingu, może 5% produkcji.
- **Comiesięczne** — zaplanowany game day na konkretnym scenariuszu; udział międzyzespołowy; analiza powdrożeniowa.
- **Kwartalne** — międzyzespołowy audyt odporności; aktualizacja mapy zależności.

### Narzędzia

- **Harness Chaos Engineering** — komercyjne; rekomendacje eksperymentów pochodzące z AI; skalowanie w dół promienia rażenia; integracja z narzędziami MCP.
- **LitmusChaos** — absolwent CNCF; oparty na workflow Kubernetes.
- **Chaos Mesh** — piaskownica CNCF; natywny Kubernetes styl CRD.
- **Gremlin** — komercyjne; szerokie wsparcie.
- **AWS FIS** / **Azure Chaos Studio** — zarządzane oferty chmurowe.

### Rozpoczęcie od małego

Pierwszy eksperyment: zabij jeden pod decode przy stałym ruchu. Obserwuj przekierowanie i odzyskiwanie. Jeśli to działa i wygląda bezpiecznie, przejdź do chaosu sieciowego.

Pierwszy eksperyment specyficzny dla LLM: wstrzyknij jeden 429 dostawcy na 5 minut. Obserwuj fallback. Większość zespołów odkrywa, że ich fallback nie został w pełni przetestowany.

### Liczby, które warto zapamiętać

- Cztery płaszczyzny: sterowania, celu, bezpieczeństwa, obserwowalności.
- Wstrzymanie tempa wypalania: 2x oczekiwanego dziennego zużycia budżetu.
- Kadencja: cotygodniowe kanarkowe, comiesięczny game day, kwartalny audyt.
- Pięć eksperymentów LLM: pamięć, sieć, dostawca, zniekształcony prompt, burza KV.

## Use It

`code/main.py` symuluje trzy eksperymenty chaosu z bramkami płaszczyzny bezpieczeństwa. Raportuje, które eksperymenty uruchomiłyby przerwanie tempa wypalania.

## Ship It

Ta lekcja produkuje `outputs/skill-chaos-plan.md`. Na podstawie stosu i dojrzałości wybiera pierwsze trzy eksperymenty i narzędzia.

## Exercises

1. Uruchom `code/main.py`. Który eksperyment uruchamia bramkę tempa wypalania i dlaczego?
2. Zaprojektuj pierwsze pięć eksperymentów chaosu dla usługi RAG opartej na vLLM. Dołącz kryteria sukcesu.
3. Twój alarm tempa wypalania wstrzymał eksperyment. Jak określasz przyczynę źródłową — chaos czy naturalną?
4. Uzasadnij, czy chaos powinien być uruchamiany w produkcji czy tylko w stagingu. Kiedy produkcja jest właściwą odpowiedzią?
5. Wymień trzy tryby awarii specyficzne dla LLM, których generyczny chaos sieciowy nie może odtworzyć.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SLI / SLO | "service targets" | Indicator + objective; required prerequisite |
| Blast radius | "scope" | Set of services / users affected by experiment |
| Burn-rate alert | "budget gate" | Fires when error-budget burn rate > 2x expected |
| Game day | "monthly drill" | Scheduled cross-team chaos exercise |
| LitmusChaos | "CNCF workflow" | Graduated CNCF Kubernetes chaos tool |
| Chaos Mesh | "CNCF CRD" | CNCF sandbox Kubernetes-native chaos |
| Harness CE | "commercial AI-assisted" | Harness chaos with AI recommendations |
| Malformed prompt | "tokenizer bomb" | Input that stalls tokenization |
| KV eviction storm | "preemption cascade" | Mass eviction triggering re-prefills |

## Further Reading

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)