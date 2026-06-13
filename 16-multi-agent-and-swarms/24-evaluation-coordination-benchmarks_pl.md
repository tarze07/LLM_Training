# Ewaluacja i Benchmarki Koordynacji

> Pięć benchmarków z lat 2025–2026 pokrywa przestrzeń ewaluacji wieloagentowej. **MultiAgentBench / MARBLE** (ACL 2025, arXiv:2503.01935) ocenia topologie gwiazda/łańcuch/drzewo/graf z milowymi KPI; **graf jest najlepszy do badań**, planowanie kognitywne dodaje ~3% osiągnięcia kamieni milowych. **COMMA** ocenia multimodalną koordynację z asymetryczną informacją; najnowocześniejsze modele, w tym GPT-4o, mają trudności z pokonaniem losowej linii bazowej. **MedAgentBoard** (arXiv:2505.12371) pokrywa cztery kategorie zadań medycznych i często stwierdza, że wiele agentów nie dominuje nad pojedynczym LLM. **AgentArch** (arXiv:2509.10769) benchmarkuje architektury agentów korporacyjnych łączące użycie narzędzi + pamięć + orkiestrację. **SWE-bench Pro** ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)) zawiera 1865 problemów w 41 repozytoriach obejmujących aplikacje biznesowe, usługi B2B i narzędzia deweloperskie; modele graniczne uzyskują ~23% na Pro w porównaniu do 70%+ na Verified — rzeczywistość dotycząca kontaminacji. Claude Opus 4.7 (kwiecień 2026) raportowany jest na **64.3%** na Pro z jawną koordynacją zespołów agentów (brak opublikowanego podstawowego źródła Anthropic — traktuj jako wstępne); Verdent (scaffold agenta) osiąga **76.1% pass@1** na Verified ([raport techniczny Verdent](https://www.verdent.ai/blog/swe-bench-verified-technical-report)). **AAAI 2026 Bridge Program WMAC** (https://multiagents.org/2026/) jest punktem centralnym społeczności w 2026. Ta lekcja opiera się na metrykach MARBLE, przeprowadza przegląd topologia-vs-metryka i ugruntowuje zasadę "samo przejście SWE-bench Verified nie jest dowodem generalizacji."

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 15 (Voting and Debate Topology), Phase 16 · 23 (Failure Modes)
**Time:** ~75 minutes

## Problem

Gdy artykuł twierdzi "nasz system wieloagentowy jest lepszy," pytanie brzmi: lepszy od czego, w czym, jak mierzony? Era 2023–2024 ewaluacji wieloagentowej była chaosem — każdy wybierał własne metryki, własne linie bazowe i własne zestawy zadań. Benchmarki z lat 2025–2026 narzuciły strukturę.

Bez wspólnych benchmarków nie można sensownie porównać dwóch systemów wieloagentowych. Co gorsza, bez benchmarków typu hold-out, modele graniczne mogą się skontaminować. SWE-bench Verified został częściowo skontaminowany w korpusach treningowych do połowy 2025; wyniki modeli granicznych zostały zawyżone; Pro został zaprojektowany jako nieskontaminowana kontrola rzeczywistości.

Ta lekcja wymienia pięć kanonicznych benchmarków z 2026, nazywa co każdy mierzy i uczy sceptycznego czytania twierdzeń benchmarkowych.

## Koncepcja

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935. Ocenia cztery topologie koordynacji (gwiazda, łańcuch, drzewo, graf) w zadaniach badawczych, programistycznych i planistycznych. Milowe KPI śledzą postęp częściowy, a nie tylko końcowy sukces.

Zmierzone wyniki:

- **Graf** najlepszy dla scenariuszy badawczych; wspiera krytykę każdy-z-każdym.
- **Łańcuch** najlepszy do programowania z krokowym udoskonalaniem.
- **Gwiazda** najlepsza do szybkiej konsolidacji faktów.
- **Podatek koordynacyjny** pojawia się powyżej ~4 agentów na grafie.
- **Planowanie kognitywne** dodaje ~3% osiągnięcia kamieni milowych we wszystkich topologiach.

Użyj, gdy: chcesz porównać topologie koordynacji obiektywnie. Repozytorium MARBLE (https://github.com/ulab-uiuc/MARBLE) udostępnia ewaluator.

### COMMA — multimodalna asymetryczna informacja

Obejmuje zadania, w których agenci mają różne modalności obserwacji i muszą koordynować bez pełnego udostępniania informacji. Raportowany wynik jest niepokojący: modele graniczne, w tym GPT-4o, mają trudności z pokonaniem **losowej linii bazowej** w kooperacji agent-agent w COMMA. Sygnał jest taki, że modalności wieloagentowe są niedotrenowane i niedoewaluowane — LLMy radzą sobie rozsądnie z kooperacją jednomodalną; koordynacja multimodalna się załamuje.

Użyj, gdy: twój system ma koordynację multimodalną lub z asymetryczną informacją. Wynik zerowy z COMMA jest ostrzeżeniem, aby mierzyć przed twierdzeniem.

### MedAgentBoard — domenowy test warunków skrajnych

arXiv:2505.12371. Cztery kategorie zadań medycznych: diagnoza, planowanie leczenia, generowanie raportów, komunikacja z pacjentem. Porównuje wiele agentów vs pojedynczy LLM vs konwencjonalne systemy regułowe.

Wnioski: wiele agentów NIE dominuje nad pojedynczym LLM w większości kategorii. Przewaga wieloagentowa jest wąska — dekompozycja zadań pomaga, gdy podzadania są wyraźnie rozdzielne (diagnoza + leczenie); szkodzi, gdy koszty koordynacji przewyższają zysk ze specjalizacji (generowanie raportów).

Użyj, gdy: twoja domena ma wyraźne linie bazowe pojedynczego LLM. Jeśli lekcja z MedAgentBoard się uogólnia, wiele proponowanych systemów wieloagentowych jest przekonstruowanych.

### AgentArch — architektury korporacyjne

arXiv:2509.10769. Środowiska korporacyjne z użyciem narzędzi, pamięcią i orkiestracją ułożonymi warstwowo. Benchmark izoluje wkład każdej warstwy: jak bardzo pomaga dodanie narzędzi? Dodanie pamięci? Dodanie orkiestracji wieloagentowej?

Użyj, gdy: projektujesz korporacyjny stos agentów i potrzebujesz uzasadnić każdą warstwę. AgentArch pomaga uniknąć kupowania funkcji, których wartości nie możesz zmierzyć.

### SWE-bench Pro — kontroler rzeczywistości

arXiv:2509.16941. 1865 problemów w 41 repozytoriach obejmujących aplikacje biznesowe, usługi B2B i narzędzia deweloperskie. Zaprojektowany jako **nieskontaminowany** z późniejszymi granicami treningowymi. Modele graniczne uzyskują ~23% na Pro w porównaniu do 70%+ na Verified. Różnica jest sygnałem kontaminacji.

Wyniki z kwietnia 2026:
- Claude Opus 4.7 na Pro: **64.3%** (raportowane z jawną koordynacją zespołów agentów; brak opublikowanego podstawowego źródła Anthropic — traktuj jako wstępne).
- Verdent (scaffold agenta) na Verified: **76.1% pass@1** ([raport techniczny](https://www.verdent.ai/blog/swe-bench-verified-technical-report)).
- Surowe wyniki modeli granicznych na Pro bez scaffoldingu agentów: ~23-35% ([artykuł SWE-bench Pro](https://arxiv.org/abs/2509.16941)).

Wniosek: "pokonaliśmy SWE-bench Verified" nie jest już dowodem możliwości. Pro jest obecnym testem kwalifikacyjnym. Scaffolding zespołów agentów daje mierzalne zyski na Pro (~30-40 punktów różnicy), co jest jednym z najsilniejszych empirycznych argumentów za koordynacją wieloagentową w 2026.

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — Workshop on Multi-Agent Coordination (https://multiagents.org/2026/). Punkt centralny społeczności w 2026 dla badań nad AI wieloagentową. Zaakceptowane artykuły i materiały warsztatowe są kanonicznym miejscem oceny nowych metod; przy decyzjach produkcyjnych odwołuj się do twierdzeń zaakceptowanych przez WMAC, a nie do preprintów arXiv.

### Czytaj twierdzenia benchmarkowe sceptycznie — lista kontrolna z 2026

Gdy ktoś twierdzi wynik wieloagentowy:

1. **Który benchmark, która część?** SWE-bench Verified vs Pro ma ogromne znaczenie. Wynik podany na niewłaściwej części jest bezwartościowy.
2. **Kontrola kontaminacji.** Czy benchmark został opublikowany po dacie granicznej trenowania modelu? Jeśli nie, traktuj z ostrożnością.
3. **Porównanie z linią bazową.** W porównaniu z pojedynczym LLM, z losową linią bazową, z wcześniejszymi pracami wieloagentowymi. Nie "z niewyjustowaną wersją tego samego systemu."
4. **Istotność statystyczna.** N prób, wartość p, przedział ufności. Modele graniczne mają wysoką wariancję; pojedyncze uruchomienia mylą.
5. **Różnorodność zadań.** Jedno zadanie czy wiele? Generalizacja ma znaczenie dla produkcji.
6. **Ujawnienie kosztów.** Tokeny na zadanie, czas ścienny. Rozwiązanie na 90% przy 20x koszcie to decyzja biznesowa, a nie twierdzenie o możliwościach.

### Czego żaden z benchmarków dobrze nie mierzy

- **Długoterminowa koordynacja.** Dni interakcji w czasie ściennym. Wszystkie obecne benchmarki działają krótko.
- **Odporność na działania adversarialne.** Co się dzieje, gdy jeden agent jest złośliwy lub skompromitowany?
- **Dryf podczas wdrożenia.** Benchmarki są statyczne; dystrybucje produkcyjne się zmieniają.
- **Wydajność znormalizowana kosztem.** Większość benchmarków raportuje surową dokładność, a nie dokładność-na-dolara.

Zbudowanie własnego wewnętrznego benchmarku dla osi, która cię faktycznie interesuje, jest często właściwym posunięciem.

## Build It

`code/main.py` to nieinteraktywny przegląd:

- Symuluje 3 systemy wieloagentowe na zabawkowym zadaniu.
- Oblicza metryki milowe w stylu MARBLE dla każdego.
- Przeprowadza kontrolę kontaminacji, wstrzymując zadania z zestawu "treningowego".
- Porównuje wyraźnie z losową linią bazową.
- Drukuje kartę wyników twierdzeń benchmarkowych.

Uruchom:

```bash
python3 code/main.py
```

Oczekiwane wyniki: karta wyników systemu z surową dokładnością, osiągnięciem kamieni milowych, kosztem-na-zadanie, deltą względem losowej linii bazowej i adnotacją o kontaminacji.

## Use It

`outputs/skill-benchmark-reader.md` czyta dowolne twierdzenie benchmarkowe o systemie wieloagentowym i stosuje listę kontrolną analizy. Wynik: ocena i zastrzeżenia.

## Ship It

Dyscyplina ewaluacji produkcyjnej:

- **Zbuduj wewnętrzny benchmark**, który odzwierciedla twoją rzeczywistą dystrybucję produkcyjną. Publiczne benchmarki informują, ale nie zastępują.
- **Dołącz losową linię bazową** w każdym porównaniu. Jeśli nie możesz pokonać losowej linii bazowej z dużym marginesem w zadaniu koordynacyjnym, zadanie może być źle postawione.
- **Raportuj koszt obok dokładności.** Koszt tokenów i czas ścienny. Zespoły operacyjne potrzebują obu.
- **Przebudowuj benchmark kwartalnie.** Dystrybucja produkcyjna się zmienia; nieaktualne benchmarki mylą.
- **Unikaj przetrenowania na opublikowanych benchmarkach.** Jeśli twój zespół optymalizuje konkretnie pod liczby SWE-bench Pro, cofniesz się w produkcji.

## Ćwiczenia

1. Uruchom `code/main.py`. Zidentyfikuj, który z trzech symulowanych systemów ma najlepszy koszt-na-kamień-milowy. Czy pokrywa się to z systemem o najwyższej surowej dokładności?
2. Przeczytaj MultiAgentBench (arXiv:2503.01935). Dla swojej własnej domeny zadań zdecyduj, którą z czterech topologii MARBLE by zalecił. Uzasadnij na podstawie wyników artykułu.
3. Przeczytaj artykuł SWE-bench Pro. Co konkretnie czyni go odpornym na kontaminację? Czy ta sama technika może być zastosowana do innych benchmarków, na których ci zależy?
4. Przeczytaj wnioski COMMA dotyczące koordynacji multimodalnej. Zaprojektuj proste zadanie koordynacji multimodalnej, które mógłbyś dodać do swojego wewnętrznego benchmarku. Co liczyłoby się jako użyteczny sygnał?
5. Zastosuj listę kontrolną twierdzeń benchmarkowych do jednego nagłówkowego wyniku z niedawnego artykułu o systemach wieloagentowych. Jaką ocenę dałbyś twierdzeniu?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|----------------|------------------------|
| MARBLE | "MultiAgentBench" | ACL 2025; topologie gwiazda/łańcuch/drzewo/graf z milowymi KPI. |
| COMMA | "Benchmark multimodalny" | Multimodalna koordynacja z asymetryczną informacją; modele graniczne mają trudności vs losowa. |
| MedAgentBoard | "Domenowy test warunków skrajnych" | Cztery kategorie medyczne; często stwierdza, że wiele agentów nie dominuje nad pojedynczym LLM. |
| AgentArch | "Benchmark korporacyjny" | Narzędzia + pamięć + orkiestracja warstwowo. |
| SWE-bench Pro | "Odporny na kontaminację" | 1865 problemów, 41 repozytoriów; ~23% vs 70%+ na Verified (sygnał kontaminacji). |
| Osiągnięcie kamieni milowych | "Częściowy kredyt" | Benchmarki nagradzające postęp, a nie tylko końcowy sukces. |
| Kontaminacja | "Benchmark wyciekł do treningu" | Po publikacji benchmarki trafiają do korpusów treningowych; wyniki rosną. |
| WMAC | "AAAI 2026 Bridge Program" | Workshop on Multi-Agent Coordination; punkt centralny społeczności. |

## Dalsza lektura

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — benchmark topologii z milowymi KPI
- [Repozytorium MARBLE](https://github.com/ulab-uiuc/MARBLE) — referencyjna implementacja
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — domenowy test warunków skrajnych; wiele agentów często nie dominuje
- [AgentArch](https://arxiv.org/abs/2509.10769) — architektury agentów korporacyjnych
- [Tabele liderów SWE-bench](https://www.swebench.com/) — wyniki Verified i Pro dla modeli granicznych
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — punkt centralny społeczności w 2026