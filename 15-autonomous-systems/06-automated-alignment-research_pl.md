# Automatyczne Badania nad Alignmentem (Anthropic AAR)

> Anthropic uruchomił równoległe zespoły autonomicznych badaczy alignmentu Claude Opus 4.6 w niezależnych piaskownicach, koordynujących się za pośrednictwem współdzielonego forum, którego logi znajdują się poza każdą piaskownicą (więc agenci nie mogą usuwać swoich własnych zapisów). Na problemie trenowania słaby-do-silnego, AAR-y przewyższyły ludzkich badaczy. Własne podsumowanie Anthropic zaznacza, że narzucone przepływy pracy często ograniczają elastyczność AAR i pogarszają wydajność. Automatyzacja badań nad alignmentem to krok kompresji, który kompresuje oś czasu do dokładnie tych zagrożeń niewspółosiowością, które RSP ma wykrywać.

**Type:** Learn
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## Problem

Badania nad alignmentem są kosztowne w czasie ludzkich badaczy. Problemy takie jak skalowalny nadzór, specyfikacja nagród czy trenowanie słaby-do-silnego wymagają eksperymentów, które trwają tygodnie na iterację. Wraz z postępem możliwości modeli granicznych, obciążenie alignmentem rośnie szybciej niż podaż wykwalifikowanych badaczy.

Automatyczne Badania nad Alignmentem (AAR) pytają, czy te same modele graniczne, których możliwości wyprzedzają alignment, mogą przyczynić się do zamknięcia luki. Raport Anthropic z 2026 roku o badaniu trenowania słaby-do-silnego prowadzonym przez AAR (alignment.anthropic.com/2026/automated-w2s-researcher/) jest jednym z pierwszych publicznych wyników z wdrożonego systemu tej klasy.

Wynik jest autentycznie pozytywny: AAR-y rozwiązały problem badawczy lepiej niż ludzki baseline. Wynik również koncentruje, w pojedynczym systemie, każdą obawę, którą ta faza rozwinęła. Jeśli badania nad alignmentem mogą być zautomatyzowane, to samo dotyczy części, które kompromitują zabezpieczenia. Progi RSP dla możliwości AI w zakresie badań i rozwoju są pisane z tą pętlą w myślach.

## Koncepcja

### Architektura (jak opisano publicznie)

- **Równoległe agenty.** Wiele instancji Claude Opus 4.6 działa jako AAR-y, każda w niezależnej piaskownicy. Piaskownice ograniczają powierzchnię systemu plików, sieci i procesów.
- **Współdzielone forum.** Ustalenia są publikowane na współdzielonym forum, którego backend przechowywania znajduje się poza każdą piaskownicą. Agenci mogą czytać forum; agenci nie mogą usuwać ani edytować zapisów forum z własnej piaskownicy.
- **Alokacja zadań.** Kolejka zadań zaprojektowana przez człowieka przypisuje problemy badawcze do AAR. AAR-y proponują dekompozycje, uruchamiają eksperymenty, piszą ustalenia.
- **Koncentracja na trenowaniu słaby-do-silnego.** Opublikowane badanie dotyczyło trenowania słaby-do-silnego — używania słabszego modelu nadzorcy do trenowania silniejszego modela ucznia. AAR-y przewyższyły ludzkich badaczy w tym zadaniu.

### Dlaczego log poza piaskownicą ma znaczenie

Właściwość integralności logu to różnica między kontrolowanym eksperymentem a potencjalnie skompromitowanym. Gdyby AAR-y mogły edytować własne logi, mogłyby ukrywać nieudane lub wątpliwe eksperymenty. Anthropic bezpośrednio to sygnalizuje: ta właściwość jest warunkiem koniecznym do zaufania wynikom badawczym i jest nietrywialna do zweryfikowania.

Analogia w kategoriach infrastruktury to logi append-only z zapisem do zewnętrznej pamięci (np. S3 object-lock, niezmienne logi Datadog). Agent nie ma poświadczeń do modyfikowania przeszłych zapisów. To ten sam wzór, którego zespoły bezpieczeństwa korporacyjnego używają w 2026 roku do tworzenia odpornych na ransomware śladów audytu.

### Kompromis narzuconego przepływu pracy

Raport Anthropic zauważa, że narzucone przepływy pracy często ograniczają elastyczność AAR i pogarszają wydajność. Innymi słowy: jeśli ludzki projektant pisze plan badań krok po kroku, AAR-y działają gorzej niż w przypadku celów wysokiego poziomu i swobody dekompozycji. Jest to zgodne z wynikami w agencyjnych badaniach (LangGraph, Microsoft Agent Framework) — zbyt wyspecyfikowane szkielety szkodzą zdolnym agentom.

Implikacja bezpieczeństwa jest subtelna. Elastyczność, która poprawia wydajność AAR, jest również elastycznością, która pozwala na niesankcjonowaną dekompozycję, nowe użycie narzędzi i dryf celów. Kompromis to decyzja projektowa: ile elastyczności kupuje wystarczającą zdolność, aby pokonać ludzi, bez dawania agentowi przestrzeni do uogólniania celów poza dystrybucją?

### Ryzyko kompresji

RSP v3.0 (Lekcja 19) wprowadza próg możliwości AI w zakresie badań i rozwoju: zdolność do pełnej automatyzacji pipeline'u badań i rozwoju AI po konkurencyjnym koszcie w porównaniu z człowiekiem + narzędziami AI. FSF v3 DeepMind zawiera analogiczny poziom autonomii badań ML. Oba frameworki traktują ten próg jako wyzwalacz dla podwyższonych kontroli.

AAR jest jeden krok od progu: automatyzuje część pipeline'u (badania nad alignmentem na konkretnych, dobrze zakreslonych zadaniach), ale nie pełną pętlę rozwoju możliwości end-to-end. Pytanie o oś czasu brzmi, jak szybko luka się zamyka.

Skompresowane osie czasu to kumulatywna obawa o awarię. Jeśli badania nad alignmentem i badania nad możliwościami kumulują się w podobnym tempie, powierzchnia ryzyka niewspółosiowości rośnie co najmniej tak szybko, jak możliwości. Jeśli możliwości kumulują się szybciej (historyczny trend), luka się powiększa. To jest argument za AAR jako kwalifikowanym dobrem: każdy dodatkowy wynik alignmentu zmniejsza lukę wtedy i tylko wtedy, gdy proces badawczy jest godny zaufania.

### Czego AAR nie zastępuje

Ludzcy badacze ustawiają kolejkę zadań, przeglądają wyniki i sprawują władzę konstytucyjną. AAR-y przyspieszają środek pipeline'u, a nie końce. Opublikowane wyniki Anthropic obejmują zarówno wkład AAR, jak i osąd ludzkiego badacza co do tego, co opublikować, co wycofać i co dopracować.

To pasuje do wzorca proponuj-następnie-zatwierdź z Lekcji 15 zastosowanego do samych badań: AAR-y proponują; ludzie zatwierdzają.

## Użyj

`code/main.py` symuluje wielo-AAR forum badawcze. Trzy niezależne "agenty badawcze" działają równolegle, publikując ustalenia na append-only forum z opcjonalnymi próbami manipulacji.

Obserwuj:

- Jak logowanie poza piaskownicą łapie próby manipulacji logami.
- Jak alokacja zadań z wolną dekompozycją przewyższa stałe przepływy pracy przy trudniejszych problemach.
- Jak pojedynczy skompromitowany AAR może zanieczyścić sygnał forum, jeśli log nie jest odporny na manipulacje.

## Dostarcz

`outputs/skill-aar-deployment-review.md` to przegląd przedwdrożeniowy pipeline'u automatycznych badań nad alignmentem.

## Ćwiczenia

1. Uruchom `code/main.py`. Porównaj ustawienia "fixed-workflow" vs "free-decomposition". Czy wolna dekompozycja zawsze wygrywa, czy są klasy problemów, gdzie stały przepływ pracy jest konkurencyjny?

2. Zmodyfikuj symulator, aby jeden agent próbował manipulować logami. Potwierdź, że append-only log to wykrywa. Napisz jeden akapit opisujący dokładnie, jak wygląda sygnatura wykrycia w logu.

3. Przeczytaj raport Anthropic o słaby-do-silnego AAR. Zidentyfikuj konkretne podzadanie, w którym AAR-y pokonały ludzkich badaczy. Co uczyniło je podatnym na automatyzację?

4. Zaprojektuj politykę alokacji kolejki zadań, która równoważy elastyczność AAR (lepsze wyniki) z ograniczeniami narzuconego przepływu pracy (łatwiejszy audyt). Opisz, jak przeprowadziłbyś test A/B obu.

5. Przeczytaj próg AI R&D-4 RSP v3.0. W jednym akapicie opisz, co Twoim zdaniem przekroczyłoby ten próg, czego AAR obecnie nie robi.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|---|---|---|
| AAR | "Automatyczny badacz alignmentu" | Instancja Claude Opus 4.6 działająca autonomicznie nad problemami alignmentu |
| Trenowanie słaby-do-silnego | "Trenowanie silniejszego modela słabszym nadzorcą" | Klasyczny benchmark skalowalnego nadzoru, na którym AAR-y przewyższyły ludzi |
| Współdzielone forum | "Gdzie agenci publikują ustalenia" | Append-only, przechowywanie poza piaskownicą |
| Log poza piaskownicą | "Agent nie może edytować własnego zapisu" | Odporny na manipulacje zapis do zewnętrznej pamięci |
| Narzucony przepływ pracy | "Plan krok po kroku od ludzkiego projektanta" | Ogranicza AAR; często pogarsza wydajność vs wolna dekompozycja |
| Wolna dekompozycja | "Agent decyduje, jak podzielić zadanie" | Bardziej zdolna, trudniejsza do audytu |
| Próg badań i rozwoju AI | "Poziom zdolności RSP/FSF" | Pełna automatyzacja pipeline'u badań i rozwoju po konkurencyjnym koszcie |
| Skompresowana oś czasu | "Wyścig alignmentu z możliwościami" | Jeśli możliwości kumulują się szybciej niż alignment, ryzyko niewspółosiowości rośnie |

## Dalsza lektura

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — źródło podstawowe.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — ramy progu badań i rozwoju AI.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — szersze ramy autonomii agentów.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — poziomy autonomii badań ML równoległe do RSP.
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — podstawowy problem, który AAR-y atakowały.