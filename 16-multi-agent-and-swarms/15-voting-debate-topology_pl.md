# Głosowanie, Samozgodność i Topologia Debaty

> Najtańsza agregacja: próbkuj N niezależnych agentów, podejmuj decyzję większością głosów. Samozgodność Wang i in. 2022 realizowała to poprzez jednokrotne próbkowanie modelu N razy. Podejście wieloagentowe rozszerza to o **heterogenicznych** agentów, aby uniknąć monokultury — różne modele, różne prompty, różne temperatury, różne konteksty. Poza głosowaniem większościowym, topologia debaty ma znaczenie: MultiAgentBench (arXiv:2503.01935, ACL 2025) oceniał koordynację w topologiach gwiazda / łańcuch / drzewo / graf i stwierdził, że **graf jest najlepszy dla zadań badawczych**, z „podatkiem koordynacyjnym" powyżej ~4 agentów. AgentVerse (ICLR 2024) dokumentuje dwa emergencjalne wzorce — zachowania ochotnicze i konformistyczne — przy czym konformizm jest zarówno atutem (osiąganie konsensusu), jak i ryzykiem (myślenie grupowe, Lekcja 24). Ta lekcja mapuje przestrzeń topologii, buduje każdy wariant i mierzy podatek koordynacyjny.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 minutes

## Problem

Debata może poprawić dokładność (Du i in., arXiv:2305.14325). Może ją też pogorszyć. To, czy debata pomaga, zależy od czterech wyborów strukturalnych:

1. Kto z kim rozmawia (topologia).
2. Liczba rund (Du 2023: zarówno rundy, jak i agenci mają niezależne znaczenie).
3. Czy agenci są heterogeniczni (różne modele bazowe przełamują monokulturę).
4. Czy obecny jest głos adversarialny (argumentowanie za tezą vs. argumentowanie przeciwko).

Zespoły, które „podpinają 5 agentów i głosowanie" do zadania, często osiągają gorsze wyniki niż pojedynczy agent. Te porażki nie są przypadkowe. Są związane z topologią i heterogenicznością. Ta lekcja jest mapą topologii.

## Koncepcja

### Samozgodność, baseline z jednym modelem

Wang i in. 2022 („Self-Consistency Improves Chain of Thought Reasoning") próbkowali ten sam model N razy w temperaturze > 0 i stosowali głosowanie większościowe na ścieżkach rozumowania. Wynik na GSM8K: znaczące zyski przy N=40 próbkach w porównaniu do pojedynczego dekodowania zachłannego. Samozgodność jest prekursorem wieloagentowego głosowania w ramach jednego agenta.

Ograniczenie: samozgodność używa jednego modelu bazowego. Błędy są skorelowane z założenia. Jeśli model ma systematyczne obciążenie, wszystkie N próbek je dzielą.

### Głosowanie wieloagentowe, rozszerzenie heterogeniczne

Zastąp N próbek N *różnymi* agentami. Różne modele bazowe (Claude, GPT, Llama), różne prompty, różny dostęp do narzędzi. Korzyść: nieskorelowane błędy. Koszt: różni agenci kosztują różne kwoty; koordynacja dodaje narzut.

Kanoniczna nazwa z 2026 roku dla heterogenicznej debaty to **A-HMAD** — Adversarial Heterogeneous Multi-Agent Debate (Adwersarialna Heterogeniczna Debeta Wieloagentowa). Nie jest powszechnie przyjęta, ale artykuły używają tego terminu na określenie „debaty różnych modeli, która redukuje skorelowane błędy wynikające z załamania monokultury".

### Cztery topologie

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (w pełni połączone)
```

Gwiazda: jeden hub, wszyscy inni rozmawiają tylko z hubem. Odpowiednik nadzorca-pracownik bez kanałów bocznych.
Łańcuch: liniowy, każdy agent widzi wynik poprzednika. Przypomina potok.
Drzewo: hierarchiczne, używane przez hierarchiczne systemy agentowe (Lekcja 06).
Graf: dowolny z dowolnym. Obejmuje w pełni połączoną klikę i dowolne DAGi.

### Podatek koordynacyjny (MultiAgentBench)

MultiAgentBench (MARBLE, ACL 2025, arXiv:2503.01935) porównywał topologie gwiazda, łańcuch, drzewo, graf w zestawie zadań obejmujących badania, kodowanie i planowanie. Kluczowe zmierzone wyniki:

- **Graf** wygrywa w zadaniach badawczych. Informacja przepływa dowolnie; agenci mogą się wzajemnie krytykować.
- **Gwiazda** wygrywa w zadaniach faktograficznych wymagających szybkiej odpowiedzi. Hub filtruje i konsoliduje.
- **Łańcuch** wygrywa w potokach krokowych (stopniowe udoskonalanie).
- **Podatek koordynacyjny** pojawia się powyżej ~4 agentów w topologii grafu. Koszt czasu rzeczywistego i tokenów rośnie szybciej niż jakość.

Granica 4 agentów jest empiryczna, nie fundamentalna. Odzwierciedla pojemność kontekstu modeli LLM w 2026 roku: kontekst każdego agenta wypełnia się wynikami innych, a wartość krańcowa dodania agenta N+1 spada, gdy wszyscy widzą wszystkich.

### Strategie Debaty Wieloagentowej („Czy powinniśmy wariować?")

arXiv:2311.17371 to przegląd strategii MAD z 2023 roku. Kluczowy wniosek potwierdzony przez innych: warianty MAD, które są *strukturalnie podobne* do samozgodności (niezależne próbkowanie + agregacja), często osiągają gorsze wyniki niż samozgodność przy tym samym budżecie. MAD pomaga najbardziej, gdy agenci są rzeczywiście heterogeniczni, a debata ma strukturę adversarialną (jeden agent argumentuje przeciw).

### Emergencjalne wzorce AgentVerse

AgentVerse (ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) dokumentuje dwa zachowania wyłaniające się z debaty wieloagentowej nawet bez jawnego projektowania:

- **Ochotnicze.** Agent oferuje pomoc („Mogę zrobić następny krok") bez monitowania. Przydatne: przydziela pracę najbardziej kompetentnemu agentowi do podzadania.
- **Konformistyczne.** Agent dostosowuje swoje stanowisko do krytyka, nawet gdy krytyk się myli. Jest to odpowiednik sykofancji w debacie (Lekcja 14).

Konformizm jest powodem, dla którego debata do momentu osiągnięcia porozumienia nagradza agresywnych. Ograniczone rundy z oddzielnym sędzią łagodzą ten problem.

### Heterogeniczność: faktyczny parametr wpływający na dokładność

Wzorzec z lat 2024-2026 w praktycznej literaturze: zastąpienie jednego z N agentów innym modelem bazowym daje większy wzrost dokładności niż zwiększenie N o 1. Intuicja to monokultura — każde nowe źródło niezależnych błędów jest warte więcej niż dodatkowa skorelowana próbka.

W granicy, heterogeniczność przewyższa liczność. Trzy różne modele biją pięć kopii jednego modelu w większości zadań z czystą prawdą podstawową.

### Metody ławy przysięgłych (jury)

Ramy Sibyl (cytowane w literaturze Minsky-LLM) formalizują „ławę przysięgłych" — mały zestaw wyspecjalizowanych agentów, którzy udoskonalają odpowiedzi poprzez głosowanie na każdym etapie. W przeciwieństwie do zwykłego głosowania większościowego, ława przysięgłych ma role: jeden agent przesłuchuje, jeden dostarcza kontekst, jeden ocenia prawdopodobieństwo. Metody ławy przysięgłych stanowią punkt pośredni między zwykłym głosowaniem (tanie, podatne na monokulturę) a pełnym MAD (drogie, podatne na konformizm).

### Kiedy głosowanie z debatą dominuje

- Pytanie ma prawdę podstawową (fakt, matematyka, zachowanie kodu). Zbieżność głosowania jest znacząca.
- Agenci mogą uzyskiwać dostęp do różnych źródeł lub narzędzi (heterogeniczność jest dostępna).
- Liczba rund jest ograniczona (zwykle 2-3) i istnieje oddzielny sędzia lub weryfikator.
- Budżet pozwala na 3-5 agentów. Powyżej 5-7 w topologii grafu dominuje podatek koordynacyjny.

### Kiedy głosowanie z debatą szkodzi

- Pytanie ma charakter opinii. Agenci zbiegają się do odpowiedzi, która wygląda na najbardziej pewną, a nie najbardziej poprawną.
- Wszyscy agenci dzielą jeden model bazowy. Monokultura sprawia, że konsensus jest bez znaczenia.
- Liczba rund jest nieograniczona. Konformizm zawsze wygrywa.
- Zadanie jest proste. Pojedynczy agent z samozgodnością przy N=5 jest tańszy i równie dokładny.

## Build It

`code/main.py` implementuje:

- `run_star(agents, hub, question)` — hub odpytuje każdego pracownika, agreguje.
- `run_chain(agents, question)` — sekwencyjne udoskonalanie.
- `run_tree(root, children, question)` — hierarchiczne z agregacją na dwóch poziomach głębokości.
- `run_graph(agents, question, rounds)` — debata wszystkich ze wszystkimi, ograniczona liczba rund.
- Skryptowany parametr heterogeniczności: każdy agent ma `error_bias` określający jego systematyczną błędność.
- Harness pomiarowy uruchamiający każdą topologię przy N=3, 5, 7 i raportujący (dokładność, liczba_tokenów, symulowany_czas).

Uruchomienie:

```
python3 code/main.py
```

Oczekiwane wyjście: tabela topologia × N → (dokładność, tokeny, opóźnienie). Graf wygrywa przy N=3-5 w zadaniach badawczych; gwiazda wygrywa w zadaniach szybko-faktograficznych; graf przy N=7 pokazuje podatek koordynacyjny (opóźnienie rośnie szybciej niż dokładność).

## Use It

`outputs/skill-topology-picker.md` to umiejętność, która czyta opis zadania i rekomenduje topologię (gwiazda / łańcuch / drzewo / graf), N (liczbę agentów), profil heterogeniczności (modele bazowe do użycia) i ograniczenie rund.

## Ship It

Dla każdego zespołu agentów:

- Zacznij od **samozgodności przy N=5** używając jednego silnego modelu bazowego. To tani baseline.
- Przejdź na **głosowanie heterogeniczne przy N=3**, jeśli dokładność ma znaczenie. Zmierz różnicę.
- Przejdź na **topologię debaty** tylko wtedy, gdy zadanie ma strukturę (badania, wieloetapowość) i możliwe jest ograniczenie rund.
- Zawsze loguj klaster mniejszościowy. Gdy mniejszość uporczywie ma rację, masz sygnał różnorodności.
- Mierz czas rzeczywisty i liczbę tokenów obok dokładności. „Lepsza dokładność przy 10x koszcie" to decyzja biznesowa.

## Ćwiczenia

1. Uruchom `code/main.py`. Narysuj krzywą podatku koordynacyjnego dla topologii grafu: dokładność vs N, tokeny vs N. Przy jakim N krzywa zmienia nachylenie?
2. Zaimplementuj A-HMAD: trzech agentów z celowo różnymi obciążeniami. Jak baseline z tym samym obciążeniem wypada w porównaniu z A-HMAD w ataku monokultury z Lekcji 14?
3. Dodaj rolę „sędziego" do topologii grafu, który nie głosuje, a jedynie ocenia końcowy konsensus. Czy zmienia to emergentne zachowanie konformistyczne?
4. Przeczytaj artykuł AgentVerse (ICLR 2024). Zidentyfikuj, które emergentne zachowanie twoja implementacja wykazuje najsilniej. Czy możesz wywołać przeciwne zachowanie poprzez zmianę promptu?
5. Przeczytaj MultiAgentBench (arXiv:2503.01935) Sekcja 4 (eksperymenty z topologią). Odtwórz wynik „graf-wygrywa-badania" na jednym zadaniu z artykułu używając swojego harnessu.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| Samozgodność | „Próbkuj N razy, głosuj" | Wang 2022. Pojedynczy model, N próbek w temperaturze > 0, głosowanie większościowe na ścieżkach rozumowania. |
| Heterogeniczność | „Różne modele" | Zbiór różnych modeli bazowych lub rodzin promptów. Przełamuje monokulturę. |
| MAD | „Debata wieloagentowa" | Ogólny termin na wymianę krytyk między agentami w rundach. Zobacz Du 2023. |
| A-HMAD | „Adwersarialna Heterogeniczna MAD" | Wariant MAD kładący nacisk na różne modele + strukturę adversarialną. |
| Topologia | „Kto z kim rozmawia" | Gwiazda, łańcuch, drzewo, graf. Określa przepływ informacji. |
| Podatek koordynacyjny | „Malejące zyski" | Powyżej ~4 agentów w grafie koszt rośnie szybciej niż jakość. |
| Zachowanie ochotnicze | „Nieproszona pomoc" | Emergencjalny wzorzec AgentVerse: agent oferuje wykonanie kroku. |
| Zachowanie konformistyczne | „Zgodność pod presją" | Emergencjalny wzorzec AgentVerse: agent dostosowuje się do krytyka. |
| Ława przysięgłych (jury) | „Mały panel specjalistów" | Zespół w stylu Sibyl z rolami (egzaminator, kontekst, oceniający). |

## Dalsza Literatura

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — baseline z jednym modelem
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — zarówno agenci, JAK i rundy mają niezależne znaczenie
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — benchmark topologii pokazujący graf najlepszy dla badań, łańcuch dla potoków
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — przegląd strategii MAD; stwierdza, że MAD często przegrywa z samozgodnością przy równym budżecie
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — emergencjalne wzorce ochotnicze i konformistyczne
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — referencyjna implementacja benchmarku