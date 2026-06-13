# Benchmarki: SWE-bench, GAIA, AgentBench

> Trzy benchmarki kotwiczą ewaluację agentów w 2026 roku. SWE-bench testuje łatanie kodu. GAIA testuje użycie narzędzi przez generalistę. AgentBench testuje wnioskowanie w wielu środowiskach. Znaj ich skład, historię kontaminacji i to, czego nie mierzą.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Tool Use)
**Time:** ~60 minutes

## Learning Objectives

- Wymień szkielet testowy SWE-bench (FAIL_TO_PASS) i wyjaśnij, dlaczego bramkuje na testach jednostkowych.
- Wyjaśnij, dlaczego istnieje SWE-bench Verified (OpenAI, 500 zadań) i co usuwa.
- Opisz projekt GAII: proste dla ludzi, trudne dla AI; trzy poziomy trudności.
- Wymień osiem środowisk AgentBench i jego główną blokadę dla open-source'owych LLM.
- Podsumuj odkrycie kontaminacji SWE-bench+ i jego implikacje.

## Problem

Liderboardy mówią ci, który model wygrywa na jednym benchmarku. Nie mówią ci:

- Czy benchmark jest skontaminowany (rozwiązania w danych treningowych, wyciek testowy).
- Czy benchmark mierzy to, na czym ci zależy (kod vs przeglądanie vs generalista).
- Czy ewaluator jest solidny (dopasowanie AST, kontrole stanu, recenzja ludzka).

Poznaj trzy kotwiczące benchmarki i ich tryby awarii, zanim zacytujesz liczbę.

## Koncepcja

### SWE-bench (Jimenez i in., ICLR 2024 oral)

- 2294 rzeczywistych zgłoszeń GitHub z 12 popularnych repozytoriów Python.
- Agent otrzymuje: bazę kodu w commicie przed naprawą + opis zgłoszenia w języku naturalnym.
- Agent tworzy: łatkę.
- Ewaluator: zastosuj łatkę, uruchom zestaw testów repozytorium. Łatka musi odwrócić testy FAIL_TO_PASS (wcześniej niezaliczone, teraz zaliczone) bez psucia testów PASS_TO_PASS.

SWE-agent (Yang i in., 2024) osiągnął 12,5% w momencie wydania, kładąc nacisk na interfejsy agent-komputer (polecenia edytora plików, składnia wyszukiwania zrozumiała dla modelu).

### SWE-bench Verified

OpenAI, sierpień 2024. Ręcznie wyselekcjonowany podzbiór 500 zadań. Usuwa niejednoznaczne zgłoszenia, zawodne testy i zadania, w których naprawa była niejasna. Podstawowy benchmark dla pytania „czy twój agent wysyła prawdziwe łatki?"

### Kontaminacja

- Ponad 94% zgłoszeń SWE-bench pochodzi sprzed granic odcięcia większości modeli.
- **SWE-bench+** wykrył, że 32,67% udanych łatek wyciekło z rozwiązań w tekście zgłoszenia (model zobaczył naprawę w opisie), a 31,08% było podejrzanych z powodu słabego pokrycia testami.
- Verified jest czystszy, ale nie wolny od kontaminacji.

Praktyczna implikacja: model, który osiąga 50% na SWE-bench, może osiągnąć 35% na SWE-bench+. Zawsze podawaj oba, jeśli twierdzisz, że masz wydajność SWE-bench.

### GAIA (Mialon i in., listopad 2023)

- 466 pytań; 300 zachowanych na prywatnym liderboardzie na huggingface.co/gaia-benchmark.
- Filozofia projektu: „koncepcyjnie proste dla ludzi (92%), ale trudne dla AI (GPT-4 z wtyczkami: 15%)."
- Testuje wnioskowanie, multimodalność, sieć WWW, użycie narzędzi.
- Trzy poziomy trudności; Poziom 3 wymaga długich łańcuchów narzędzi między modalnościami.

GAIA to benchmark do mierzenia „zdolności generalisty." Nie myl z benchmarkami specyficznymi dla kodu.

### AgentBench (Liu i in., ICLR 2024)

- 8 środowisk obejmujących kod (Bash, DB, KG), gry (Alfworld, LTP), sieć (WebShop, Mind2Web) i generację otwartą.
- Wieloturowe, ~4k-13k tur na podział.
- Główne odkrycie: długoterminowe wnioskowanie, podejmowanie decyzji i podążanie za instrukcjami są blokadami dla OSS LLM w doganianiu komercyjnych.

### Czego te benchmarki nie mierzą

- Rzeczywistego kosztu operacyjnego (tokeny, czas ścienny).
- Zachowania bezpieczeństwa w warunkach adwersarialnych.
- Wydajności w twojej domenie (używaj własnych ewaluacji, Lekcja 30).
- Awarii ogonowych (benchmarki uśredniają; operatorzy produkcyjni dbają o najgorszy 1%).

### Gdzie benchmarkowanie zawodzi

- **Fiksacja na pojedynczej liczbie.** SWE-bench 50% mówi ci mniej niż rozkład kosztów P50/P75/P95 + kroków.
- **Skontaminowane twierdzenia.** Raportowanie SWE-bench bez wspominania Verified lub SWE-bench+ jest mylące.
- **Benchmark jako cel rozwoju.** Optymalizacja pod benchmark odbiega od produkcyjnej użyteczności.

## Build It

`code/main.py` implementuje zabawkowy szkielet w stylu SWE-bench:

- Syntetyczne zadania naprawy błędów (3 zadania).
- Skryptowego „agenta", który proponuje łatki.
- Uruchamiacz testów, który sprawdza FAIL_TO_PASS (błąd teraz naprawiony) i PASS_TO_PASS (nic nie zepsute).
- Klasyfikator trudności w stylu GAIA oparty na głębokości dekompozycji pytania.

Uruchom:

```
python3 code/main.py
```

Wynik pokazuje wskaźnik rozwiązania na zadanie + na trudność i konkretyzuje reguły ewaluatora.

## Use It

- **SWE-bench Verified** dla agentów kodu. Zawsze podawaj wyniki Verified.
- **GAIA** dla agentów-generalistów. Używaj podziału prywatnego liderboardu.
- **AgentBench** do porównań w wielu środowiskach.
- **Niestandardowe ewaluacje** (Lekcja 30) dla rzeczywistego kształtu twojego produktu.

## Ship It

`outputs/skill-benchmark-harness.md` buduje szkielet w stylu SWE-bench dla dowolnej pary baza kodu–zadanie z bramkowaniem FAIL_TO_PASS / PASS_TO_PASS.

## Ćwiczenia

1. Przenieś zabawkowy szkielet, aby działał na prawdziwym repozytorium (wybierz jedno ze swoich). Napisz 3 testy FAIL_TO_PASS dla znanych błędów.
2. Dodaj metrykę liczenia kroków. Na twoich 3 zadaniach, ile kroków agenta na rozwiązanie?
3. Przeczytaj artykuł SWE-bench+. Zaimplementuj sprawdzanie wycieku rozwiązania (dopasuj wzór tekstu zgłoszenia do diffa).
4. Pobierz pytanie GAIA z publicznego podziału. Prześledź, co zrobiłby agent klasy GPT-4. Jakich narzędzi potrzebuje?
5. Przeczytaj podział na środowiska AgentBench. Które środowisko odpowiada powierzchni twojego produktu? Jak wygląda tam „SOTA"?

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| SWE-bench | „Benchmark agentów kodu" | 2294 zgłoszeń GitHub; łatka musi odwrócić testy FAIL_TO_PASS |
| SWE-bench Verified | „Czysty SWE-bench" | 500 ręcznie wyselekcjonowanych zadań, OpenAI |
| FAIL_TO_PASS | „Bramka naprawy" | Testy wcześniej niezaliczone, które muszą być zaliczone po łatce |
| PASS_TO_PASS | „Bramka bez regresji" | Testy, które były zaliczone i nadal muszą być zaliczone |
| GAIA | „Benchmark generalisty" | 466 pytań łatwych dla ludzi / trudnych dla AI, wielonarzędziowych |
| AgentBench | „Benchmark wielośrodowiskowy" | 8 środowisk; długi horyzont, wieloturowe |
| Kontaminacja | „Wyciek zbioru treningowego" | Zadania benchmarkowe obecne w treningu modelu |
| SWE-bench+ | „Audyt kontaminacji" | 32,67% wycieku rozwiązań w udanych łatkach SWE-bench |

## Dalsza lektura

- [Jimenez i in., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — oryginalny benchmark
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — wyselekcjonowany podzbiór
- [Mialon i in., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — benchmark generalisty
- [Liu i in., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — zestaw wielośrodowiskowy