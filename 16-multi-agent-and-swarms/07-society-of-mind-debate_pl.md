# Społeczeństwo Umysłu i Debaty Wieloagentowe

> Założenie Minsky'ego z 1986 roku — inteligencja jako społeczeństwo specjalistów — jest odkrywane na nowo co dekadę. W 2023 Du i in. przekształcili je w konkretny algorytm: wiele instancji LLM proponuje odpowiedzi, czyta nawzajem swoje odpowiedzi, krytykuje i aktualizuje. Przez N rund zbiegają się do konsensusu, który bije zero-shot CoT i refleksję w sześciu zadaniach wnioskowania i faktyczności. Liczą się dwa odkrycia: zarówno **wielu agentów**, jak i **wiele rund** przyczyniają się niezależnie. Społeczeństwo bije monolog pojedynczego agenta; wymiana wielorundowa bije jednorazowe głosowanie.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Problem

Samo-zgodność — próbkuj jeden model wiele razy i weź odpowiedź większościową — to najtańsza poprawa wnioskowania, jaką możesz dołożyć. Działa, ale szybko nasyca się. Możesz podwoić liczbę próbek i nie zobaczyć kolejnego znaczącego skoku.

Debata przełamuje nasycenie. Zamiast N niezależnych próbek z jednego modelu, N agentów czyta nawzajem swoje rozumowanie i rewiduje. Korelacja między próbkami spada (nie są już i.i.d.), a punkt zbieżności jest często poprawny tam, gdzie głosowanie i.i.d. było pewne, a błędne.

## Concept

### Algorytm Du i in. 2023

Z arXiv:2305.14325 (ICML 2024):

1. Każdy z N agentów produkuje wstępną odpowiedź na pytanie.
2. Dla rundy r = 2..R: każdemu agentowi pokazywane są odpowiedzi innych agentów z rundy r-1 i proszony o „biorąc je pod uwagę, podaj zaktualizowaną odpowiedź."
3. Po R rundach, głosowanie większościowe na końcowe odpowiedzi.

Artykuł testuje na MMLU, GSM8K, biografiach, MATH i benchmarkach faktyczności. Debata konsekwentnie bije CoT i Self-Reflection.

### Dwa niezależne pokrętła

Ablacje z tego samego artykułu:

- **Sama liczba agentów** (1 runda, głosowanie większościowe N) bije pojedynczego agenta w większości zadań, ale osiąga plateau.
- **Sama liczba rund** (1 agent widzący własne wcześniejsze rozumowanie) ledwo pomaga — znana słabość refleksji.
- **Oba razem** produkuje duże skoki. Wielorundowa wymiana między wieloma agentami napędza zysk.

### Dlaczego to działa

Dwa mechanizmy:

1. **Ekspozycja na niezgodę.** Gdy agent widzi łańcuch rozumowania innego agenta z innym wnioskiem, musi go albo uzasadnić, albo zaktualizować. Tak czy inaczej, kontekst dla rundy r+1 jest bogatszy niż dla rundy r.
2. **Redukcja skorelowanych błędów.** W samo-zgodności wszystkie próbki pochodzą z tego samego modelu, więc błędy korelują — uśredniasz w pewną, ale błędną odpowiedź. Różne modele lub różne ziarna dekorrelują. Różne *przedyskutowane poglądy* dekorrelują jeszcze bardziej.

### Heterogeniczna debata

A-HMAD i powiązane kontynuacje używają *różnych modeli bazowych* dla różnych agentów. Debatowanie Llama + Claude + GPT redukuje załamanie monokultury (Lekcja 26), ponieważ skorelowane błędy jednej rodziny modeli nie są współdzielone przez inne.

Wada: słaby model uczestniczący w debacie może przeciągnąć konsensus w stronę swojej błędnej odpowiedzi (zobacz „Should we be going MAD?", arXiv:2311.17371).

### NLSOM — rozszerzenie 129-agentowe

Zhuge i in. („Mindstorms in Natural Language-Based Societies of Mind", arXiv:2305.17066) przeskalowali ten pomysł do społeczeństw liczących 129 członków. Wynik: specjalizacja i samoorganizacja wyłaniają się ze skalą, a system przewyższa pojedynczego agenta w zadaniach takich jak wizualne odpowiadanie na pytania.

### Tryby awarii

- **Kaskada służalczości.** Wszyscy agenci deferują do agenta, który brzmi najbardziej pewnie. Debata zapada się do najgłośniejszego głosu. Promptowanie ról antagonistycznych („jeden agent musi argumentować przeciwne stanowisko") pomaga.
- **Dryf tematu.** Debaty na wiele rund dryfują od oryginalnego pytania. Mitygacja: ponownie wstrzykuj pytanie każdej rundy.
- **Eksplozja obliczeniowa.** N agentów × R rund = N·R wywołań LLM, każde z rosnącym kontekstem. 5-agentowa, 5-rundowa debata to 25 wywołań z rosnącym kontekstem. Koszt na pytanie może przekroczyć 10× pojedyncze wywołanie CoT.

## Build It

`code/main.py` uruchamia 3-agentową × 3-rundową debatę na pytaniu matematycznym, gdzie każdy agent zaczyna z inną (potencjalnie błędną) odpowiedzią. Agenci są skryptowani — każdy „aktualizuje" przez uśrednianie odpowiedzi sąsiadów ważone skryptowanym poziomem ufności. Zbieżność jest widoczna w dzienniku runda po rundzie.

Demo pokazuje dwa kluczowe efekty:

- Pojedyncza runda wymiany przesuwa agentów bliżej poprawnej odpowiedzi.
- Dodatkowe rundy po rundzie 2 wykazują malejące zyski (zgodne z plateau Du i in.).

Uruchom:

```
python3 code/main.py
```

## Use It

`outputs/skill-debate-configurator.md` konfiguruje debatę dla nowego zadania: liczba agentów, liczba rund, heterogeniczność (ten sam model vs mieszany), przydział ról (symetryczny vs jeden-adwersarz). Szacuje również koszt tokenów przed uruchomieniem.

## Ship It

Jeśli wdrażasz debatę:

- **Ogranicz rundy do 3.** Du i in. pokazują, że 3 rundy przechwytują większość zysku. Więcej to koszt, nie jakość.
- **Ogranicz agentów do 5.** Powyżej 5, rozrost kontekstu i koszt dominują.
- **Heterogenicznie domyślnie.** Co najmniej dwa różne modele bazowe w puli.
- **Miejsce adwersarza.** Jeden agent promptowany, aby się nie zgadzać niezależnie. Przełamuje służalczość.
- **Rejestruj każdą rundę.** Systemy debat, które ukrywają pośrednie rundy, nie mogą być debugowane ani audytowane.

## Exercises

1. Uruchom `code/main.py`, a następnie ustaw liczbę rund na 5 i obserwuj malejące zyski. W której rundzie dodatkowa zbieżność się zatrzymuje?
2. Dodaj czwartego agenta z rolą adwersarza: zawsze nie zgadzaj się z bieżącą większością. Czy to przełamuje czy pogarsza zbieżność?
3. Wykreśl (wydrukuj) wynik zgodności na rundę (ułamek agentów na odpowiedzi większościowej). Kiedy osiąga 1.0 i czy to jest równoważne z „poprawną"?
4. Przeczytaj sekcję ablacji Du i in. 4. Odtwórz wynik „tylko agenci" vs „tylko rundy" vs „oba" używając tego kodu.
5. Przeczytaj „Should we be going MAD?" (arXiv:2311.17371) i wymień dwa warianty debaty poza round-robin — np. prowadzona przez sędziego, łańcuch debaty, adwersarz.

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Społeczeństwo Umysłu | „Pomysł Minsky'ego" | Inteligencja jako współdziałający specjaliści; ramy z 1986 roku obecnie zoperacjonalizowane przez debatę LLM. |
| Debaty wieloagentowe | „Agenci się kłócą" | N agentów proponuje, krytykuje nawzajem, rewiduje przez R rund, głosowanie większościowe. |
| Konsensus | „Zgadzają się" | Nie epistemiczna prawda — tylko ułamek-na-odpowiedzi-większościowej. Może być pewnie błędny. |
| Rundy | „Kroki wymiany" | Jedna runda = każdy agent czyta innych i aktualizuje raz. |
| Heterogeniczna debata | „Mieszaj rodziny modeli" | Używanie różnych modeli bazowych do dekorrelacji błędów. |
| Kaskada służalczości | „Wszyscy zgadzają się z głośnym" | Awaria debaty, w której agenci deferują do najbardziej pewnego agenta niezależnie od poprawności. |
| NLSOM | „129-agentowe społeczeństwo" | Natural-language society of mind; skalowana wersja Zhuge i in. |
| Skorelowany błąd | „Ten sam model, ten sam błąd" | Dlaczego samo-zgodność nasyca się; debata na różnych poglądach dekorreluje. |

## Further Reading

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — artykuł referencyjny, ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — 129-agentowy NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — benchmarki wariantów debaty
- [Debate project page](https://composable-models.github.io/llm_debate/) — kod Du i in., dema i szczegóły ablacji