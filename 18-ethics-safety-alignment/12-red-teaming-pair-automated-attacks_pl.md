# Red-Teaming: PAIR i Zautomatyzowane Ataki

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419). PAIR — Prompt Automatic Iterative Refinement — to kanoniczny zautomatyzowany jailbreak czarnoskrzynkowy. Atakujący LLM z promptem systemowym czerwonego zespołu iteracyjnie proponuje jailbreak dla docelowego LLM, gromadząc próby i odpowiedzi we własnej historii czatu jako informację zwrotną w kontekście. PAIR zazwyczaj succeeds w ciągu 20 zapytań, o rzędy wielkości bardziej wydajny niż GCG (tokenowe przeszukiwanie gradientowe Zou et al.) i bez potrzeby dostępu do białej skrzynki. PAIR jest teraz standardową linią bazową w JailbreakBench (arXiv:2404.01318) i HarmBench, obok GCG, AutoDAN, TAP i Persuasive Adversarial Prompt.

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time:** ~75 minutes

## Learning Objectives

- Opisz algorytm PAIR: prompt systemowy atakującego, iteracyjne udoskonalanie, informacja zwrotna w kontekście.
- Wyjaśnij, dlaczego PAIR jest ściśle bardziej wydajny niż GCG, gdy cel jest czarnoskrzynkowy.
- Wymień cztery inne linie bazowe zautomatyzowanych ataków (GCG, AutoDAN, TAP, PAP) i podaj jedną cechę wyróżniającą każdego.
- Opisz protokoły ewaluacji JailbreakBench i HarmBench oraz co oznacza „wskaźnik sukcesu ataku" (ASR) w każdym z nich.

## The Problem

Red-teaming był kiedyś czynnością manualną. Niewielka liczba eksperckich testerów konstruowała kontradyktoryjne prompt i śledziła, które działały. To nie skaluje się: wskaźnik sukcesu ataku potrzebuje statystycznej próbki, a cel zmienia się z każdym wydaniem modelu. PAIR operacjonalizuje red-teamming jako problem optymalizacyjny z czarnoskrzynkowym celem.

## The Concept

### PAIR algorithm

Wejścia:
- Docelowy LLM T (model, który atakujemy).
- Sędzia LLM J (ocenia, czy odpowiedź jest jailbreakiem).
- Atakujący LLM A (optymalizator czerwonego zespołu).
- Ciąg celu G: „odpowiedz na [szkodliwą instrukcję]."
- Budżet K (zwykle 20 zapytań).

Pętla, dla k in 1..K:
1. A otrzymuje prompt z celem G i historią par (prompt, odpowiedź) do tej pory.
2. A emituje nowy prompt p_k.
3. Wyślij p_k do T; otrzymaj odpowiedź r_k.
4. J ocenia (p_k, r_k) względem celu.
5. Jeśli wynik >= próg, zatrzymaj — jailbreak znaleziony.
6. W przeciwnym razie, dołącz (p_k, r_k) do historii A; kontynuuj.

Wynik empiryczny (NeurIPS 2023): >50% wskaźnika sukcesu ataku przeciwko GPT-3.5-turbo, Llama-2-7B-chat; średnia liczba zapytań do sukcesu w zakresie 10-20.

### Why PAIR is efficient

GCG (Zou et al. 2023) przeszukuje kontradyktoryjne sufiksy tokenowe przez gradient; wymaga dostępu do białej skrzynki i produkuje nieczytelne sufiksy. PAIR jest czarnoskrzynkowy i produkuje ataki w języku naturalnym, które przenoszą się między modelami. Informacja zwrotna w kontekście PAIR pozwala atakującemu uczyć się z każdego odrzucenia; GCG nie ma odpowiednika (każda nowa aktualizacja tokenu musi na nowo odkrywać poprzedni postęp).

### Related automated attacks

- **GCG (Zou et al. 2023, arXiv:2307.15043).** Tokenowe przeszukiwanie gradientowe dla kontradyktoryjnych sufiksów. Biała skrzynka, przenośne, produkuje nieczytelne ciągi znaków.
- **AutoDAN (Liu et al. 2023).** Ewolucyjne przeszukiwanie promptów, kierowane hierarchicznym celem.
- **TAP (Mehrotra et al. 2024).** Drzewo ataków z przycinaniem — rozwidla wiele rolloutów w stylu PAIR.
- **PAP (Zeng et al. 2024).** Persuasive Adversarial Prompts — koduje ludzkie techniki perswazji jako szablony promptów.

### JailbreakBench and HarmBench

Oba (2024) standaryzują ewaluację:

- JailbreakBench (arXiv:2404.01318). 100 szkodliwych zachowań w 10 kategoriach polityki OpenAI. Wskaźnik sukcesu ataku (ASR) jako główna metryka. Wymaga sędziego (GPT-4-turbo, Llama Guard lub StrongREJECT).
- HarmBench (Mazeika et al. 2024). 510 zachowań w 7 kategoriach, z semantycznymi i funkcjonalnymi testami szkodliwości. Porównuje 18 ataków przeciwko 33 modelom.

ASR jest zwykle raportowany przy ustalonym budżecie zapytań. Porównywanie ataków wymaga dopasowanych budżetów; 90% ASR przy 200 zapytaniach nie jest porównywalne z 85% ASR przy 20.

### Reason it matters for 2026 deployments

Każde laboratorium graniczne uruchamia teraz PAIR i TAP przeciwko modelom produkcyjnym przed wydaniem. Trajektorie ASR pojawiają się w kartach modeli (Lekcja 26) i załącznikach przypadków bezpieczeństwa (Lekcja 18). Atak nie jest egzotyczny — to standardowa infrastruktura.

### Where this fits in Phase 18

Lekcja 12 to podstawa zautomatyzowanych ataków. Lekcja 13 (Many-Shot Jailbreaking) to uzupełniający exploit długości. Lekcja 14 (ASCII Art / Visual) to atak kodowania. Lekcja 15 (Indirect Prompt Injection) to powierzchnia ataku produkcyjnego w 2026 roku. Lekcja 16 obejmuje odpowiedniki narzędzi obronnych (Llama Guard, Garak, PyRIT).

## Use It

`code/main.py` buduje zabawkową pętlę PAIR. Cel to udawany klasyfikator, który odrzuca „oczywiste" szkodliwe prompt (filtr słów kluczowych). Atakujący to regułowy udoskonalacz, który próbuje parafrazy, roli-play i kodowania. Sędzia ocenia odpowiedź. Obserwujesz, jak atakujący succeeds w ~5-15 iteracjach przeciwko filtrowi słów kluczowych i zawodzi przeciwko filtrowi semantycznemu.

## Ship It

Ta lekcja produkuje `outputs/skill-attack-audit.md`. Otrzymując raport ewaluacji czerwonego zespołu, audytuje: które ataki zostały uruchomione (PAIR, GCG, TAP, AutoDAN, PAP), przy jakim budżecie każdy, z jakim sędzią, na którym zestawie szkodliwych zachowań (JailbreakBench, HarmBench, wewnętrzny).

## Exercises

1. Uruchom `code/main.py`. Zmierz średnią liczbę zapytań do sukcesu dla trzech wbudowanych strategii atakującego. Wyjaśnij, które założenie obrony celu każda wykorzystuje.

2. Zaimplementuj czwartą strategię atakującego (np. tłumaczenie na inny język, kodowanie base64). Zgłoś nową średnią liczbę zapytań do sukcesu przeciwko celowi z filtrem słów kluczowych i celowi z filtrem semantycznym.

3. Przeczytaj Chao et al. 2023 Rysunek 5 (porównanie PAIR vs GCG). Opisz dwa scenariusze, w których GCG jest preferowany pomimo przewagi wydajnościowej PAIR.

4. JailbreakBench raportuje ASR przeciwko ustalonemu zestawowi celów. Zaprojektuj dodatkową metrykę, która mierzy różnorodność ataków (wariancję w udanych promptach). Wyjaśnij, dlaczego różnorodność ma znaczenie dla ewaluacji obrony.

5. TAP (Mehrotra 2024) rozszerza PAIR o rozwidlanie + przycinanie. Naszkicuj rozszerzenie w stylu TAP do `code/main.py` i opisz kompromis między kosztem obliczeniowym a wskaźnikiem sukcesu.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| PAIR | „zautomatyzowany jailbreak" | Prompt Automatic Iterative Refinement; pętla atakujący-LLM + sędzia-LLM |
| GCG | „gradientowy jailbreak" | Białoskrzynkowe tokenowe przeszukiwanie gradientowe dla kontradyktoryjnych sufiksów |
| Attack success rate (ASR) | „% jailbreaków przy k zapytaniach" | Główna metryka; musi być raportowana z budżetem zapytań i tożsamością sędziego |
| Judge LLM | „sędzia" | LLM, który ocenia, czy odpowiedź spełnia szkodliwy cel |
| JailbreakBench | „ewaluacja" | Standaryzowany zestaw szkodliwych zachowań z oznakowanymi kategoriami |
| HarmBench | „szerszy benchmark" | 510 zachowań, funkcjonalne + semantyczne testy szkodliwości |
| TAP | „drzewo ataków" | PAIR z rozwidlaniem + przycinaniem; lepszy ASR przy wyższym nakładzie obliczeniowym |

## Further Reading

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — artykuł PAIR, NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — artykuł GCG
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — standaryzowana ewaluacja
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — szersza ewaluacja