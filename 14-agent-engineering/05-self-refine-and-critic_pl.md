# Self-Refine i CRITIC: Iteracyjna poprawa wyjścia

> Self-Refine (Madaan et al., 2023) używa jednego LLM w trzech rolach — generuj, opinia, udoskonalaj — w pętli. Średni zysk: +20 absolutne na 7 zadaniach. CRITIC (Gou et al., 2023) utwardza krok opinii, kierując weryfikację przez zewnętrzne narzędzia. W 2026 ten wzorzec jest dostarczany w każdym frameworku jako "evaluator-optimizer" (Anthropic) lub pętla guardrail (OpenAI Agents SDK).

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 minutes

## Learning Objectives

- Podaj trzy prompty Self-Refine (generuj, opinia, udoskonalaj) i wyjaśnij, dlaczego historia ma znaczenie dla promptu udoskonalania.
- Wyjaśnij krytyczne spostrzeżenie CRITIC: LLMy są zawodne w samoweryfikacji bez zewnętrznego ugruntowania.
- Zaimplementuj w stdlib pętlę Self-Refine z historią i opcjonalnym zewnętrznym weryfikatorem.
- Mapuj ten wzorzec na workflow "evaluator-optimizer" Anthropic i guardrails wyjścia OpenAI Agents SDK.

## The Problem

Agent produkuje odpowiedź, która jest prawie poprawna. Może wiersz kodu ma błąd składni. Może podsumowanie jest za długie. Może plan pomija przypadek brzegowy. To, czego chcesz, to: agent krytykuje własne wyjście, a potem je naprawia.

Self-Refine pokazuje, że to działa z pojedynczym modelem, bez danych treningowych, bez RL. Ale jest haczyk: LLMy są słabe w samoweryfikacji twardych faktów. CRITIC nazywa poprawkę — kieruj krok weryfikacji przez zewnętrzne narzędzia (wyszukiwanie, interpreter kodu, kalkulator, runner testów).

Razem te dwie prace definiują domyślne ustawienie 2026 dla iteracyjnej poprawy: generuj, weryfikuj (zewnętrznie, gdy możliwe), udoskonalaj, zatrzymaj, gdy weryfikator przejdzie.

## The Concept

### Self-Refine (Madaan et al., NeurIPS 2023)

Jeden LLM, trzy role:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop, gdy feedback mówi "brak problemów" lub budżet wyczerpany.
```

Kluczowy szczegół: `refine` widzi pełną historię — wszystkie poprzednie wyjścia i krytyki — więc nie powtarza błędów. Praca abluje to: usuń historię, a jakość gwałtownie spada.

Nagłówek: +20 absolutnej poprawy średnio na 7 zadaniach (matematyka, kod, akronimy, dialog) w tym GPT-4. Żadnego treningu, żadnych zewnętrznych narzędzi, pojedynczy model.

### CRITIC (Gou et al., arXiv:2305.11738, v4 luty 2024)

Słabość Self-Refine: krok opinii to LLM oceniający siebie. W przypadku twierdzeń faktycznych jest to zawodne (halucynacja często wygląda przekonująco dla modelu, który ją wyprodukował). CRITIC zastępuje `feedback(task, output)` przez `verify(task, output, tools)`, gdzie `tools` zawiera:

- Wyszukiwarkę dla twierdzeń faktycznych.
- Interpreter kodu dla poprawności kodu.
- Kalkulator dla arytmetyki.
- Weryfikatory domenowe (testy jednostkowe, sprawdzacze typów, lintery).

Weryfikator produkuje ustrukturyzowaną krytykę opartą na wynikach narzędzi. Udoskonalacz następnie warunkuje się na tej krytyce.

Nagłówek: CRITIC przewyższa Self-Refine w zadaniach faktycznych, ponieważ krytyka jest ugruntowana. W zadaniach bez zewnętrznych weryfikatorów (pisanie kreatywne, formatowanie), CRITIC sprowadza się do Self-Refine.

### Warunek zatrzymania

Dwa popularne kształty:

1. **Weryfikator przechodzi.** Zewnętrzny test zwraca sukces. Preferowane, gdy dostępne (testy jednostkowe, sprawdzacz typów, asercja guardrail).
2. **Brak opinii.** Model mówi "wyjście jest w porządku." Tańsze, ale zawodne; połącz z limitem maksymalnej liczby iteracji.

Domyślne ustawienie 2026: połącz je. "Zatrzymaj, jeśli weryfikator przejdzie LUB model mówi OK ORAZ iteracje >= 2 LUB iteracje >= max_iterations."

### Evaluator-Optimizer (Anthropic, 2024)

Post Anthropic z grudnia 2024 nazywa to jednym z pięciu wzorców workflow. Dwie role:

- Evaluator: ocenia wyjście i produkuje krytykę.
- Optimizer: rewiduje wyjście na podstawie krytyki.

Pętla, aż evaluator przejdzie. To jest Self-Refine/CRITIC w ujęciu Anthropic. Kluczowy szczegół inżynieryjny dodany przez Anthropic: prompty evaluatora i optimizera powinny być znacząco różne, aby model nie tylko przybijał pieczątkę.

### Guardrails wyjścia OpenAI Agents SDK

OpenAI Agents SDK dostarcza ten wzorzec jako "output guardrails." Guardrail to walidator, który uruchamia się na końcowym wyjściu agenta. Jeśli guardrail zadziała (podnosi `OutputGuardrailTripwireTriggered`), wyjście jest odrzucane i agent może spróbować ponownie. Guardrails mogą wywoływać narzędzia (styl CRITIC) lub być czystymi funkcjami (styl Self-Refine).

### Pułapki 2026

- **Pętle pieczątkowe.** Ten sam model wykonujący generację i krytykę w tym samym stylu promptu zbiega do "wygląda dobrze." Używaj strukturalnie różnych promptów lub mniejszego, tańszego modelu do krytyki.
- **Nadmierne udoskonalanie.** Każde przejście udoskonalania dodaje opóźnienie i tokeny. Budżet 1-3 przejść; potem eskaluj do przeglądu ludzkiego.
- **CRITIC dla trywialnych zadań.** Jeśli nie ma zewnętrznego weryfikatora, CRITIC degeneruje się do Self-Refine; nie płać opóźnienia za atrapę weryfikatora.

## Build It

`code/main.py` implementuje Self-Refine i CRITIC na zabawkowym zadaniu: wyprodukuj krótką listę punktowaną na dany temat. Weryfikator sprawdza format (3 punkty, każdy poniżej 60 znaków). CRITIC dodaje zewnętrzny "weryfikator faktów", który karze znane halucynacje.

Komponenty:

- `generate` — skryptowany producent.
- `feedback` — samokrytyka w stylu LLM.
- `verify_external` — ugruntowany weryfikator w stylu CRITIC.
- `refine` — przepisuje wyjście na podstawie historii.
- Warunek zatrzymania — weryfikator przechodzi lub maksymalnie 4 iteracje.

Uruchom:

```
python3 code/main.py
```

Porównaj uruchomienia Self-Refine vs CRITIC. CRITIC łapie błąd faktyczny, który Self-Refine przeoczył, ponieważ zewnętrzny weryfikator ma ugruntowanie, którego samokrytyk nie ma.

## Use It

Evaluator-optimizer Anthropic to ten wzorzec w języku przyjaznym Claude. Guardrails wyjścia OpenAI Agents SDK mają kształt CRITIC (guardrails mogą wywoływać narzędzia). LangGraph dostarcza węzeł refleksji, który wygląda jak Self-Refine. Google Gemini 2.5 Computer Use dodaje evaluatora bezpieczeństwa na każdy krok, który jest wariantem CRITIC: każda akcja jest weryfikowana przed zatwierdzeniem.

## Ship It

`outputs/skill-refine-loop.md` konfiguruje pętlę evaluator-optimizer, biorąc pod uwagę kształt zadania, dostępność weryfikatora i budżet iteracji. Emituje prompty dla generatora, evaluatora/weryfikatora i optimizera, plus politykę zatrzymania.

## Exercises

1. Uruchom zabawkę z max_iterations=1. Czy CRITIC wciąż pomaga?
2. Zastąp zewnętrzny weryfikator zaszumionym (losowe 30% fałszywych alarmów). Co robi pętla? To jest rzeczywistość 2026 większości stosów guardrail.
3. Zaimplementuj wariant "generator-krytyk na różnych modelach": duży model generuje, mały krytykuje. Czy bije ten sam model?
4. Przeczytaj CRITIC Sekcję 3 (arXiv:2305.11738 v4). Wymień trzy kategorie narzędzi weryfikacji i podaj przykład dla każdej.
5. Mapuj `output_guardrails` OpenAI Agents SDK na rolę weryfikatora CRITIC. Co SDK robi źle, a co dobrze?

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|------|----------------|------------------------|
| Self-Refine | "LLM, który się naprawia" | Pętla generuj -> opinia -> udoskonalaj w jednym modelu, z historią |
| CRITIC | "Weryfikacja oparta na narzędziach" | Zastąp opinię zewnętrznym weryfikatorem (wyszukiwanie, kod, kalkulator, testy) |
| Evaluator-Optimizer | "Wzorzec workflow Anthropic" | Dwie role — evaluator ocenia, optimizer rewiduje — pętla do zbieżności |
| Output guardrail | "Kontrola post-hoc" | Walidator OpenAI Agents SDK, który uruchamia się po wyprodukowaniu wyjścia przez agenta |
| Krok weryfikacji | "Faza krytyki" | Decyzja nośna: ugruntowana lub samooceniana |
| Historia udoskonalania | "Co model już próbował" | Poprzednie wyjścia + krytyki dołączane do promptu udoskonalania; usuń, a jakość się załamuje |
| Pętla pieczątkowa | "Awaria samozgodności" | Krytyka tym samym promptem zwraca "wygląda dobrze"; napraw strukturalnie różnymi promptami |
| Warunek zatrzymania | "Test zbieżności" | Weryfikator przechodzi LUB brak opinii ORAZ limit iteracji; nigdy pojedynczy warunek |

## Further Reading

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — kanoniczna praca
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — weryfikacja oparta na narzędziach
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — wzorzec workflow evaluator-optimizer
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — output guardrails jako weryfikatory w kształcie CRITIC

(End of file - total 140 lines)