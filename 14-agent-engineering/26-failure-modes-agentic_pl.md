# Tryby awarii: Dlaczego agenci się psują

> MASFT (Berkeley, 2025) kataloguje 14 trybów awarii wieloagentowych w 3 kategoriach. Taksonomia Microsoftu dokumentuje, jak istniejące awarie AI nasilają się w środowiskach agentowych. Dane terenowe z branży zbiegają się do pięciu powtarzających się trybów: halucynacyjne akcje, rozszerzanie zakresu, kaskadowe błędy, utrata kontekstu, niewłaściwe użycie narzędzi.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- Wymień trzy kategorie awarii MASFT i co najmniej cztery konkretne tryby w każdej.
- Wyjaśnij, dlaczego awarie agentowe amplifikują istniejące tryby awarii AI (uprzedzenia, halucynacje).
- Opisz pięć powtarzających się w branży trybów i ich łagodzenie.
- Zaimplementuj detektor w stdlib, który oznacza ślady agentów etykietami trybów awarii.

## Problem

Zespoły wdrażają agentów, którzy działają na 90% śladów. 10% awarii to nie przypadkowy szum — mieszczą się w niewielkiej liczbie powtarzających się kategorii. Gdy potrafisz je nazwać, możesz je monitorować i naprawiać.

## Koncepcja

### MASFT (Berkeley, arXiv:2503.13657)

Taksonomia awarii systemów wieloagentowych. 14 trybów awarii zgrupowanych w 3 kategorie. Cohen's Kappa między oceniającymi 0.88 — kategorie są wiarygodnie rozróżnialne.

Główne twierdzenie: awarie to fundamentalne wady projektowe systemów wieloagentowych, a nie ograniczenia LLM do naprawienia lepszymi modelami bazowymi.

### Taksonomia Microsoftu trybów awarii w agentowych systemach AI

- Istniejące awarie AI (uprzedzenia, halucynacje, wycieki danych) amplifikują się w środowiskach agentowych.
- Nowe awarie wynikają z autonomii: niezamierzone działanie na skalę, niewłaściwe użycie narzędzi, dryf misji.
- Biała księga to rejestr ryzyka dla produktów agentowych.

### Charakteryzowanie błędów w agentowym AI (arXiv:2603.06847)

- Awarie wynikają z orkiestracji, ewolucji stanu wewnętrznego i interakcji ze środowiskiem.
- Nie tylko "zły kod" lub "złe wyjście modelu."

### Przegląd halucynacji agentów LLM (arXiv:2509.18970)

Dwie główne manifestacje:

1. **Odchylenie od instrukcji** — agent nie podąża za promptem systemowym.
2. **Niewłaściwe użycie kontekstu długodystansowego** — agent zapomina lub błędnie stosuje kontekst z wcześniejszych tur.

Błędy pod-intencji: Pominięcie (opuszczony krok), Redundancja (powtórzony krok), Nieporządek (kroki w złej kolejności).

### Pięć powtarzających się w branży trybów

Analizy terenowe Arize, Galileo, NimbleBrain 2024-2026 zbiegają się do:

1. **Halucynacyjne akcje.** Agent wywołuje narzędzie, które nie istnieje, lub fabrykuje argumenty.
2. **Rozszerzanie zakresu.** Agent rozszerza zadanie poza prośbę użytkownika (tworzy dodatkowe PR, wysyła dodatkowe e-maile).
3. **Kaskadowe błędy.** Jedno błędne wywołanie wyzwala efekty dalsze. Halucynacja widma SKU wyzwala cztery wywołania API — incydent wielosystemowy.
4. **Utrata kontekstu.** Zadania długoterminowe zapominają o ograniczeniach z wczesnych tur.
5. **Niewłaściwe użycie narzędzi.** Wywołuje właściwe narzędzie z błędnymi argumentami lub całkowicie złe narzędzie.

Kaskadowość jest zabójcą. Agenci nie potrafią odróżnić "nie udało mi się" od "zadanie jest niemożliwe" i często halucynują komunikat sukcesu na błędach 400, aby zamknąć pętlę.

### Łagodzenie: bramki na każdym kroku

Zautomatyzowane bramki weryfikacyjne na każdym kroku łańcucha rozumowania, sprawdzające ugruntowanie faktów względem stanu środowiska. Konkretnie:

- Klasyfikator bezpieczeństwa na krok (Lekcja 21).
- Walidacja argumentów wywołania narzędzia (Lekcja 06).
- Krzyżowe sprawdzanie pobranych treści względem znanych faktów (Lekcja 05, CRITIC).
- Wykrywanie halucynacji sukcesu przez ponowne sondowanie stanu (czy plik został faktycznie utworzony?).

### Gdzie monitorowanie awarii zawodzi

- **Oznaczanie tylko awarii.** Większość awarii agentów produkuje wyjście wyglądające na poprawne. Potrzebne są kontrole na poziomie treści.
- **Brak punktu odniesienia.** Wykrywanie dryfu potrzebuje ostatniego znanego dobrego stanu; bez niego nie możesz stwierdzić "to się pogarsza."
- **Nadmierne alarmowanie.** Każda awaria wywołuje powiadomienie. Grupuj i ograniczaj częstotliwość.

## Build It

`code/main.py` implementuje znacznik trybów awarii w stdlib:

- Syntetyczny zestaw śladów obejmujący pięć trybów.
- Funkcje detektorów na tryb (wzorce sygnatur w wywołaniach narzędzi, wyjściach, powtarzających się akcjach).
- Znacznik, który etykietuje każdy ślad i raportuje rozkład trybów.

Uruchom:

```
python3 code/main.py
```

Wynik: etykiety na ślad + zagregowany rozkład, tania reprodukcja tego, co pokazuje klastrowanie śladów Phoenix.

## Use It

- **Phoenix** dla klastrowania dryfu produkcyjnego (Lekcja 24).
- **Langfuse** dla odtwarzania sesji + adnotacji.
- **Własne** dla sygnatur specyficznych dla domeny, których twoja platforma obserwowalności nie wykryje.

## Ship It

`outputs/skill-failure-detector.md` generuje detektory trybów awarii dostosowane do twojej domeny, podłączone do magazynu śladów.

## Exercises

1. Dodaj detektor "halucynacji sukcesu": agent zwraca sukces, ale stan docelowy jest niezmieniony.
2. Oznacz 100 prawdziwych śladów z produktu, który zbudowałeś. Który tryb dominuje? Jaki jest koszt jego naprawy?
3. Zaimplementuj metrykę "promień kaskady": mając awarię na kroku N, ile kroków dalszych została dotknięta?
4. Przeczytaj 14 trybów awarii MASFT. Wybierz trzy, które dotyczą twojego produktu. Napisz detektory.
5. Podłącz jeden detektor do zadania CI: nieudana kompilacja, jeśli >=5% śladów oznacza tryb.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| MASFT | "Taksonomia awarii wieloagentowych" | 14-trybowa kategoryzacja Berkeley |
| Cascading error | "Błąd kaskadowy" | Jeden wczesny błąd propaguje się przez N kroków |
| Context loss | "Zapomniał ograniczenia" | Długa tura gubi fakty z wczesnych tur |
| Tool misuse | "Złe narzędzie / złe argumenty" | Poprawne wywołanie, błędne użycie |
| Success hallucination | "Sfałszowane zakończenie" | Agent twierdzi sukces na 400; stan niezmieniony |
| Scope creep | "Przekroczenie zakresu" | Agent robi więcej niż proszono |
| Instruction-following deviation | "Nieposłuszeństwo" | Ignoruje prompt systemowy lub ograniczenie użytkownika |
| Sub-intention errors | "Błędy planu" | Pominięcie, redundancja, nieporządek w wykonaniu planu |

## Further Reading

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 trybów awarii, 3 kategorie
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — rejestr ryzyka
- [Arize Phoenix](https://docs.arize.com/phoenix) — klastrowanie dryfu w praktyce
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — kiedy prostsze wzorce całkowicie unikają trybów