# GroupChat i Wybór Mówcy

> AutoGen GroupChat i AG2 GroupChat współdzielą jedną konwersację pomiędzy N agentami; funkcja selektora (LLM, round-robin lub własna) wybiera, kto mówi następny. To archetyp emergentnej konwersacji wieloagentowej — agenci nie znają swojej roli w statycznym grafie, po prostu reagują na współdzieloną pulę. Semantyka GroupChat z AutoGen v0.2 została zachowana w forku AG2; AutoGen v0.4 przepisało ją jako zdarzeniowy model aktorowy. Microsoft przeniósł AutoGen w tryb utrzymania w lutym 2026 i połączył je z Semantic Kernel w Microsoft Agent Framework (RC luty 2026). Prymityw GroupChat przetrwał w obu ścieżkach — AG2 i Microsoft Agent Framework — naucz się go raz, używaj wszędzie.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Problem

Statyczne grafy (LangGraph) są świetne, gdy przepływ pracy jest znany. Prawdziwe rozmowy nie są statyczne: czasami programista pyta recenzenta, czasami badacza, czasami pisarza. Hardkodowanie każdego możliwego przekazania powoduje eksplozję krawędzi. Potrzebujesz *agentów reagujących na współdzieloną pulę*, z pewną funkcją decydującą, kto mówi dalej.

To właśnie robi AutoGen GroupChat.

## Koncepcja

### Kształt

```
              ┌─── współdzielona pula ────┐
              │   m1  m2  m3  ...        │
              └─────────┬────────────────┘
                        │ (każdy czyta wszystko)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A Agent B  Agent C  Agent D  Selektor
                                           │
                                           ▼
                                  "następny mówca = C"
```

Każdy agent widzi każdą wiadomość. Funkcja selektora jest wywoływana w każdej turze, aby wybrać, kto mówi następny.

### Trzy smaki selektora

**Round-robin.** Stały cykl. Deterministyczny. Skaluje się liniowo z N, ale ignoruje kontekst — programista dostaje swoją kolej, nawet gdy temat dotyczy przeglądu prawnego.

**Wybór przez LLM.** Wywołanie LLM, który czyta ostatnie wiadomości i zwraca najlepszego następnego mówcę. Świadomy kontekstu, ale wolny: każda tura dodaje wywołanie LLM. Domyślne w AutoGen.

**Własny.** Funkcja Pythona z dowolną logiką. Typowe: wybór przez LLM z regułami rezerwowymi (np. "zawsze dawaj głos weryfikatorowi po programiście").

### API ConversableAgent

```
agent = ConversableAgent(
    name="coder",
    system_message="Piszesz w Pythonie.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` przechowuje selektor. Gdy agent zakończy turę, menedżer wywołuje selektor, który zwraca następnego agenta. Pętla trwa aż do spełnienia warunku zakończenia.

### Zakończenie

Trzy popularne wzorce:

- **Maksymalna liczba rund.** Twardy limit całkowitej liczby tur.
- **Token "TERMINATE".** Agenci mogą emitować wiadomość wartownika; menedżer zatrzymuje się, gdy taka pojawi się.
- **Sprawdzenie osiągnięcia celu.** Lekki weryfikator uruchamiany w każdej turze zatrzymuje czat po osiągnięciu celu.

### Podział AutoGen → AG2 i połączenie z Microsoft Agent Framework

Na początku 2025 roku Microsoft rozpoczął znaczące przepisanie AutoGen (v0.4) wokół zdarzeniowego modelu aktorowego. Społeczność sfolkowała semantykę GroupChat z AutoGen v0.2 jako AG2, zachowując API, które wcześni użytkownicy zintegrowali.

W lutym 2026 Microsoft ogłosił, że AutoGen przechodzi w tryb utrzymania, a zdarzeniowy model aktorowy zostaje połączony z **Microsoft Agent Framework** (RC luty 2026, obecnie połączony z Semantic Kernel). Koncepcja GroupChat przetrwała w obu ścieżkach; szczegóły implementacji różnią się. AG2 jest preferowanym upstreamem dla kodu zgodnego z v0.2.

### Kiedy GroupChat pasuje

- **Emergentne konwersacje.** Nie chcesz wstępnie okablować każdego możliwego następnego mówcy.
- **Zadania z mieszaniem ról.** Programista pyta badacza, badacz pyta archiwistę, archiwista pyta programistę z powrotem. Przepływ nie jest DAG.
- **Eksploracyjne rozwiązywanie problemów.** Myśl "burza mózgów", a nie "linia montażowa".

### Kiedy zawodzi

- **Ścisły determinizm.** Selektor LLM może być niespójny. Ten sam prompt, różne uruchomienia, różni następni mówcy.
- **Kaskady pochlebstw.** Agenci dostosowują się do tego, kto mówił najbardziej pewnie. Przeciwdziałaj jawnym promptem.
- **Wzdęcie kontekstu.** Każdy agent czyta każdą wiadomość; po 10 turach kontekst jest ogromny. Używaj projekcji (Lekcja 15), aby zakres widoku.
- **Gorący mówcy.** Jeden agent dominuje rozmowę, ponieważ selektor faworyzuje jego specjalizacje. Wprowadź balans mówców jako cechę selektora.

### GroupChat a nadzorca

Te same prymitywy, różne domyślne:

- Nadzorca: jeden agent planuje, a inni wykonują. Selektor to "zapytaj planistę, co robić."
- GroupChat: wszyscy agenci są równorzędni; selektor to funkcja na współdzielonej puli.

Oba używają czterech prymitywów z Lekcji 04. GroupChat domyślnie używa orkiestracji wybieranej przez LLM i stanu współdzielonego pełnej puli.

## Zbuduj To

`code/main.py` implementuje GroupChat od podstaw w stdlib. Trzej agenci (programista, recenzent, menedżer), warianty round-robin i wybierany przez LLM, oraz zakończenie na token `TERMINATE`.

Demo wyświetla transkrypt konwersacji plus ślad decyzji selektora dla obu wariantów.

Uruchom:

```
python3 code/main.py
```

## Użyj Tego

`outputs/skill-groupchat-selector.md` konfiguruje selektor GroupChat dla danego zadania — round-robin vs wybierany przez LLM vs własny, oraz jakie dane wejściowe selektora (ostatnie wiadomości, specjalizacje agentów, liczniki tur) użyć.

## Wdróż To

Lista kontrolna:

- **Maksymalna liczba rund.** Zawsze. 10-20 dla typowych zadań.
- **Metryka balansu mówców.** Śledź tury na agenta; alertuj, gdy nierównowaga przekroczy próg.
- **Token zakończenia.** `TERMINATE` lub dedykowany agent weryfikator.
- **Projekcja lub zakresowa pamięć.** Po ~10 wiadomościach rozważ danie każdemu agentowi tylko zakresowego widoku, aby zapobiec wzdęciu kontekstu.
- **Logowanie selektora.** Dla wariantów wybieranych przez LLM, loguj zarówno dane wejściowe selektora, jak i jego wybór. W przeciwnym razie debugowanie jest niemożliwe.

## Ćwiczenia

1. Uruchom `code/main.py`. Porównaj konwersację w trybie round-robin vs wybieranym przez LLM. Który agent dominuje w każdym z nich?
2. Dodaj regułę "maksymalna-liczba-wypowiedzi-na-agenta" w selektorze. Jak wpływa na transkrypt?
3. Zaimplementuj zakończenie po osiągnięciu celu: zatrzymaj, gdy recenzent zwróci "zatwierdzono." Jak często wyzwala się przed limitem rund?
4. Przeczytaj stabilną dokumentację AutoGen na temat GroupChat (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html). Zidentyfikuj domyślny selektor używany przez `GroupChatManager`.
5. Przeczytaj repozytorium AG2 (https://github.com/ag2ai/ag2) i porównaj jego GroupChat v0.2 z wersją zdarzeniową v0.4. Jaką konkretną właściwość (przepustowość, odporność na awarie, komponowalność) dodaje v0.4?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|----------------|------------------------|
| GroupChat | "Agenci w jednym pokoju czatu" | Współdzielona pula wiadomości + funkcja selektora. Prymityw AutoGen / AG2. |
| Wybór mówcy | "Kto mówi następny" | Funkcja wybierająca następnego agenta. Round-robin, wybierany przez LLM lub własny. |
| GroupChatManager | "Gospodarz spotkania" | Komponent AutoGen, który posiada selektor i zapętla tury. |
| ConversableAgent | "Podstawowy agent" | Klasa bazowa AutoGen; agent, który może wysyłać i odbierać wiadomości. |
| Token zakończenia | "Słowo stop" | Ciąg wartownika (zwykle `TERMINATE`), który kończy czat. |
| Gorący mówca | "Jeden agent dominuje" | Tryb awarii, w którym selektor ciągle wybiera tego samego agenta. |
| Wzdęcie kontekstu | "Pula rośnie bez ograniczeń" | Każdy agent czyta każdą poprzednią wiadomość; kontekst rośnie z liczbą tur. |
| Projekcja | "Zakresowy widok" | Widok specyficzny dla roli we współdzielonej puli, zapobiegający wzdęciu kontekstu. |

## Dalsza Literatura

- [Dokumentacja GroupChat AutoGen](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — referencyjna implementacja
- [Repozytorium AG2](https://github.com/ag2ai/ag2) — kontynuacja społecznościowa AutoGen v0.2
- [Dokumentacja Microsoft Agent Framework](https://microsoft.github.io/agent-framework/) — połączony następca, RC luty 2026
- [Notatki wydania AutoGen v0.4](https://microsoft.github.io/autogen/stable/) — szczegóły przepisania na zdarzeniowy model aktorowy