# Przekazania i Rutyny — Orkiestracja Bezstanowa

> Swarm od OpenAI (październik 2024) sprowadził orkiestrację wieloagentową do dwóch prymitywów: **rutyn** (instrukcje + narzędzia jako prompt systemowy) i **przekazań** (narzędzie zwracające innego Agenta). Brak maszyny stanów, brak DSL-a rozgałęzień — LLM routuje, wywołując odpowiednie narzędzie przekazania. OpenAI Agents SDK (marzec 2025) to produkcyjny następca. Sam Swarm pozostaje najczystszą koncepcyjną referencją — cały jego kod źródłowy mieści się w kilkuset liniach. Wzorzec jest wiralny, ponieważ powierzchnia API to mniej więcej "agent = prompt + narzędzia; przekazanie = funkcja zwracająca agenta." Ograniczenie: bezstanowy, więc pamięć jest problemem wywołującego.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Problem

Każdy framework wieloagentowy chce, abyś nauczył się jego DSL-a: węzły i krawędzie LangGraph, załogi i zadania CrewAI, GroupChat i menedżery AutoGen. DSL-e to prawdziwe abstrakcje, ale sprawiają, że wszystko wydaje się cięższe, niż być powinno.

Swarm idzie w przeciwnym kierunku: używa zdolności wywoływania narzędzi, którą model już ma. Przekazania stają się wywołaniami narzędzi. Orkiestratorem jest ten agent, który aktualnie prowadzi konwersację. Maszyna stanów jest ukryta w promptach systemowych agentów.

## Koncepcja

### Dwa prymitywy

**Rutyna.** Prompt systemowy definiujący rolę agenta i dostępne narzędzia. Myśl o tym jak o zakresowym zestawie instrukcji: "jesteś agentem triażowym; jeśli użytkownik pyta o zwroty, przekaż do agenta ds. zwrotów."

**Przekazanie.** Narzędzie, które agent może wywołać i które zwraca nowy obiekt Agenta. Środowisko Swarm wykrywa wartość zwracaną Agenta i przełącza aktywnego agenta na następną turę.

To cała abstrakcja.

```
def transfer_to_refunds():
    return refund_agent  # Swarm widzi zwrot Agent → przełącza aktywnego agenta

triage_agent = Agent(
    name="triage",
    instructions="Kieruj użytkownika do odpowiedniego specjalisty.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

Prompt systemowy agenta triażowego sprawia, że wybiera on odpowiednie przekazanie na podstawie wiadomości użytkownika. Wywoływanie narzędzi przez LLM wykonuje routing.

### Dlaczego jest wiralny

- **Małe API.** Dwie koncepcje do nauczenia.
- **Używa tego, co model już robi.** Wywoływanie narzędzi jest już produkcyjnej jakości u wszystkich dostawców.
- **Brak ciężaru maszyny stanów.** Nie opisujesz grafu; prompty agentów opisują, do kogo przekazują.

### Kompromis bezstanowości

Swarm jest jawnie bezstanowy między uruchomieniami. Framework przechowuje historię wiadomości podczas uruchomienia, ale niczego nie utrwala. Pamięć, ciągłość, długotrwałe zadania — wszystko to problem wywołującego.

W produkcji (OpenAI Agents SDK, marzec 2025) to była jedna z głównych rzeczy, które się zmieniły: SDK dodaje wbudowane zarządzanie sesjami, bariery ochronne i śledzenie, zachowując prymityw przekazania.

### Kiedy Swarm/przekazania pasują

- **Wzorce triażowe.** Agent pierwszej linii kieruje użytkownika do specjalisty.
- **Przekazania oparte na umiejętnościach.** "Jeśli zadanie wymaga kodu, wywołaj programistę; jeśli wymaga badań, wywołaj badacza."
- **Krótkie, ograniczone konwersacje.** Obsługa klienta, FAQ-do-zgłoszenia, proste przepływy pracy.

### Kiedy Swarm ma trudności

- **Długie sesje ze współdzieloną pamięcią.** Przekazania resetują stan konwersacji do promptu nowego agenta plus historii. Brak trwałego stanu między agentami bez pamięci zarządzanej przez wywołującego.
- **Równoległe wykonanie.** Przekazanie to jeden-na-raz — aktywny agent się przełącza. Równoległość wymaga, aby wywołujący orkiestrował wiele uruchomień Swarm.
- **Audyt i odtwarzanie.** Uruchomienia bezstanowe są trudne do dokładnego odtworzenia; wybór przekazania przez LLM nie jest deterministyczny.

### OpenAI Agents SDK (marzec 2025)

Produkcyjny następca dodaje:

- **Stan sesji.** Trwały wątek między uruchomieniami.
- **Bariery ochronne.** Haki walidacji wejścia/wyjścia.
- **Śledzenie.** Każde wywołanie narzędzia i przekazanie jest logowane.
- **Filtry przekazań.** Kontrola, jaki kontekst jest przenoszony przy przekazaniu.

Prymityw przekazania przetrwał; wokół niego dodano produkcyjną ergonomię.

### Swarm a GroupChat

Oba używają routingu sterowanego przez LLM, ale różnią się tym, **kto wybiera następny**:

- GroupChat: selektor (funkcja lub LLM) wybiera następnego mówcę z zewnątrz.
- Swarm: bieżący agent wybiera swojego następcę, wywołując narzędzie przekazania.

Swarm to "agent decyduje, co dalej"; GroupChat to "menedżer decyduje, co dalej." Decyzja Swarm żyje w wywołaniu narzędzia aktywnego agenta; decyzja GroupChat żyje w `GroupChatManager`.

## Zbuduj To

`code/main.py` implementuje Swarm od podstaw: dataclass Agenta, mechanizm przekazania (narzędzie zwraca Agenta) oraz pętlę uruchomieniową wykrywającą przełączenia agentów.

Demo: agent triażowy kieruje do specjalistów ds. zwrotów, sprzedaży lub wsparcia. Każdy specjalista ma własne narzędzia. Pętla uruchomieniowa wyświetla każde przekazanie.

Uruchom:

```
python3 code/main.py
```

## Użyj Tego

`outputs/skill-handoff-designer.md` projektuje topologię przekazań dla danego zadania: którzy agenci istnieją, jakie przekazania mogą wywoływać, jaki kontekst jest przenoszony.

## Wdróż To

Lista kontrolna:

- **Logowanie przekazań.** Każde przekazanie zapisuje zdarzenie śledzenia z agenta-źródła, agenta-celu i migawką kontekstu.
- **Reguły transferu kontekstu.** Zdecyduj, co jest przenoszone przy przekazaniu: pełna historia (kosztowna), ostatnie N wiadomości lub podsumowanie.
- **Bariera ochronna na przekazaniu.** Przekazanie do specjalisty z innymi uprawnieniami narzędzi musi być uwierzytelnione — w przeciwnym razie wstrzyknięcie promptu może wymusić niechciane przekazania.
- **Wykrywanie pętli.** Dwaj agenci przekazujący sobie nawzajem to częsta awaria; wykryj prostą kontrolą pierścienia ostatnich K.
- **Agent rezerwowy.** Jeśli cel przekazania nie istnieje, spadnij do bezpiecznego domyślnego.

## Ćwiczenia

1. Uruchom `code/main.py`, przekaż do agenta ds. zwrotów. Potwierdź, że aktywny agent w drugiej turze to refund.
2. Dodaj regułę wykrywania pętli: jeśli ci sami dwaj agenci przekazali sobie 3 razy z rzędu, wymuś wyjście. Zaprojektuj rezerwowy.
3. Przeczytaj dokumentację OpenAI Agents SDK na temat filtrów przekazań. Zaimplementuj wersję "podsumuj-przy-przekazaniu": wychodzący agent kompresuje kontekst do podsumowania w punktach przed przejęciem przez wchodzącego agenta.
4. Porównaj przekazanie Swarm z selektorem GroupChatManager. Który wzorzec pogarsza wstrzyknięcie promptu i dlaczego?
5. Przeczytaj cookbook Swarm (https://developers.openai.com/cookbook/examples/orchestrating_agents). Zidentyfikuj jedną jawną decyzję projektową, którą Swarm podejmuje, a którą OpenAI Agents SDK zmieniło lub zachowało.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|----------------|------------------------|
| Rutyna | "Prompt agenta" | Prompt systemowy + lista narzędzi. Definiuje rolę i dostępne przekazania. |
| Przekazanie | "Transfer do innego agenta" | Narzędzie, które aktywny agent może wywołać, zwracające nowego Agenta. Środowisko przełącza aktywnego agenta. |
| Bezstanowy | "Brak pamięci między uruchomieniami" | Swarm niczego nie utrwala; pamięć jest odpowiedzialnością wywołującego. |
| Aktywny agent | "Kto teraz mówi" | Agent aktualnie prowadzący konwersację. Przekazanie to zmienia. |
| Transfer kontekstu | "Co jest przenoszone przy przekazaniu" | Zasady określające, jaką historię widzi wchodzący agent: pełną, ostatnie N lub podsumowaną. |
| Pętla przekazań | "Agenci ping-pongują" | Tryb awarii, w którym dwaj agenci ciągle przekazują sobie nawzajem. |
| OpenAI Agents SDK | "Produkcyjny Swarm" | Następca z marca 2025; dodaje sesje, bariery ochronne, śledzenie na bazie prymitywu przekazania. |
| Filtr przekazań | "Brama przy transferze" | Funkcja SDK do inspekcji i modyfikacji kontekstu na granicy przekazania. |

## Dalsza Literatura

- [Cookbook OpenAI — Orkiestracja Agentów: Rutyny i Przekazania](https://developers.openai.com/cookbook/examples/orchestrating_agents) — referencyjne przedstawienie
- [Repozytorium OpenAI Swarm](https://github.com/openai/swarm) — oryginalna implementacja, zachowana jako koncepcyjna referencja
- [Dokumentacja OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — produkcyjny następca z sesjami i śledzeniem
- [Notatki Anthropic o przekazaniach w Claude](https://docs.anthropic.com/en/docs/claude-code) — jak subagenci Claude Code używają wzorca podobnego do przekazania przez `Task`