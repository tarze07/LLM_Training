# OpenAI Agents SDK: Przekazania, zabezpieczenia, śledzenie

> OpenAI Agents SDK to lekki wieloagentowy framework zbudowany na Responses API. Pięć prymitywów: Agent, Handoff, Guardrail, Session, Tracing. Przekazania to narzędzia nazwane `transfer_to_<agent>`. Zabezpieczenia uruchamiają się na wejściu lub wyjściu. Śledzenie jest domyślnie włączone.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## Learning Objectives

- Wymień pięć prymitywów OpenAI Agents SDK.
- Wyjaśnij przekazania: dlaczego są modelowane jako narzędzia, jaki kształt nazwy widzi model i jak przenosi się kontekst.
- Rozróżnij zabezpieczenia wejściowe, wyjściowe i narzędziowe; wyjaśnij `run_in_parallel` vs tryb blokujący.
- Zaimplementuj w stdlib środowisko uruchomieniowe z przekazaniami + zabezpieczeniami + śledzeniem w stylu span.

## The Problem

Agenci, którzy nie potrafią czysto delegować, kończą wpychając wszystko do jednego promptu. Agenci bez zabezpieczeń wysyłają PII, wyniki naruszające politykę lub zapętlają się w nieskończoność. SDK OpenAI kodyfikuje trzy prymitywy, które czynią pracę wieloagentową wykonalną.

## The Concept

### Pięć prymitywów

1. **Agent.** LLM + instrukcje + narzędzia + przekazania.
2. **Handoff.** Delegacja do innego agenta. Reprezentowana modelowi jako narzędzie nazwane `transfer_to_<agent_name>`.
3. **Guardrail.** Walidacja na wejściu (tylko pierwszy agent), wyjściu (tylko ostatni agent) lub wywołaniu narzędzia (na funkcję narzędziową).
4. **Session.** Automatyczna historia rozmowy między turami.
5. **Tracing.** Wbudowane spany dla generacji LLM, wywołań narzędzi, przekazań, zabezpieczeń.

### Przekazania jako narzędzia

Model widzi `transfer_to_billing_agent` na swojej liście narzędzi. Wywołanie go sygnalizuje środowisku:

1. Skopiuj kontekst rozmowy (lub zwiń go przez `nest_handoff_history` beta).
2. Zainicjuj agenta docelowego z jego instrukcjami.
3. Kontynuuj uruchomienie z agentem docelowym.

Jest to wzór nadzorcy (Lekcja 13 / Lekcja 28) sproduktyzowany.

### Zabezpieczenia

Trzy rodzaje:

- **Zabezpieczenia wejściowe.** Uruchamiane na wejściu pierwszego agenta. Odrzucają niebezpieczne lub niezgodne z zakresem żądania przed jakimkolwiek wywołaniem LLM.
- **Zabezpieczenia wyjściowe.** Uruchamiane na wyjściu ostatniego agenta. Wykrywają wycieki PII, naruszenia polityki, nieprawidłowe odpowiedzi.
- **Zabezpieczenia narzędziowe.** Uruchamiane na funkcję narzędziową. Walidują argumenty, sprawdzają uprawnienia, audytują wykonanie.

Tryb:

- **Równoległy** (domyślny). LLM zabezpieczeń działa równolegle z głównym LLM. Niższe opóźnienie ogonowe. Jeśli zadziała, praca głównego LLM jest odrzucana (marnotrawstwo tokenów).
- **Blokujący** (`run_in_parallel=False`). LLM zabezpieczeń działa najpierw. Jeśli zadziała, żadne tokeny nie są marnowane na główne wywołanie.

Zadziałania zgłaszają `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`.

### Śledzenie

Domyślnie włączone. Każda generacja LLM, wywołanie narzędzia, przekazanie i zabezpieczenie emituje span. `OPENAI_AGENTS_DISABLE_TRACING=1` rezygnuje. `add_trace_processor(processor)` kieruje spany do własnego backendu obok OpenAI.

### Sesje

`Session` przechowuje historię rozmowy w backendzie (SQLite, Redis, niestandardowy). `Runner.run(agent, input, session=session)` automatycznie ładuje i dołącza.

### Gdzie ten wzór zawodzi

- **Dryf przekazania.** Agent A przekazuje do agenta B, który przekazuje z powrotem do agenta A. Dodaj licznik skoków.
- **Ominięcie zabezpieczeń.** Zabezpieczenia narzędziowe uruchamiają się tylko na funkcjach narzędziowych; wbudowane narzędzia (czytnik plików, pobieranie sieciowe) wymagają oddzielnej polityki.
- **Nadmierne śledzenie.** Wrażliwa treść w spanach. Połącz z regułami przechwytywania treści OTel GenAI (Lekcja 23) — przechowuj zewnętrznie, odwołuj się przez ID.

## Build It

`code/main.py` implementuje kształt SDK w stdlib:

- `Agent`, `FunctionTool`, `Handoff` (jako narzędzie funkcyjne z semantyką transferu).
- `Runner` z zabezpieczeniami wejścia/wyjścia/narzędzi, wysyłką przekazań i licznikiem skoków.
- Prosty emiter spanów pokazujący kształt śladu.
- Agent triażu, który przekazuje do rozliczeń lub wsparcia na podstawie zapytania użytkownika; zabezpieczenie zadziała na jednym wejściu.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje dwa udane przekazania, jedno zadziałanie zabezpieczenia wejściowego i drzewo spanów odzwierciedlające to, co emituje prawdziwe SDK.

## Use It

- **OpenAI Agents SDK** dla produktów OpenAI-first.
- **Claude Agent SDK** (Lekcja 17) dla produktów Claude-first.
- **LangGraph** (Lekcja 13) gdy chcesz jawny stan i trwałe wznawianie.
- **Niestandardowe** gdy potrzebujesz dokładnej kontroli (głos, multi-provider, federacyjne wdrożenia).

## Ship It

`outputs/skill-agents-sdk-scaffold.md` tworzy szkielet aplikacji Agents SDK z agentem triażu, przekazaniami, zabezpieczeniami wejścia/wyjścia/narzędzi, magazynem sesji i procesorem śladów.

## Exercises

1. Dodaj licznik skoków przekazania: odmów po N transferach. Prześledź zachowanie.
2. Zaimplementuj `nest_handoff_history` jako opcję — zwiń poprzednie wiadomości w jedno podsumowanie przed transferem.
3. Napisz blokujące zabezpieczenie wyjściowe. Porównaj opóźnienie na promptach, które by zadziałały, vs tych, które przechodzą.
4. Podłącz `add_trace_processor` do rejestratora JSON. Jaki kształt emituje na span?
5. Przeczytaj dokumentację SDK. Przenieś swoją zabawkę stdlib na `openai-agents-python`. Co wymodelowałeś źle?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "LLM + instrukcje" | Typ Agent w SDK; posiada narzędzia i przekazania |
| Handoff | "Transfer" | Narzędzie, które model wywołuje, aby delegować do innego agenta |
| Guardrail | "Sprawdzenie polityki" | Walidacja na wejściu / wyjściu / wywołaniu narzędzia |
| Tripwire | "Zadziałanie zabezpieczenia" | Wyjątek zgłaszany, gdy zabezpieczenie odrzuca |
| Session | "Magazyn historii" | Pamięć rozmowy utrwalana między uruchomieniami |
| Tracing | "Spany" | Wbudowana obserwowalność nad LLM + narzędzie + przekazanie + zabezpieczenie |
| Blokujące zabezpieczenie | "Sprawdzenie sekwencyjne" | Zabezpieczenie działa najpierw; brak marnowania tokenów przy zadziałaniu |
| Równoległe zabezpieczenie | "Sprawdzenie współbieżne" | Zabezpieczenie działa równolegle; niższe opóźnienie, marnuje tokeny przy zadziałaniu |

## Further Reading

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — prymitywy, przekazania, zabezpieczenia, śledzenie
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — odpowiednik w stylu Claude
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — kiedy w ogóle sięgać po przekazania
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — standard, do którego mapują się spany Agents SDK