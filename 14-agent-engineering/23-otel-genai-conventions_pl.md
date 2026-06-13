# Konwencje Semantyczne OpenTelemetry GenAI

> Grupa SIG GenAI OpenTelemetry (uruchomiona w kwietniu 2024) definiuje standardowy schemat telemetrii agentów. Nazwy spanów, atrybuty i reguły przechwytywania treści konwergują między dostawcami, aby ślady agentów znaczyły to samo w Datadog, Grafanie, Jaeger i Honeycomb.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 minutes

## Learning Objectives

- Wymień kategorie spanów GenAI: model/klient, agent, narzędzie.
- Rozróżnij `invoke_agent` CLIENT vs INTERNAL i kiedy każdy ma zastosowanie.
- Wypisz atrybuty GenAI najwyższego poziomu: nazwa dostawcy, model żądania, ID źródła danych.
- Wyjaśnij kontrakt przechwytywania treści: opcjonalność, `OTEL_SEMCONV_STABILITY_OPT_IN`, zalecenie referencji zewnętrznych.

## Problem

Każdy dostawca wymyśla własne nazwy spanów. Zespoły operacyjne kończą z budowaniem pulpitów na każdy framework. SIG GenAI OpenTelemetry rozwiązuje to, definiując jeden standard, do którego celuje cały ekosystem.

## Koncepcja

### Kategorie spanów

1. **Spany modelu/klienta.** Obejmują surowe wywołania LLM. Emitowane przez SDK dostawców (Anthropic, OpenAI, Bedrock) i adaptery modeli frameworków.
2. **Spany agenta.** `create_agent` (gdy agent jest konstruowany) i `invoke_agent` (gdy działa).
3. **Spany narzędzia.** Po jednym na wywołanie narzędzia; połączone ze spanem agenta relacją rodzic–dziecko.

### Nazewnictwo spanów agenta

- Nazwa spanu: `invoke_agent {gen_ai.agent.name}` jeśli nazwany; w przeciwnym razie `invoke_agent`.
- Rodzaj spanu:
  - **CLIENT** — dla zdalnych usług agenta (OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL** — dla frameworków agentów w procesie (LangChain, CrewAI, lokalny ReAct).

### Kluczowe atrybuty

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — ID modelu.
- `gen_ai.response.model` — rozwiązany model (może różnić się od żądania z powodu routingu).
- `gen_ai.agent.name` — identyfikator agenta.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — dla RAG: który korpus lub magazyn był konsultowany.

Konwencje specyficzne dla technologii istnieją dla Anthropic, Azure AI Inference, AWS Bedrock, OpenAI.

### Przechwytywanie treści

Domyślna reguła: instrumentacje NIE POWINNY przechwytywać wejść/wyjść domyślnie. Przechwytywanie jest opcjonalne przez:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

Zalecany wzorzec produkcyjny: przechowuj treść zewnętrznie (S3, twój magazyn logów), rejestruj referencje na spanach (ID wskaźników, nie proza). To jest obrona przed zatruciem treścią z Lekcji 27 wbudowana w obserwowalność.

### Stabilność

Większość konwencji jest eksperymentalna od marca 2026. Włącz stabilny podgląd przez:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ mapuje atrybuty GenAI natywnie do swojego schematu LLM Observability. Inne backendy (Grafana, Honeycomb, Jaeger) obsługują surowe atrybuty.

### Gdzie ten wzorzec zawodzi

- **Przechwytywanie pełnych promptów w spanach.** PII, sekrety, dane klientów w śladach, które operacje mogą czytać. Przechowuj zewnętrznie.
- **Brak `gen_ai.provider.name`.** Pulpity wielodostawcowe psują się, gdy brakuje atrybucji.
- **Spany bez linków rodzica.** Osierocone spany narzędzi. Zawsze propaguj kontekst.
- **Nieustawienie opcjonalności stabilności.** Twoje atrybuty mogą zostać przemianowane przy aktualizacji backendu.

## Build It

`code/main.py` implementuje emiter spanów w stdlib zgodny z konwencjami GenAI:

- `Span` ze schematem atrybutów GenAI.
- `Tracer` z `start_span`, zagnieżdżone konteksty.
- Skryptowane uruchomienie agenta, które emituje: `create_agent`, `invoke_agent` (INTERNAL), spany na narzędzie, spany `chat` dla wywołań LLM.
- Tryb przechwytywania treści, który przechowuje prompty zewnętrznie i rejestruje ID na spanach.

Uruchom:

```
python3 code/main.py
```

Wynik: drzewo spanów ze wszystkimi wymaganymi atrybutami GenAI oraz „zewnętrzny magazyn" pokazujący opcjonalne referencje treści.

## Use It

- **Datadog LLM Observability** (v1.37+) mapuje atrybuty natywnie.
- **Langfuse / Phoenix / Opik** (Lekcja 24) — auto-instrumentacja ekosystemu.
- **Jaeger / Honeycomb / Grafana Tempo** — surowe ślady OTel; buduj pulpity z atrybutów GenAI.
- **Samodzielnie hostowane** — uruchom Kolektor OTel z procesorem GenAI.

## Ship It

`outputs/skill-otel-genai.md` podłącza spany OTel GenAI do istniejącego agenta z domyślnym przechwytywaniem treści i magazynowaniem referencji zewnętrznych.

## Ćwiczenia

1. Zainstrumentuj swoją pętlę ReAct z Lekcji 01 za pomocą `invoke_agent` (INTERNAL) + spanów na narzędzie. Wyślij do instancji Jaeger.
2. Dodaj przechwytywanie treści w trybie „tylko referencje": prompty do SQLite, atrybuty spanów niosą tylko ID wierszy.
3. Przeczytaj specyfikację `gen_ai.data_source.id`. Podłącz ją do swojego wyszukiwania Mem0 z Lekcji 09.
4. Ustaw `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` i zweryfikuj, że twoje atrybuty nie zostaną przemianowane przez kolektor.
5. Zbuduj pulpit: „które błędy narzędzi korelują z którymi modelami" z samych atrybutów GenAI.

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| SIG GenAI | „Grupa OpenTelemetry GenAI" | Grupa robocza OTel definiująca schemat |
| invoke_agent | „Span agenta" | Nazwa spanu reprezentującego uruchomienie agenta |
| Span CLIENT | „Zdalne wywołanie" | Span dla wywołania zdalnej usługi agenta |
| Span INTERNAL | „W procesie" | Span dla uruchomienia agenta w procesie |
| gen_ai.provider.name | „Dostawca" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | „Źródło RAG" | Który korpus/magazyn został trafiony przy wyszukiwaniu |
| Przechwytywanie treści | „Logowanie promptów" | Opcjonalne przechwytywanie wiadomości; przechowuj zewnętrznie w produkcji |
| Opcjonalność stabilności | „Tryb podglądu" | Zmienna środowiskowa do przypięcia eksperymentalnych konwencji |

## Dalsza lektura

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — specyfikacja
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — spany GenAI domyślnie
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — wbudowane spany OTel
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — propagacja kontekstu śledzenia W3C