# OpenTelemetry GenAI — Śledzenie Wywołań Narzędzi od Końca do Końca

> Agent wywołuje pięć narzędzi, trzy serwery MCP i dwóch pod-agentów. Potrzebujesz jednego śladu obejmującego to wszystko. Semantyczne konwencje OpenTelemetry GenAI (stabilne atrybuty od v1.37 w górę) są standardem na rok 2026, natywnie wspieranym przez Datadog, Langfuse, Arize Phoenix, OpenLLMetry i AgentOps. Ta lekcja wymienia wymagane atrybuty, omawia hierarchię spanów (agent → LLM → narzędzie) i dostarcza emiter spanów oparty na stdlib, który można podłączyć do dowolnego eksportera OTel.

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Learning Objectives

- Wymienić wymagane atrybuty OTel GenAI dla spana LLM i spana wykonania narzędzia.
- Zbudować hierarchię śladu obejmującą pętlę agenta, wywołanie LLM, wywołanie narzędzia i dyspozycję klienta MCP.
- Zdecydować, jakie treści przechwytywać (opt-in) vs maskować (domyślnie).
- Emitować spany do lokalnego kolektora (Jaeger, Langfuse) bez przepisywania kodu narzędzi.

## The Problem

Debugowanie z lutego 2026: użytkownik raportuje "mój agent czasami odpowiada w 30 sekund, innym razem w 3 sekundy." Brak śladów. Logi pokazują wywołanie LLM, ale nie wysłanie narzędzia, nie komunikację tam i z powrotem z serwerem MCP, nie pod-agenta. Zgadujesz. W końcu znajdujesz: jeden serwer MCP okazjonalnie zawiesza się przy zimnym starcie.

Bez śledzenia end-to-end nie możesz tego znaleźć. OTel GenAI to naprawia.

Konwencje zostały ustalone w latach 2025-2026 w ramach grupy OpenTelemetry semantic-conventions. Definiują stabilne nazwy atrybutów, aby Datadog, Langfuse, Phoenix, OpenLLMetry i AgentOps wszystkie parsowały te same spany. Instrumentujesz raz; wysyłasz do dowolnego backendu.

## The Concept

### Hierarchia spanów

```
agent.invoke_agent  (górny, span INTERNAL)
 ├── llm.chat       (span CLIENT)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (span CLIENT)
 ├── llm.chat       (span CLIENT)
 └── subagent.invoke (INTERNAL)
```

Całość zagnieżdża się pod jednym identyfikatorem śladu. Identyfikatory spanów łączą relacje rodzic-dziecko.

### Wymagane atrybuty

Według semconv z lat 2025-2026:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — żądany model (np. `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — model faktycznie obsłużony.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — identyfikator odpowiedzi dostawcy do korelacji.

Dla spanów narzędzi:

- `gen_ai.tool.name` — identyfikator narzędzia.
- `gen_ai.tool.call.id` — identyfikator konkretnego wywołania.
- `gen_ai.tool.description` — opis narzędzia (opcjonalnie).

Dla spanów agentów:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Rodzaje spanów

- `SpanKind.CLIENT` dla wywołań przekraczających granicę procesu (dostawca LLM, serwer MCP).
- `SpanKind.INTERNAL` dla własnych kroków pętli agenta i wykonania narzędzia.

### Przechwytywanie treści na zasadzie opt-in

Domyślnie spany niosą metryki i czas — nie promptów ani odpowiedzi. Duże ładunki i PII są domyślnie wyłączone. Ustaw `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` i odpowiednie zmienne środowiskowe przechwytywania treści, aby włączyć zawartość. Przejrzyj uważnie przed włączeniem w produkcji.

### Zdarzenia na spanach

Zdarzenia na poziomie tokenów można dodawać jako zdarzenia spanów:

- `gen_ai.content.prompt` — komunikaty wejściowe.
- `gen_ai.content.completion` — komunikaty wyjściowe.
- `gen_ai.content.tool_call` — wywołanie narzędzia w formie, w jakiej zostało zapisane.

Zdarzenia porządkują się w czasie w ramach spana w celu szczegółowego odtworzenia.

### Eksportery

Spany OTel eksportują do:

- **Jaeger / Tempo.** OSS, on-prem.
- **Langfuse.** Specyficzne dla obserwowalności LLM; wizualizuje użycie tokenów.
- **Arize Phoenix.** Ewaluacje + śledzenie w jednym.
- **Datadog.** Komercyjny; natywnie parsuje atrybuty `gen_ai.*`.
- **Honeycomb.** Zorientowany kolumnowo; przyjazny dla zapytań.

Wszystkie mówią OTLP, formatem transmisyjnym. Twój kod się tym nie przejmuje.

### Propagacja przez MCP

Gdy klient MCP wywołuje serwer, wstrzyknij nagłówek W3C traceparent do żądania. Streamable HTTP obsługuje standardowe nagłówki. Stdio natywnie nie przenosi nagłówków HTTP; plan działania specyfikacji na 2026 rok omawia dodanie pola `_meta.traceparent` w wywołaniach JSON-RPC.

Do czasu wdrożenia: dołącz traceparent w `_meta` każdego żądania ręcznie. Serwer loguje identyfikator śladu.

### Metryki

Oprócz spanów, semconv GenAI definiuje metryki:

- `gen_ai.client.token.usage` — histogram.
- `gen_ai.client.operation.duration` — histogram.
- `gen_ai.tool.execution.duration` — histogram.

Użyj ich do dashboardów, które nie potrzebują szczegółów każdego wywołania.

### Warstwa AgentOps

AgentOps (założony 2024) specjalizuje się w obserwowalności GenAI. Owiń popularne frameworki (LangGraph, Pydantic AI, CrewAI), aby automatycznie emitować spany OTel. Przydatne, jeśli twój stos korzysta z obsługiwanego frameworka; w przeciwnym razie użyj ręcznej instrumentacji.

## Use It

`code/main.py` emituje spany w kształcie OTel na stdout (w formacie zbliżonym do OTLP-JSON) dla agenta, który wywołuje LLM, wysyła dwa narzędzia i wykonuje jedną rundę MCP. Żadnego prawdziwego eksportera — lekcja skupia się na kształcie spana i zestawie atrybutów. Wklej wynik do przeglądarki kompatybilnej z OTLP lub po prostu go przeczytaj.

Na co zwrócić uwagę:

- Identyfikator śladu jest współdzielony przez wszystkie spany.
- Powiązania rodzic-dziecko są kodowane przez `parentSpanId`.
- Wymagane atrybuty `gen_ai.*` są wypełnione.
- Przechwytywanie treści jest domyślnie wyłączone; jeden scenariusz włącza je przez zmienną środowiskową.

## Ship It

Ta lekcja produkuje `outputs/skill-otel-genai-instrumentation.md`. Mając kod agenta, umiejętność tworzy plan instrumentacji: gdzie dodać spany, które atrybuty wypełnić i które eksportery wybrać.

## Exercises

1. Uruchom `code/main.py`. Policz spany i zidentyfikuj, który jest CLIENT, a który INTERNAL.

2. Włącz przechwytywanie treści (zmienna środowiskowa) i potwierdź, że zdarzenia `gen_ai.content.prompt` i `gen_ai.content.completion` się pojawiają. Zwróć uwagę na implikacje dla PII.

3. Dodaj metrykę wykonania narzędzia `gen_ai.tool.execution.duration` i emituj ją jako próbkę histogramu na wywołanie.

4. Propaguj traceparent z nadrzędnego spana agenta do pola `_meta.traceparent` żądania MCP. Zweryfikuj, że serwer MCP zobaczyłby ten sam identyfikator śladu.

5. Przeczytaj specyfikację semconv OTel GenAI. Zidentyfikuj jeden atrybut wymieniony w semconv, którego kod tej lekcji NIE emituje. Dodaj go.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | Otwarty standard dla śladów, metryk i logów |
| GenAI semconv | "Semantyczne konwencje GenAI" | Stabilne nazwy atrybutów dla spanów LLM / narzędzi / agentów |
| `gen_ai.*` | "Przestrzeń nazw atrybutów" | Wszystkie atrybuty GenAI mają ten prefiks |
| Span | "Operacja mierzona w czasie" | Jednostka pracy z początkiem, końcem i atrybutami |
| Trace | "Pochodzenie między spanami" | Drzewo spanów współdzielących identyfikator śladu |
| SpanKind | "CLIENT / SERVER / INTERNAL" | Wskazówki dotyczące kierunku spana |
| OTLP | "OpenTelemetry Line Protocol" | Format transmisyjny dla eksporterów |
| Opt-in content | "Przechwytywanie promptów/odpowiedzi" | Domyślnie wyłączone; zmienna środowiskowa do włączenia |
| traceparent | "Nagłówek W3C" | Propaguje kontekst śledzenia między usługami |
| Exporter | "Narzędzie wysyłające do backendu" | Komponent wysyłający spany do Jaeger / Datadog / itp. |

## Further Reading

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — kanoniczne konwencje dla spanów, metryk i zdarzeń GenAI
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — lista atrybutów spanów LLM i wykonania narzędzi
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — span `invoke_agent` na poziomie agenta
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — źródło prawdy na GitHub
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — przewodnik integracji produkcyjnej