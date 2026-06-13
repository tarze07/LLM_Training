# Capstone 11 — Kokpit Obserwowalności i Ewaluacji LLM

> Langfuse przeszedł na open-core. Arize Phoenix opublikował mapowania semconv GenAI 2026. Helicone i Braintrust podwoili stawkę na atrybucję kosztów na użytkownika. OpenLLMetry od Traceloop stał się de-facto SDK do instrumentacji. Produkcyjny kształt to ClickHouse dla śladów, Postgres dla metadanych, Next.js dla UI i mała armia zadań ewaluacyjnych (DeepEval, RAGAS, LLM-judge) działających na próbkowanych śladach. Zbuduj jeden samodzielnie hostowany, pozyskuj dane z co najmniej czterech rodzin SDK i zademonstruj wykrycie wstrzykniętej regresji w mniej niż pięć minut.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P17 · P18
**Time:** 25 hours

## Problem

Każdy zespół AI działający produkcyjnie w 2026 roku utrzymuje płaszczyznę obserwowalności obok modelu. Atrybucja kosztów. Wykrywanie halucynacji. Monitorowanie dryfu. Sygnał jailbreaku. Kokpity SLO. Alarmy o wycieku PII. Otwartoźródłowe referencje — Langfuse, Phoenix, OpenLLMetry — zbiegły się do semantycznych konwencji GenAI OpenTelemetry jako schematu pozyskiwania. Możesz teraz instrumentować OpenAI, Anthropic, Google, LangChain, LlamaIndex i vLLM za pomocą jednego SDK i wysyłać kompatybilne spany.

Zbudujesz samodzielnie hostowany kokpit, który pozyskuje dane z co najmniej czterech rodzin SDK, uruchamia mały zestaw zadań ewaluacyjnych na próbkowanych śladach, wykrywa dryf i alarmuje. Miara pomiarowa: mając celowo wstrzykniętą regresję (prompt, który zaczyna produkować PII), kokpit wykrywa ją i odpalają alarm w mniej niż pięć minut.

## Koncepcja

Pozyskiwanie to OTLP HTTP. SDK produkuje spany GenAI-semconv: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. Spany lądują w ClickHouse dla analityki kolumnowej; metadane (użytkownicy, sesje, aplikacje) lądują w Postgresie.

Ewaluacje działają jako zadania wsadowe na próbkowanych śladach. DeepEval punktuje wierność, toksyczność i trafność odpowiedzi. RAGAS punktuje metryki wyszukiwania, gdy ślad niesie kontekst wyszukiwania. Niestandardowe LLM-judge wykonują sprawdzenia specyficzne dla domeny (wyciek PII, odpowiedź poza polityką). Przebiegi ewaluacji zapisują z powrotem do ClickHouse jako spany ewaluacyjne połączone z nadrzędnym śladem.

Wykrywanie dryfu obserwuje rozkłady przestrzeni osadzeń w czasie (PSI lub dywergencja KL na osadzeniach promptów) plus trendy wyników ewaluacji. Alarmy zasilają Prometheus Alertmanager, a następnie Slack / PagerDuty. UI to Next.js 15 z Recharts.

## Architektura

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Stack

- Pozyskiwanie: OpenTelemetry SDK + semantyczne konwencje GenAI; transport OTLP HTTP
- Kolektor: OpenTelemetry Collector z procesorem tail-sampling (dla kontroli kosztów)
- Przechowywanie: ClickHouse dla spanów, Postgres dla metadanych, S3 dla archiwum surowych zdarzeń
- Ewaluacje: DeepEval, RAGAS 0.2, pakiet ewaluatorów Arize Phoenix, niestandardowy LLM-judge
- Dryf: PSI / KL na połączonych osadzeniach promptów (sentence-transformers) co tydzień
- Alarmowanie: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- SDK obsługiwane od razu: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

## Build It

1. **Konfiguracja kolektora.** OpenTelemetry Collector z odbiornikiem OTLP HTTP, próbnikiem ogona zatrzymującym 100% błędnych śladów i 10% sukcesów oraz eksporterami do ClickHouse i S3.

2. **Schemat ClickHouse.** Tabela `spans` z kolumnami odzwierciedlającymi GenAI semconv: `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id` plus worek JSON dla długich ładunków. Dodaj indeksy wtórne według user_id i app_id.

3. **Test pokrycia SDK.** Napisz małą aplikację kliencką używającą każdego SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) z auto-instrumentacją OpenLLMetry. Zweryfikuj, że każda produkuje kanoniczne spany GenAI lądujące w ClickHouse.

4. **Zadania ewaluacyjne.** Zaplanowane zadanie odczytuje ostatnie 15 minut próbkowanych śladów i uruchamia DeepEval faithfulness, toxicity i answer relevance. Wyniki to spany ewaluacyjne połączone z nadrzędnym śladem.

5. **Niestandardowy LLM-judge.** Sędzia wykrywający wyciek PII: mając odpowiedź, wywołaj strażniczy LLM, aby ocenił prawdopodobieństwo wycieku PII. Odpowiedzi z wysokim wynikiem trafiają do kolejki triażowej.

6. **Wykrywanie dryfu.** Tygodniowe zadanie oblicza PSI między połączonymi osadzeniami promptów tego tygodnia a bazą z ostatnich 4 tygodni. Jeśli PSI powyżej progu, alarmuj.

7. **Kokpit.** Next.js 15 ze stronami: przegląd (spany/s, koszt/użytkownik, p95 opóźnienia), ślady (wyszukiwanie + waterfall), ewaluacje (trend wierności, toksyczność), dryf (PSI w czasie), alarmy.

8. **Łańcuch alarmowania.** Eksporter Prometheus odczytuje agregaty wyników ewaluacji i percentyle opóźnień; Alertmanager kieruje do Slack dla ostrzeżeń i PagerDuty dla krytycznych naruszeń.

9. **Sonda regresji.** Wstrzyknij błąd: oceniany chatbot zaczyna wyciekać fałszywe SSN 1% czasu. Zmierz MTTR: od wdrożenia błędu do alarmu Slack.

## Use It

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Ship It

`outputs/skill-llm-observability.md` jest rezultatem. Mając aplikację LLM, kokpit pozyskuje jej ślady, uruchamia ewaluacje, alarmuje o dryfie i wyświetla podział kosztów/użytkownik w Next.js.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Pokrycie schematu śladów | Liczba rodzin SDK produkujących kanoniczne spany GenAI (cel: 6+) |
| 20 | Poprawność ewaluacji | Wyniki DeepEval / RAGAS vs ręcznie oznaczony zestaw |
| 20 | UX kokpitu | MTTR na wstrzykniętej regresji (cel poniżej 5 minut) |
| 20 | Koszt / skala | Utrzymane pozyskiwanie przy 1k spanów/s bez zaległości |
| 15 | Alarmowanie + wykrywanie dryfu | Łańcuch Prometheus/Alertmanager przećwiczony od końca do końca |
| **100** | | |

## Ćwiczenia

1. Dodaj niestandardową instrumentację dla frameworka Haystack. Zweryfikuj, że kanoniczne spany lądują w ClickHouse z wiernymi atrybutami `gen_ai.*`.

2. Zamień DeepEval na ewaluatory Phoenix na tych samych śladach. Zmierz dryf wyników między dwoma silnikami ewaluacji.

3. Wyostrz detektor dryfu: oblicz PSI na app-id, a nie globalnie. Pokaż ślady dryfu na aplikację.

4. Dodaj stronę "wpływ na użytkownika": koszt-na-użytkownika i wskaźnik awarii-na-użytkownika z wykresami.

5. Zbuduj politykę próbkowania ogona, która zatrzymuje 100% śladów z toksycznością > 0.5 plus 10% stratyfikowaną próbkę reszty. Zmierz wprowadzone odchylenie próbkowania.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| GenAI semconv | "Atrybuty OTel LLM" | Specyfikacja OpenTelemetry 2025 dla atrybutów spanów LLM (system, model, tokeny) |
| Tail sampling | "Próbkowanie po śladzie" | Kolektor decyduje, czy zachować lub odrzucić ślad po jego zakończeniu (może podejrzeć błędy) |
| PSI | "Indeks stabilności populacji" | Metryka dryfu porównująca dwie dystrybucje; > 0.2 zazwyczaj sygnalizuje znaczący dryf |
| LLM-judge | "Ewaluacja jako model" | LLM punktujący wyjście innego LLM według rubryki (wierność, toksyczność, PII) |
| Tail-sampling policy | "Reguła zatrzymywania" | Reguła decydująca, które ślady zachować vs odrzucić; błędne + współczynnik próbkowania |
| Eval span | "Powiązany ślad ewaluacji" | Span potomny niosący wynik ewaluacji połączony z oryginalnym spanem wywołania LLM |
| Cost per user | "Ekonomia jednostkowa" | Koszt dolarowy przypisany do user_id w oknie; kluczowa metryka produktowa |

## Dalsza lektura

- [Langfuse](https://github.com/langfuse/langfuse) — the reference open-core observability platform
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — alternate reference with strong drift support
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — auto-instrumentation SDK family
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the ingest schema
- [Helicone](https://www.helicone.ai) — alternate hosted observability
- [Braintrust](https://www.braintrust.dev) — alternate eval-first platform
- [ClickHouse documentation](https://clickhouse.com/docs) — columnar span store
- [DeepEval](https://github.com/confident-ai/deepeval) — evaluator library