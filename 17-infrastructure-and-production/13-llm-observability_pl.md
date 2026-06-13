# Wybór stosu obserwowalności LLM

> Rynek obserwowalności 2026 dzieli się na dwie kategorie. Platformy deweloperskie (LangSmith, Langfuse, Comet Opik) łączą monitorowanie z ewaluacjami, zarządzaniem promptami, powtórkami sesji. Narzędzia bramy/instrumentacji (Helicone, SigNoz, OpenLLMetry, Phoenix) koncentrują się na telemetrii. Langfuse ma rdzeń na licencji MIT z silną równowagą OSS (50K zdarzeń/miesiąc za darmo w chmurze). Phoenix jest natywny dla OpenTelemetry na licencji Elastic License 2.0 — doskonały do wizualizacji dryfu/RAG, ale nie jest trwałym backendem produkcyjnym. Arize AX używa integracji zero-copy Iceberg/Parquet, twierdząc, że jest 100x tańszy niż monolityczna obserwowalność. LangSmith prowadzi dla LangChain/LangGraph, $39/użytkownik/miesiąc, samo-hostowanie tylko w Enterprise. Helicone jest oparty na proxy z konfiguracją 15-30 min, 100K żądań/miesiąc za darmo, ale ma mniejszą głębię w śladach agentów. Powszechny wzorzec produkcyjny: brama (Helicone/Portkey) + platforma ewaluacyjna (Phoenix/TruLens) sklejone przez OpenTelemetry.

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Learning Objectives

- Odróżnij platformy deweloperskie (zintegrowane: ewaluacje + prompty + sesje) od narzędzi bramy/telemetrii (ślady + metryki tylko).
- Przyporządkuj sześć głównych narzędzi (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) do ich licencji, cen i przypadków użycia.
- Wyjaśnij wzorzec klejenia OpenTelemetry, który pozwala połączyć narzędzie bramy z oddzielną platformą ewaluacyjną.
- Nazwij wyróżnik kosztowy 2026 (podejście zero-copy Arize AX vs monolityczne pozyskiwanie) i podaj zgrubny mnożnik 100x.

## The Problem

Wdrożyłeś funkcję LLM. Działa. Nie masz wglądu w awarie promptów, pętle narzędzi, regresje latencji, piki kosztów ani współczynnik trafień pamięci podręcznej promptów. Googlujesz „obserwowalność LLM" i dostajesz osiem narzędzi, wszystkie twierdzące, że rozwiązują ten sam problem na trzech różnych poziomach cenowych.

Nie rozwiązują tego samego problemu. LangSmith odpowiada „dlaczego to uruchomienie LangGraph się nie udało?" Phoenix odpowiada „czy mój potok RAG dryfuje?" Helicone odpowiada „która aplikacja pali tokeny?" Langfuse odpowiada „czy mogę samo-hostować całość?" Różne narzędzia, różni odbiorcy.

Wybór obejmuje cztery osie: stos (LangChain? surowe SDK? wielu dostawców?), tolerancja licencji (tylko MIT? Elastic OK? komercyjne w porządku?), budżet (darmowy poziom? $100/miesiąc? $1000/miesiąc?) i samo-hostowanie (konieczne? miłe? nigdy?).

## The Concept

### Dwie kategorie

**Platformy deweloperskie** łączą obserwowalność z ewaluacjami, zarządzaniem promptami, wersjonowaniem zestawów danych, powtórkami sesji. Przeprowadzasz eksperymenty, widzisz, który prompt zadziałał, regresję zestawu danych z nowym promptem wobec starych zwycięzców. LangSmith, Langfuse, Comet Opik.

**Narzędzia bramy/telemetrii** instrumentują wywołania inferencji — prompt, odpowiedź, tokeny, latencja, model, koszt. Helicone, SigNoz, OpenLLMetry, Phoenix. Minimalistyczne. Mogą być łączone z oddzielnym narzędziem ewaluacyjnym przez OpenTelemetry.

### Langfuse — równowaga OSS

- Rdzeń na licencji Apache / MIT; samo-hostowanie przez Docker.
- Darmowy poziom w chmurze: 50K zdarzeń/miesiąc. Płatny: $29/miesiąc dla zespołu.
- Ewaluacje, zarządzanie promptami, ślady, zestawy danych. Rozsądne pokrycie wszystkich czterech funkcji platformy deweloperskiej.
- Punkt słodki: chcesz funkcji klasy LangSmith, ale musisz samo-hostować lub pozostać na licencji OSS.

### Phoenix (Arize) — telemetria-first, natywny OpenTelemetry

- Elastic License 2.0; samo-hostowanie trywialne.
- Doskonały w wizualizacji RAG i dryfu. Wykresy rozrzutu przestrzeni embeddingów dostarczane jako pierwszorzędne.
- Nie zaprojektowany jako trwały backend produkcyjny — głównie obserwowalność na czas dewelopmentu.
- Punkt słodki: rozwój potoku RAG, debugowanie dryfu, łączy się z oddzielną bramą dla produkcji.

### Arize AX — zagranie o skalę

- Komercyjny. Integracja data lake zero-copy przez Iceberg/Parquet.
- Twierdzi ~100x taniej niż monolityczna obserwowalność (klasa Datadog) przy skali. Matematyka: przechowujesz ślady we własnym Parquet na S3; Arize czyta bezpośrednio.
- Punkt słodki: >10M śladów/dzień, istniejące data lake, chcesz pulpitów nawigacyjnych specyficznych dla LLM bez cen Datadog.

### LangSmith — LangChain/LangGraph first

- Komercyjny, $39/użytkownik/miesiąc. Samo-hostowanie tylko w Enterprise.
- Najlepszy w klasie dla stosów LangChain i LangGraph. Jeśli nie jesteś na żadnym z nich, jest mniej przekonujący.
- Punkt słodki: zespół zobowiązany do LangChain, skłonny płacić.

### Helicone — minimalny opłacalny oparty na proxy

- Konfiguracja 15-30 minut przez zmianę `OPENAI_API_BASE` na proxy Helicone.
- Licencja MIT; 100K żądań/miesiąc za darmo, płatne $20/miesiąc+.
- Obejmuje failover, buforowanie, limity szybkości — działa również jako brama.
- Mniejsza głębia w śladach agentów / wieloetapowych.
- Punkt słodki: szybki start, aplikacja jedno-stosowa, potrzebujesz bramy + obserwowalności w jednym.

### Opik (Comet) — platforma deweloperska OSS

- Apache 2.0, w pełni OSS.
- Podobny zestaw funkcji do Langfuse z dziedzictwem Comet.
- Punkt słodki: zespoły ML już na Comet, chcą obserwowalności LLM w tym samym panelu.

### SigNoz — pełny APM natywny dla OpenTelemetry

- Apache 2.0. Obsługuje ogólny APM plus LLM przez OpenTelemetry.
- Punkt słodki: ujednolicona obserwowalność między usługami i wywołaniami LLM.

### Klej: OpenTelemetry + konwencje semantyczne GenAI

OpenTelemetry opublikowało konwencje semantyczne GenAI pod koniec 2025 (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`). Narzędzia konsumujące OTel mogą współdziałać. Wzorzec produkcyjny, który się wyłania:

1. Emituj OTel z konwencjami GenAI z każdego wywołania LLM.
2. Kieruj do bramy (Helicone / Portkey) na co dzień.
3. Podwójnie wysyłaj do platformy ewaluacyjnej (Phoenix / Langfuse) dla regresji.
4. Archiwizuj w data lake (Iceberg) do długoterminowej analizy przez Arize AX lub DuckDB.

### Pułapka: instrumentacja na niewłaściwej warstwie

Instrumentacja wewnątrz frameworka agenta (np. dodawanie śladów LangSmith) wiąże cię z tym frameworkiem. Instrumentacja na warstwie HTTP/SDK OpenAI (przez OpenLLMetry lub twoją bramę) jest przenośna.

### Próbkowanie — nie możesz zatrzymać wszystkiego

Przy >1M żądań/dzień, pełne przechowywanie śladów kosztuje więcej niż wywołania LLM. Próbkuj według reguł: 100% błędów, 100% wysokiego kosztu, 5% sukcesów. Zawsze utrzymuj agregaty; utrzymuj surowe dla długiego ogona.

### Liczby, które powinieneś zapamiętać

- Langfuse darmowa chmura: 50K zdarzeń/miesiąc.
- LangSmith: $39/użytkownik/miesiąc.
- Helicone darmowe: 100K żądań/miesiąc.
- Twierdzenie Arize AX: ~100x taniej niż monolityczne przy skali.
- Konwencje GenAI OpenTelemetry: dostarczone 2025, szeroko przyjęte 2026.

## Use It

`code/main.py` symuluje dzień 1M śladów w różnych strategiach przechowywania (100% pozyskiwanie, próbkowanie, próbkowanie + błędy). Raportuje koszt przechowywania i co jest tracone w każdej strategii.

## Ship It

Ta lekcja produkuje `outputs/skill-observability-stack.md`. Biorąc stos, skalę, budżet, postawę licencyjną, wybiera narzędzie(a).

## Exercises

1. Twój zespół na LangChain chce OSS samo-hostowanej obserwowalności. Wybierz Langfuse lub Opik i uzasadnij.
2. Przy 5M śladów/dzień z wycenami Datadog $150K/miesiąc, oblicz próg rentowności dla Arize AX.
3. Zaprojektuj zestaw atrybutów GenAI OpenTelemetry, który wytyczne Twojej organizacji powinny nakazać dla każdego wywołania LLM.
4. Przedstaw argument, czy sam Phoenix jest wystarczający do produkcji. Kiedy nie wystarcza?
5. Helicone ma 20 ms narzutu proxy. Przy P99 TTFT 300 ms, czy to akceptowalne? A jeśli SLA wynosi 100 ms?

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| OpenLLMetry | „OTel dla LLM" | Otwartoźródłowa instrumentacja OpenTelemetry dla LLM |
| GenAI conventions | „atrybuty OTel" | Standardowe nazwy atrybutów OTel dla wywołań LLM |
| LangSmith | „obserwowalność LangChain" | Komercyjna platforma związana z ekosystemem LangChain |
| Langfuse | „OSS LangSmith" | MIT OSS z podobnym zestawem funkcji |
| Phoenix | „narzędzie deweloperskie Arize" | Platforma deweloperska/ewaluacyjna natywna dla OpenTelemetry |
| Arize AX | „obserwowalność skali" | Komercyjna obserwowalność zero-copy Iceberg/Parquet |
| Helicone | „obserwowalność proxy" | Proxy HTTP zbierające telemetrię LLM + funkcje bramy |
| Opik | „Comet LLM" | Platforma deweloperska OSS Apache 2.0 od Comet |
| Session replay | „ponowne uruchomienie śladu" | Odtworzenie pełnej sesji agenta z wywołaniami narzędzi |
| Eval | „test offline" | Uruchomienie modelu/promptu kandydackiego na oznaczonym zbiorze danych |

## Further Reading

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)

(End of file - total 141 lines)