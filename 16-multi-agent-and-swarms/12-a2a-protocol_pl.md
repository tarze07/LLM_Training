# A2A — Protokół Agent-do-Agenta

> Google ogłosiło A2A w kwietniu 2025; do kwietnia 2026 specyfikacja jest dostępna na https://a2a-protocol.org/latest/specification/ i popiera ją 150+ organizacji. A2A jest poziomym uzupełnieniem MCP (Lekcja 13): gdzie MCP jest pionowy (agent ↔ narzędzia), A2A jest równorzędny (agent ↔ agent). Definiuje Karty Agenta (odkrywanie), zadania z artefaktami (tekst, dane strukturalne, wideo), nieprzezroczyste cykle życia zadań i uwierzytelnianie. Systemy produkcyjne coraz częściej łączą MCP z A2A. Google Cloud wdrożyło obsługę A2A do Vertex AI Agent Builder w latach 2025-2026.

**Type:** Learn + Build
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Problem

Twój agent musi wywołać innego agenta w innym systemie. Jak? Możesz wystawić endpoint HTTP, zdefiniować niestandardowy schemat JSON i mieć nadzieję, że druga strona go rozumie. Każda para agentów staje się niestandardową integracją.

A2A to uniwersalny protokół przesyłania dla takiego wywołania. Standardowe odkrywanie, standardowy model zadania, standardowy transport, standardowe artefakty. Jak HTTP+REST, ale dla agentów jako obywateli pierwszej klasy.

## Koncepcja

### Cztery elementy

**Karta Agenta.** Dokument JSON pod `/.well-known/agent.json` opisujący agenta: nazwa, umiejętności, endpointy, obsługiwane modalności, wymagania uwierzytelniania. Odkrywanie odbywa się przez odczyt karty.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Zadanie.** Jednostka pracy. Asynchroniczny, stanowy obiekt z cyklem życia: `submitted → working → completed / failed / canceled`. Klient wysyła zadanie, sonduje lub subskrybuje aktualizacje.

**Artefakt.** Typ wyniku produkowanego przez zadanie. Tekst, strukturalny JSON, obraz, wideo, audio. Artefakty są typowane, więc różne modalności są obywatelami pierwszej klasy.

**Nieprzezroczysty cykl życia.** A2A nie narzuca *jak* zdalny agent rozwiązuje zadanie. Klient widzi przejścia stanów i artefakty; implementacja może używać dowolnego frameworka.

### Podział MCP/A2A

- **MCP** (Lekcja 13): agent ↔ narzędzie. Agent czyta/zapisuje przez JSON-RPC do serwera narzędzi. Bezstanowy domyślnie.
- **A2A**: agent ↔ agent. Protokół równorzędny; obie strony są agentami z własnym rozumowaniem.

Produkcyjne systemy wieloagentowe używają obu. Uczestnik A2A wywołuje narzędzia MCP po swojej stronie. Podział utrzymuje czystość dwóch zagadnień.

### Przepływ odkrywania

```
Klient                     Serwer agenta
  ├──GET /.well-known/agent.json──>
  <──JSON Karty Agenta───────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% ukończono
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

Lub ze strumieniowaniem: subskrypcja SSE do `/tasks/{id}/events` dla aktualizacji push.

### Uwierzytelnianie

A2A wspiera trzy popularne wzorce:

- **Token Bearer** — OAuth2 lub nieprzezroczysty.
- **mTLS** — wzajemne TLS; organizacje udowodniają sobie tożsamość.
- **Podpisane żądania** — HMAC na ładunku.

Uwierzytelnianie jest deklarowane w Karcie Agenta; klienci odkrywają i stosują się.

### 150+ organizacji do kwietnia 2026

Przyjęcie przez przedsiębiorstwa napędzało skalę A2A. Najważniejsze: A2A stało się sposobem, w jaki systemy agentowe przedsiębiorstw przekraczają granice zaufania. Google Cloud wydało Vertex AI Agent Builder z obsługą A2A; Microsoft Agent Framework je wspiera; większość głównych frameworków (LangGraph, CrewAI, AutoGen) dostarcza adaptery A2A.

### Gdzie A2A wygrywa

- **Międzyorganizacyjne wywołania.** Agent w firmie A wywołuje agenta w firmie B. Bez A2A każda para to niestandardowy kontrakt.
- **Heterogeniczne frameworki.** Agent LangGraph wywołuje agenta CrewAI wywołuje niestandardowego agenta Python. A2A normalizuje.
- **Typowane artefakty.** Wynik wideo, strukturalny JSON, audio — wszystkie pierwszej klasy.
- **Długotrwałe zadania.** Nieprzezroczysty cykl życia + sondowanie sprawia, że zadania trwające godziny są proste.

### Gdzie A2A ma trudności

- **Wrażliwe na opóźnienia mikro-wywołania.** Cykl życia A2A jest asynchroniczny. Sub-milisekundowe agent-do-agenta nie pasuje; użyj bezpośredniego RPC.
- **Ściśle powiązani agenci w procesie.** Jeśli obaj agenci działają w tym samym procesie Pythona, HTTP round-trip A2A jest przesadą.
- **Małe zespoły.** Narzut specyfikacji jest realny; agenci wewnętrzni mogą nie potrzebować formalności.

### A2A a ACP, ANP, NLIP

Kilka powiązanych specyfikacji pojawiło się w latach 2024-2026:

- **ACP** (IBM/Linux Foundation) — poprzednik A2A, węższy zakres.
- **ANP** (Agent Network Protocol) — skoncentrowany na odkrywaniu równorzędnych, zdecentralizowany.
- **NLIP** (Ecma Natural Language Interaction Protocol, standaryzowany grudzień 2025) — typ treści w języku naturalnym.

A2A jest najszerzej przyjętym protokołem równorzędnym od kwietnia 2026. Zobacz arXiv:2505.02279 (Liu i in., "A Survey of Agent Interoperability Protocols") dla porównania.

## Zbuduj To

`code/main.py` implementuje minimalny serwer i klient A2A używając `http.server` i JSON. Serwer:

- udostępnia `/.well-known/agent.json`,
- akceptuje `POST /tasks`,
- zarządza stanem zadania,
- zwraca artefakty na `GET /tasks/{id}`.

Klient:

- pobiera Kartę Agenta,
- wysyła zadanie,
- sonduje aż do ukończenia,
- odczytuje artefakt.

Uruchom:

```
python3 code/main.py
```

Skrypt uruchamia serwer w wątku tła, a następnie uruchamia klienta względem niego. Zobaczysz pełny przepływ: odkrywanie, wysłanie, sondowanie, artefakt.

## Użyj Tego

`outputs/skill-a2a-integrator.md` projektuje integrację A2A: zawartość Karty Agenta, schematy zadań, wybór uwierzytelniania, strumieniowanie vs sondowanie.

## Wdróż To

Lista kontrolna:

- **Przypnij wersję specyfikacji.** A2A wciąż ewoluuje; Karta Agenta powinna deklarować wersję protokołu.
- **Idempotentne tworzenie zadań.** Zduplikowane wysłania (próby sieciowe) powinny produkować jedno zadanie.
- **Schematy artefaktów.** Zadeklaruj, jakie kształty zwraca agent; konsumenci powinni walidować.
- **Limity szybkości + uwierzytelnianie.** A2A jest publiczny; stosuj standardowe zabezpieczenia webowe.
- **Martwa skrzynka dla nieudanych zadań.** Inspekcjonuj wzorce w czasie dla powtarzających się typów awarii.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że klient odkrywa serwer i otrzymuje poprawny artefakt.
2. Dodaj drugą umiejętność do serwera (np. "summarize"). Zaktualizuj Kartę Agenta. Napisz klienta, który wybiera umiejętność na podstawie typu zadania.
3. Zaimplementuj endpoint strumieniowania SSE: `/tasks/{id}/events` emitujący zmiany stanu. Co klient musi robić inaczej?
4. Przeczytaj specyfikację A2A (https://a2a-protocol.org/latest/specification/). Zidentyfikuj trzy rzeczy, które specyfikacja nakazuje, a których to demo nie implementuje.
5. Porównaj A2A (odkrywanie przez Kartę Agenta) z MCP (listowanie możliwości po stronie serwera przez `listTools`). Jaki jest kompromis między samoopisującymi się agentami a badaniem możliwości?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|------|----------------|------------------------|
| A2A | "Agent-do-agenta" | Protokół równorzędny dla agentów do wywoływania innych agentów między systemami. Google 2025. |
| Karta Agenta | "Wizytówka agenta" | JSON pod `/.well-known/agent.json` opisujący umiejętności, endpointy, uwierzytelnianie. |
| Zadanie | "Jednostka pracy" | Asynchroniczny stanowy obiekt z cyklem życia; artefakty produkowane po ukończeniu. |
| Artefakt | "Wynik" | Typowane wyjście: tekst, strukturalny JSON, obraz, wideo, audio. Multimedia pierwszej klasy. |
| Nieprzezroczysty cykl życia | "Jak jest rozwiązane to sprawa agenta" | Klient widzi przejścia stanów; serwer może swobodnie wybrać framework/narzędzia. |
| Odkrywanie | "Znajdowanie agenta" | `GET /.well-known/agent.json` zwraca kartę. |
| MCP vs A2A | "Narzędzia vs równorzędni" | MCP: pionowy agent ↔ narzędzie. A2A: poziomy agent ↔ agent. |
| ACP / ANP / NLIP | "Protokoły siostrzane" | Sąsiednie specyfikacje; A2A jest najszerzej przyjęte w 2026. |

## Dalsza Literatura

- [Specyfikacja A2A](https://a2a-protocol.org/latest/specification/) — kanoniczna specyfikacja
- [Blog Google Developers — Ogłoszenie A2A](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — post uruchomieniowy z kwietnia 2025
- [Repozytorium A2A GitHub](https://github.com/a2aproject/A2A) — referencyjne implementacje i SDK
- [Liu i in. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — porównanie MCP, ACP, A2A, ANP