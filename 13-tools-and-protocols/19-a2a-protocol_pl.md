# A2A — Protokół Agent-do-Agenta

> MCP to agent-do-narzędzia. A2A (Agent2Agent) to agent-do-agenta — otwarty protokół umożliwiający współpracę nieprzezroczystych agentów zbudowanych na różnych frameworkach. Wydany przez Google w kwietniu 2025, przekazany Linux Foundation w czerwcu 2025, osiągnięcie v1.0 w kwietniu 2026 z 150+ wspierającymi organizacjami, w tym AWS, Cisco, Microsoft, Salesforce, SAP i ServiceNow. Wchłonął IBM ACP i dodał rozszerzenie płatności AP2. Ta lekcja omawia Agent Card, cykl życia Task i dwa wiązania transportowe.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (podstawy MCP), Phase 13 · 08 (klient MCP)
**Time:** ~75 minut

## Learning Objectives

- Odróżnić przypadki użycia agent-do-narzędzia (MCP) od agent-do-agenta (A2A).
- Opublikować Agent Card pod `/.well-known/agent.json` z umiejętnościami i metadanymi endpointu.
- Przejść przez cykl życia Task (submitted → working → input-required → completed / failed / canceled / rejected).
- Używać Messages z Parts (text, file, data) i Artifacts jako wyników.

## Problem

Agent obsługi klienta musi zlecić pisanie raportu wyspecjalizowanemu agentowi pisarzowi. Opcje przed A2A:

- Niestandardowe REST API. Działa, ale każda para to jednorazowe rozwiązanie.
- Współdzielona baza kodu. Wymaga, aby obaj agenci uruchamiali ten sam framework.
- MCP. Nie pasuje: MCP służy do wywoływania narzędzi, a nie do współpracy dwóch agentów przy zachowaniu nieprzezroczystego wewnętrznego rozumowania każdego z nich.

A2A wypełnia tę lukę. Modeluje interakcję jako jeden agent wysyłający Task do drugiego, z cyklem życia, wiadomościami i artefaktami. Wewnętrzny stan wywoływanego agenta pozostaje nieprzezroczysty — wywołujący widzi tylko przejścia stanów zadania i końcowe wyniki.

A2A to protokół "pozwalający agentom z różnych frameworków rozmawiać ze sobą". Nie zastępuje MCP; oba są komplementarne.

## Koncepcja

### Agent Card

Każdy agent zgodny z A2A publikuje kartę pod `/.well-known/agent.json`:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Odkrywanie opiera się na URL: pobierz kartę, poznaj URL endpointu A2A, wylicz umiejętności.

### Podpisane Agent Cards (AP2)

Rozszerzenie AP2 (wrzesień 2025) dodaje podpisy kryptograficzne do Agent Cards. Wydawca podpisuje własną kartę JWT-em; konsumenci weryfikują. Zapobiega podszywaniu się.

### Cykl życia Task

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (pętla przez message)
```

Klienci inicjują przez `tasks/send`. Wywoływany agent przechodzi przez stany; klienci subskrybują aktualizacje stanu przez SSE lub poll.

### Messages i Parts

Wiadomość niesie jeden lub więcej Parts:

- `text` — zwykła treść.
- `file` — blob base64 z mimeType.
- `data` — typowany JSON (strukturalne wejście dla wywoływanego agenta).

Przykład:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

Wyniki to Artifacts, nie surowe ciągi znaków. Artifact to nazwany, typowany wynik:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts mogą być strumieniowane jako fragmenty. Wywołujący akumuluje.

### Dwa wiązania transportowe

1. **JSON-RPC przez HTTP.** Endpoint `/a2a`, POST dla żądań, opcjonalne SSE dla strumieniowania. Domyślne wiązanie.
2. **gRPC.** Dla środowisk korporacyjnych, gdzie gRPC jest natywny.

Oba wiązania przenoszą ten sam logiczny kształt wiadomości.

### Zachowanie nieprzezroczystości

Kluczowa zasada projektowa: wewnętrzny stan wywoływanego agenta jest nieprzezroczysty. Wywołujący widzi stan zadania i artefakty. Łańcuch myślowy wywoływanego agenta, jego wywołania narzędzi, delegacja do pod-agentów — wszystko niewidoczne. Różni się to od MCP, gdzie wywołania narzędzi są przezroczyste.

Uzasadnienie: A2A umożliwia konkurentom współpracę bez ujawniania wewnętrznych mechanizmów. A2A może być "wywołaj tego agenta obsługi klienta" bez uczenia się przez wywołującego, jak ten agent implementuje obsługę.

### Harmonogram

- **2025-04-09.** Google ogłasza A2A.
- **2025-06-23.** Przekazane Linux Foundation.
- **2025-08.** Wchłania IBM ACP.
- **2025-09.** Rozszerzenie AP2 (Agent Payments) wydane.
- **2026-04.** v1.0 wydane ze 150+ wspierającymi organizacjami.

### Relacja z MCP

| Wymiar | MCP | A2A |
|-----------|-----|-----|
| Przypadek użycia | Agent-do-narzędzia | Agent-do-agenta |
| Nieprzezroczystość | Przezroczyste wywołania narzędzi | Nieprzezroczyste wewnętrzne rozumowanie |
| Typowy wywołujący | Środowisko uruchomieniowe agenta | Inny agent |
| Stan | Wynik wywołania narzędzia | Task z cyklem życia |
| Autoryzacja | OAuth 2.1 (Phase 13 · 16) | JWT-podpisane Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC przez HTTP / gRPC |

Użyj MCP, gdy chcesz wywołać konkretne narzędzie. Użyj A2A, gdy chcesz zlecić całe zadanie innemu agentowi. Wiele systemów produkcyjnych używa obu: agent używa MCP dla swojej warstwy narzędziowej i A2A dla swojej warstwy współpracy.

## Użyj tego

`code/main.py` implementuje minimalny harness A2A: agent badawczy publikuje swoją kartę, agent pisarz otrzymuje `tasks/send` z częściami zawierającymi PDF i instrukcję tekstową, przechodzi przez working → input_required → working → completed i zwraca artefakt tekstowy. Wszystko w stdlib; używa transportu w pamięci, aby skupić się na kształtach wiadomości.

Na co zwrócić uwagę:

- Kształt JSON Agent Card.
- Przypisanie id Task i przejścia stanów.
- Wiadomości z częściami mieszanych typów.
- Gałąź input-required w środku zadania.
- Zwrot Artifact po zakończeniu.

## Ship It

Ta lekcja produkuje `outputs/skill-a2a-agent-spec.md`. Dla nowego agenta, który powinien być wywoływalny przez innych agentów, skill produkuje JSON Agent Card, schemat umiejętności i blueprint endpointu.

## Ćwiczenia

1. Uruchom `code/main.py`. Prześledź pełny cykl życia Task, włączając pauzę input-required, gdzie wywoływany agent prosi o wyjaśnienie.

2. Dodaj podpisaną Agent Card. Podpisz HMAC-em na kanonicznym JSON karty. Napisz weryfikator i potwierdź, że zawodzi na zmutowanej karcie.

3. Zaimplementuj strumieniowanie Task: agent pisarz emituje trzy przyrostowe fragmenty artefaktu przez SSE, a wywołujący je akumuluje.

4. Zaprojektuj agenta A2A, który opakowuje serwer MCP. Mapuj każde narzędzie MCP na umiejętność A2A. Zwróć uwagę na kompromisy — jaka nieprzezroczystość jest tracona?

5. Przeczytaj ogłoszenie A2A v1.0 i zidentyfikuj jedną funkcję, która nie jest jeszcze zaimplementowana przez żaden framework według kwietnia 2026. (Wskazówka: dotyczy delegacji Task wieloskokowej.)

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| A2A | "Protokół Agent-do-Agenta" | Otwarty protokół dla nieprzezroczystej współpracy agentów |
| Agent Card | "`.well-known/agent.json`" | Opublikowane metadane opisujące umiejętności i endpoint agenta |
| Skill | "Jednostka wywoływalna" | Nazwana operacja obsługiwana przez agenta (analog do narzędzia MCP) |
| Task | "Jednostka delegacji" | Element pracy z cyklem życia i końcowym artefaktem |
| Message | "Wejście Task" | Nosi Parts (text, file, data) |
| Part | "Typowany fragment" | Element `text` / `file` / `data` wiadomości |
| Artifact | "Wyjście Task" | Nazwany, typowany wynik zwracany po zakończeniu |
| AP2 | "Protokół Płatności Agentów" | Rozszerzenie podpisanych Agent Cards dla zaufania i płatności |
| Nieprzezroczystość | "Współpraca czarnej skrzynki" | Wewnętrzne mechanizmy wywoływanego agenta są ukryte przed wywołującym |
| Input-required | "Pauza Task" | Stan cyklu życia, gdy agent potrzebuje więcej informacji |

## Dalsza lektura

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — kanoniczna specyfikacja A2A
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — referencyjne implementacje i SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — transfer zarządzania z czerwca 2025
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — plan działania i dynamika partnerów
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — notatki do wydania v1.0 i wskazówki dotyczące kompatybilności wstecznej
