# Capstone 13 — Serwer MCP z Rejestrem i Zarządzaniem

> Model Context Protocol przestał być przyszłością i stał się domyślnym specyfikacją użycia narzędzi w 2026 roku. Anthropic, OpenAI, Google i każde główne IDE dostarczają klientów MCP. Pinterest opublikował swój wewnętrzny ekosystem serwerów MCP. Rejestr AAIF sformalizował metadane możliwości w `.well-known`. AWS ECS opublikował referencyjne wdrożenie bezstanowe. goose-agent Blocka umieścił ten sam protokół w hostowanym asystencie. Produkcyjny kształt 2026 to: transport StreamableHTTP, zakresy OAuth 2.1, bramkowanie polityki OPA i rejestr, który pozwala zespołom platformowym odkrywać, walidować i włączać serwery. Zbuduj to od końca do końca.

**Type:** Capstone
**Languages:** Python (server, via FastMCP) or TypeScript (@modelcontextprotocol/sdk), Go (registry service)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P14 · P17 · P18
**Time:** 25 hours

## Problem

MCP stał się lingua franca użycia narzędzi. Claude Code, Cursor 3, Amp, OpenCode, Gemini CLI i każdy zarządzany agent teraz konsumuje serwery MCP. Wyzwania produkcyjne nie dotyczą tworzenia serwerów (FastMCP to ułatwia), ale wdrażania ich na skalę z wymaganiami korporacyjnymi: zakresy OAuth na dzierżawcę, polityka OPA na destrukcyjnych narzędziach, bezstanowe skalowanie StreamableHTTP, rejestr do odkrywania, logi audytu na wywołanie narzędzia. Wewnętrzny ekosystem MCP Pinteresta i specyfikacja Rejestru AAIF wyznaczają standard 2026.

Zbudujesz serwer MCP udostępniający 10 wewnętrznych narzędzi (Postgres tylko do odczytu, listowanie S3, Jira, Linear, Datadog, itd.), rejestr UI do odkrywania przez platformę i bramkę zatwierdzania przez człowieka dla destrukcyjnych narzędzi. Test obciążenia demonstruje horyzontalne skalowanie StreamableHTTP. Ślad audytu spełnia wymogi korporacyjnego przeglądu bezpieczeństwa.

## Koncepcja

Rewizja MCP 2026 nakazuje StreamableHTTP jako domyślny transport. W przeciwieństwie do wcześniejszego kształtu stdio-i-SSE, StreamableHTTP jest domyślnie bezstanowy: pojedynczy endpoint HTTP akceptuje żądania JSON-RPC, strumieniuje odpowiedzi i obsługuje długożyciowe połączenia dla powiadomień. Bezstanowość oznacza horyzontalną skalowalność za load balancerem.

Autoryzacja to OAuth 2.1 z zakresami na narzędzie. Token niesie zakresy takie jak `jira:read`, `s3:list`, `postgres:query:readonly`. Serwer MCP sprawdza zakresy w czasie wywołania narzędzia, nie tylko na starcie sesji. Dla wysokiego ryzyka narzędzi, serwer odrzuca każde wywołanie, którego zakres nie został podniesiony do `approved:by:human` w ciągu ostatnich N minut — to podniesienie pochodzi z karty recenzji Slack.

Rejestr to osobny serwis. Każdy serwer MCP udostępnia dokument `.well-known/mcp-capabilities` z manifestem narzędzi, URL transportu, wymaganiami auth. Rejestr sonduje, waliduje i indeksuje. Zespoły platformowe używają UI rejestru, aby zobaczyć, jakie narzędzia są dostępne, jakie zakresy są potrzebne i które zespoły je posiadają.

## Architektura

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## Stack

- Framework serwera: FastMCP (Python) lub `@modelcontextprotocol/sdk` (TypeScript)
- Transport: StreamableHTTP przez HTTPS (bezstanowy)
- Auth: OAuth 2.1 z tożsamością obciążenia przez SPIFFE / SPIRE
- Polityka: OPA / Rego reguły na narzędzie; serwis decyzji politycznych na żądanie
- Rejestr: samodzielnie hostowany, konsumuje manifesty `.well-known/mcp-capabilities`
- Zatwierdzanie przez człowieka: interaktywna wiadomość Slack dla destrukcyjnych narzędzi
- Wdrożenie: AWS ECS Fargate lub Fly.io, jeden serwer na dzierżawcę lub współdzielony z zakresowaniem dzierżawcy
- Audyt: strukturalny JSONL na dzierżawcę z rodowodem na wywołanie

## Build It

1. **Powierzchnia narzędzi.** Udostępnij 10 wewnętrznych narzędzi: Postgres read-only query, S3 list objects, Jira search/fetch, Linear search/fetch, Datadog metric query, PagerDuty on-call lookup, GitHub read-only, Notion search, Slack search, Salesforce read. Każde narzędzie ma typowany schemat i etykietę zakresu.

2. **Serwer FastMCP.** Zamontuj narzędzia. Skonfiguruj transport StreamableHTTP. Dodaj middleware do introspekcji tokena OAuth i egzekwowania zakresu.

3. **Polityka OPA.** Reguły Rego na narzędzie: jakie zakresy zezwalają na wywołanie, jakie maskowanie PII ma zastosowanie, jakie ograniczenia rozmiaru ładunku. Serwis decyzji wywoływany przy każdym wywołaniu narzędzia.

4. **Serwis rejestru.** Osobny serwis Go lub TS, który sonduje `.well-known/mcp-capabilities` z zarejestrowanych serwerów, waliduje z JSON Schema i udostępnia UI do listowania / wyszukiwania / walidacji / włączania-wyłączania.

5. **Manifest możliwości.** Każdy serwer udostępnia `.well-known/mcp-capabilities` z: listą narzędzi, wymaganiami auth, URL transportu, zespołem właścicielskim, SLO.

6. **Separacja destrukcyjnych narzędzi.** Narzędzia, które mutują stan (Jira create, Linear create, Postgres write), znajdują się na drugim serwerze MCP z bardziej rygorystycznym przepływem auth: tokeny muszą mieć zakres `approved:by:human` podniesiony przez kartę Slack w ciągu 15 minut.

7. **Log audytu.** JSONL tylko do dołączania na dzierżawcę: `{timestamp, user, tool, args_redacted, response_redacted, outcome}`. Maskowanie PII przez Presidio przed zapisem.

8. **Test obciążenia.** 100 równoczesnych klientów na StreamableHTTP. Zademonstruj horyzontalne skalowanie przez dodanie drugiej repliki; pokaż, jak load balancer redystrybuuje bez lepkości sesji.

9. **Testy zgodności.** Uruchom oficjalny zestaw testów zgodności MCP przeciwko obu serwerom. Przejdź wszystkie obowiązkowe sekcje.

## Use It

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## Ship It

`outputs/skill-mcp-server.md` opisuje rezultat. Produkcyjny serwer MCP + rejestr + warstwa audytu dla wewnętrznych narzędzi z zakresami OAuth 2.1 i bramkowaniem OPA.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Zgodność ze specyfikacją | StreamableHTTP + manifest możliwości przechodzi testy zgodności MCP |
| 20 | Bezpieczeństwo | Egzekwowanie zakresów, pokrycie OPA na każdym narzędziu, higiena sekretów |
| 20 | Obserwowalność | Log audytu na wywołanie narzędzia z maskowaniem PII |
| 20 | Skala | Demonstracja horyzontalnego skalowania w teście obciążenia 100 klientów |
| 15 | UX rejestru | Przepływ odkrywania / walidacji / włączania-wyłączania |
| **100** | | |

## Ćwiczenia

1. Dodaj nowe narzędzie (Confluence search). Dostarcz je przez przepływ walidacji rejestru bez dotykania podstawowego serwera.

2. Napisz politykę OPA, która maskuje wyniki zapytań Postgres zawierające kolumny o nazwie `email`, `ssn` lub `phone`. Przećwicz z zapytaniem próbnym.

3. Porównaj benchmark StreamableHTTP vs stdio na lokalnym opóźnieniu. Raportuj p50/p95 na wywołanie.

4. Zaimplementuj limit na dzierżawcę: maksymalnie N wywołań na minutę na narzędzie na dzierżawcę. Egzekwuj przez drugą regułę OPA.

5. Uruchom zestaw testów zgodności MCP z [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) i napraw każdą awarię.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| StreamableHTTP | "Transport MCP 2026" | Bezstanowe HTTP + strumieniowanie; zastępuje SSE + stdio dla serwerów sieciowych |
| Capability manifest | "Dokument Well-known" | `.well-known/mcp-capabilities` z listą narzędzi, auth, URL transportu |
| OPA / Rego | "Silnik polityki" | Open Policy Agent do autoryzacji wywołań narzędzi według zewnętrznych reguł |
| Scope elevation | "Zatwierdzone-przez-człowieka" | Krótkożyciowy zakres przyznany przez zatwierdzenie Slack, wymagany dla destrukcyjnych narzędzi |
| Registry | "Odkrywanie narzędzi" | Serwis indeksujący serwery MCP z ich manifestów możliwości |
| Workload identity | "SPIFFE / SPIRE" | Kryptograficzna tożsamość serwisu do wydawania tokenów OAuth |
| Conformance suite | "Testy specyfikacji" | Oficjalna bateria testów MCP dla StreamableHTTP + poprawności manifestu narzędzi |

## Dalsza lektura

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, capability metadata, registry
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — the 2026 registry spec
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — reference production deployment
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — the reference internal deployment
- [Block `goose` MCP usage](https://block.github.io/goose/) — reference agent consumption pattern
- [FastMCP](https://github.com/jlowin/fastmcp) — Python server framework
- [Open Policy Agent](https://www.openpolicyagent.org/) — policy engine reference
- [SPIFFE / SPIRE](https://spiffe.io) — workload identity reference