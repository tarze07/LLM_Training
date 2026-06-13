# Capstone — Zbuduj Kompletny Ekosystem Narzędzi

> Faza 13 nauczyła każdego elementu. Ten capstone łączy je w jeden system o kształcie produkcyjnym: serwer MCP z narzędziami + zasobami + promptami + zadaniami + UI, OAuth 2.1 na brzegu, bramka RBAC, klient wielu serwerów, wywołanie pod-agenta A2A, śledzenie OTel w kolektorze, wykrywanie zatrucia narzędzi w CI oraz pakiet AGENTS.md + SKILL.md. Pod koniec będziesz w stanie obronić każdy wybór architektoniczny.

**Type:** Build
**Languages:** Python (stdlib, end-to-end ecosystem harness)
**Prerequisites:** Phase 13 · 01 through 21
**Time:** ~120 minutes

## Learning Objectives

- Złożyć serwer MCP udostępniający narzędzia, zasoby, prompty i zadanie z aplikacją `ui://`.
- Umieścić przed serwerem bramkę OAuth 2.1, która egzekwuje RBAC i przypięte skróty.
- Napisać klienta wielu serwerów, który śledzi z atrybutami OTel GenAI od końca do końca.
- Delegować część obciążenia do pod-agenta A2A; zweryfikować zachowanie nieprzezroczystości.
- Zapakować cały stos z AGENTS.md + SKILL.md, aby inni agenci mogli go obsługiwać.

## The Problem

Dostarcz system "badaj i raportuj":

- Użytkownik pyta: "podsumuj trzy najczęściej cytowane artykuły z arXiv z 2026 roku na temat protokołów agentów."
- System: przeszukaj arXiv przez MCP; deleguj podsumowanie artykułów do wyspecjalizowanego agenta piszącego przez A2A; agreguj wyniki; wyrenderuj interaktywny raport jako zasób MCP Apps `ui://`; rejestruj każdy krok w OTel.

Wszystkie prymitywy z Fazy 13 pojawiają się. To nie jest zabawka — produkcyjne systemy asystentów badawczych dostarczone w 2026 roku przez Anthropic (produkt Claude Research), OpenAI (GPTs z Apps SDK) i strony trzecie mają dokładnie taki kształt.

## The Concept

### Architektura

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                       |
                                                       +- MCP tool: arxiv_search (pure)
                                                       +- MCP resource: notes://recent
                                                       +- MCP prompt: /research_topic
                                                       +- MCP task: generate_report (long)
                                                       +- MCP Apps UI: ui://report/current
                                                       +- A2A call: writer-agent (tasks/send)
                                                       |
                                                       +- OTel GenAI spans
```

### Hierarchia śladów

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

Jeden identyfikator śladu. Każdy span ma odpowiednie atrybuty `gen_ai.*`.

### Postawa bezpieczeństwa

- OAuth 2.1 + PKCE z przypięciem wskaźnika zasobu do bramki.
- Bramka przechowuje poświadczenia nadrzędne; użytkownik nigdy ich nie widzi.
- RBAC: `alice` ma `research:read`, `research:write`, może wywoływać wszystkie narzędzia. `bob` ma `research:read`, nie może wywoływać `generate_report`.
- Przypięty manifest opisów: odrzuć każdy serwer, którego skróty narzędzi się zmieniły.
- Audyt Zasady Dwóch: żadne narzędzie nie łączy niezaufanego wejścia, wrażliwych danych i działania o konsekwencjach.

### Renderowanie

Końcowe zadanie `generate_report` zwraca bloki treści plus zasób `ui://report/current`. Host klienta (Claude Desktop itp.) renderuje interaktywny dashboard w sandboksowym iframe. Dashboard zawiera posortowaną listę artykułów, liczby cytowań i przycisk, który wywołuje `host.callTool('summarize_paper', {arxiv_id})` dla dowolnego artykułu klikniętego przez użytkownika.

### Pakowanie

Całość jest dostarczana jako:

```
research-system/
  AGENTS.md                     # konwencje projektu
  skills/
    run-research/
      SKILL.md                  # przepływ pracy najwyższego poziomu
  servers/
    research-mcp/               # serwer MCP
      pyproject.toml
      src/
  agents/
    writer/                     # agent A2A
  gateway/
    config.yaml                 # RBAC + przypięty manifest
```

Użytkownicy wdrażają przez `docker compose up`. Użytkownicy Claude Code, Cursor, Codex i opencode mogą obsługiwać system, wywołując umiejętność `run-research`.

### Wkład każdej lekcji Fazy 13

| Lekcja | Co capstone wykorzystuje |
|--------|------------------------|
| 01-05 | Interfejs narzędzi, przenośność między dostawcami, wywołania równoległe, schematy, linting |
| 06-10 | Prymitywy MCP, serwer, klient, transporty, zasoby + prompty |
| 11-14 | Próbkowanie, korzenie + elcytagia, zadania asynchroniczne, aplikacje `ui://` |
| 15-17 | Zatrucie narzędzi, OAuth 2.1, bramka + rejestr |
| 18 | Delegacja pod-agenta A2A |
| 19 | Śledzenie OTel GenAI |
| 20 | Bramka rutingu dla warstwy LLM |
| 21 | Pakowanie SKILL.md + AGENTS.md |

## Use It

`code/main.py` łączy wzorce poprzednich lekcji w jeden uruchamialny demo. Wszystko w stdlib, wszystko w procesie, abyś mógł przeczytać wszystko od początku do końca. Uruchamia pełny przepływ dla scenariusza badaj-i-raportuj: uzgadnianie z bramką, symulowany OAuth 2.1, scalone tools/list, `generate_report` jako zadanie, wywołanie A2A do piszącego, zwrócony zasób `ui://`, wyemitowane spany OTel.

Na co zwrócić uwagę:

- Jeden identyfikator śladu przez każdy skok.
- Polityka bramki blokuje drugiego użytkownika przed zapisem.
- Cykl życia zadania przechodzi working → completed i zwraca zarówno tekst, jak i treść `ui://`.
- Stan wewnętrzny wywołania A2A jest nieprzezroczysty dla orkiestratora.
- AGENTS.md i SKILL.md to jedyne pliki, których inny agent potrzebuje, aby odtworzyć przepływ pracy.

## Ship It

Ta lekcja produkuje `outputs/skill-ecosystem-blueprint.md`. Mając potrzebę produktową (badania, podsumowywanie, automatyzacja), umiejętność produkuje pełną architekturę: które prymitywy MCP, które mechanizmy kontroli bramki, które wywołania A2A, którą telemetrię, które pakowanie.

## Exercises

1. Uruchom `code/main.py`. Zwróć uwagę na pojedynczy identyfikator śladu i jak spany się zagnieżdżają. Policz, ile prymitywów z Fazy 13 demo dotyka.

2. Rozszerz demo: dodaj drugi backendowy serwer MCP (np. `bibliography`) i potwierdź, że bramka scala jego narzędzia w tę samą przestrzeń nazw.

3. Zastąp fałszywego agenta piszącego A2A prawdziwym działającym w podprocesie. Użyj narzędzia z Lekcji 19.

4. Dodaj krok maskowania PII w bramce rutingu między orkiestratorem a LLM. Potwierdź, że adresy e-mail w zapytaniu użytkownika są usuwane.

5. Napisz AGENTS.md dla członka zespołu, który będzie utrzymywać ten system. Powinien zająć mniej niż pięć minut do przeczytania i dać im wszystko, czego potrzebują do obsługi capstone w Cursor lub Codex.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Capstone | "Demo integracyjne Fazy 13" | System end-to-end wykorzystujący każdy prymityw |
| Research and report | "Scenariusz" | Wzorzec: szukaj, podsumuj, wyrenderuj |
| Ecosystem | "Wszystkie elementy razem" | Serwer + klient + bramka + pod-agent + telemetria + pakiet |
| Trace hierarchy | "Pojedynczy identyfikator śladu" | Span każdego skoku współdzieli ślad; rodzic-dziecko przez identyfikatory spanów |
| Gateway-issued token | "Uwierzytelnianie przechodnie" | Klient widzi tylko token bramki; bramka przechowuje poświadczenia nadrzędne |
| Merged namespace | "Wszystkie narzędzia w jednej płaskiej liście" | Scalanie wielu serwerów w bramce, prefiks przy kolizji |
| Opacity boundary | "Wywołanie A2A ukrywa wnętrze" | Rozumowanie pod-agenta niewidoczne dla orkiestratora |
| Three-layer stack | "AGENTS.md + SKILL.md + MCP" | Kontekst projektu + przepływ pracy + narzędzia |
| Defense-in-depth | "Wiele warstw bezpieczeństwa" | Przypięte skróty, OAuth, RBAC, Zasada Dwóch, dziennik audytu |
| Spec compliance matrix | "Co dostarczamy, czego wymaga specyfikacja" | Lista kontrolna mapująca rezultaty na wymagania z 2025-11-25 |

## Further Reading

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — skonsolidowany podręcznik
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — kierunek rozwoju protokołu
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — dokumentacja A2A v1.0
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — kanoniczne konwencje śledzenia
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — wzorce produkcyjnego środowiska agenta