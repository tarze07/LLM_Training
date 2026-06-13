# Umiejętności i SDK Agentów — Anthropic Skills, AGENTS.md, OpenAI Apps SDK

> MCP mówi "jakie narzędzia istnieją." Umiejętności mówią "jak wykonać zadanie." Stos z 2026 roku łączy oba. Umiejętności Agentów Anthropic (otwarty standard, grudzień 2025) są dostarczane jako SKILL.md z progresywnym ujawnianiem. SDK Apps od OpenAI to MCP plus metadane widżetów. AGENTS.md (obecnie w ponad 60 000 repozytoriów) znajduje się w katalogu głównym repozytorium jako kontekst agenta na poziomie projektu. Ta lekcja wyjaśnia, co każde pokrywa i buduje minimalny zestaw SKILL.md + AGENTS.md, który działa u różnych agentów.

**Type:** Learn
**Languages:** Python (stdlib, parser i ładowarka SKILL.md)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## Learning Objectives

- Rozróżnić trzy warstwy: AGENTS.md (kontekst projektu), SKILL.md (wielokrotnego użytku know-how), MCP (narzędzia).
- Napisać SKILL.md z frontmatterem YAML i progresywnym ujawnianiem.
- Załadować pliki umiejętności w stylu systemu plików do środowiska uruchomieniowego agenta.
- Połączyć umiejętność z serwerem MCP i AGENTS.md, aby jeden pakiet działał w Claude Code, Cursor i Codex.

## The Problem

Inżynier destyluje przepływ pracy pisania notatek wydania w wieloetapowy prompt: "Przeczytaj ostatnio scalone PRy. Pogrupuj według obszaru. Podsumuj każdy. Napisz wpis changeloga zgodny ze stylem zespołu. Opublikuj w wersji roboczej na Slacku." Umieszczają to w dokumencie Notion dla swojego zespołu.

Teraz chcą używać tego przepływu pracy z Claude Code, Cursor i Codex CLI. Każdy agent ma inny sposób ładowania instrukcji: polecenia slash w Claude Code, reguły Cursor, `.codex.md` w Codex. Inżynier kopiuje przepływ pracy trzy razy i utrzymuje trzy kopie.

AGENTS.md i SKILL.md razem to naprawiają:

- **AGENTS.md** znajduje się w katalogu głównym repozytorium. Każdy kompatybilny agent czyta go na starcie sesji. "Jak działa ten projekt? Jakie są konwencje? Które polecenia uruchamiają testy?"
- **SKILL.md** to przenośny pakiet: frontmatter YAML (nazwa, opis) + treść w markdown + opcjonalne zasoby. Agenci obsługujący umiejętności ładują je po nazwie na żądanie.
- **MCP** (Faza 13 · 06-14) obsługuje narzędzia, które umiejętność musi wywołać.

Trzy warstwy, jeden przenośny artefakt.

## The Concept

### AGENTS.md (agents.md)

Uruchomiony pod koniec 2025, przyjęty przez ponad 60 000 repozytoriów do kwietnia 2026. Jeden plik w katalogu głównym repozytorium. Format:

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agenci czytają to na starcie sesji i używają do kalibracji swojego zachowania dla tego projektu. Każdy agent kodowania w 2026 wspiera AGENTS.md: Claude Code, Cursor, Codex, Copilot Workspace, opencode, Windsurf, Zed.

### Format SKILL.md

Umiejętności Agentów Anthropic (wydane jako otwarty standard w grudniu 2025):

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter deklaruje tożsamość umiejętności. Treść to prompt pokazywany modelowi, gdy umiejętność jest ładowana.

### Progresywne ujawnianie

Umiejętności mogą odwoływać się do pod-zasobów, które agent pobiera tylko w razie potrzeby. Przykład:

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md mówi "zobacz style-guide.md po reguły stylu." Agent pobiera style-guide.md tylko wtedy, gdy umiejętność jest aktywnie uruchomiona. Pozwala to uniknąć rozdymania promptu szczegółami, których model może nie potrzebować.

### Odkrywanie przez system plików

Środowiska uruchomieniowe agentów skanują znane katalogi w poszukiwaniu plików SKILL.md:

- `~/.anthropic/skills/*/SKILL.md`
- Projekt `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

Ładowanie odbywa się według nazwy folderu i `name` z frontmatteru. Claude Code, Anthropic Claude Agent SDK i SkillKit (między-agentowy) wszystkie podążają za tym wzorcem.

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk` (TypeScript) i `claude-agent-sdk` (Python) ładują umiejętności na starcie sesji, udostępniając je jako wywoływalne "agentów" w środowisku uruchomieniowym. Pętla agenta wysyła do umiejętności, gdy użytkownik ją wywoła.

### OpenAI Apps SDK

Uruchomiony w październiku 2025; zbudowany bezpośrednio na MCP. Unified poprzednie Connectors OpenAI i Custom GPT Actions pod jedną powierzchnią programistyczną. Aplikacja Apps SDK to:

- Serwer MCP (narzędzia, zasoby, prompty).
- Plus metadane widżetów dla interfejsu ChatGPT.
- Plus opcjonalny zasób MCP Apps `ui://` dla interaktywnych powierzchni.

Ten sam protokół, bogatsze UX.

### Między-agentowa przenośność przez SkillKit

Narzędzia takie jak SkillKit i podobne warstwy dystrybucji między-agentowej tłumaczą pojedynczy SKILL.md na natywny format każdego z 32+ agentów AI (Claude Code, Cursor, Codex, Gemini CLI, OpenCode itd.). Jedno źródło prawdy; wielu odbiorców.

### Stos trzech warstw

| Warstwa | Plik | Ładowany gdy | Cel |
|-------|------|-------------|---------|
| AGENTS.md | katalog główny repo | start sesji | konwencje na poziomie projektu |
| SKILL.md | katalog umiejętności | umiejętność wywołana | wielokrotnego użytku przepływ pracy |
| Serwer MCP | proces zewnętrzny | potrzebne narzędzia | wywoływalne akcje |

Wszystkie trzy się łączą: agent czyta AGENTS.md na starcie sesji, użytkownik wywołuje umiejętność, instrukcje umiejętności zawierają wywołania narzędzi MCP, agent wysyła je przez klienta MCP.

## Use It

`code/main.py` dostarcza parser i ładowarkę SKILL.md w stdlib. Odkrywa umiejętności w `./skills/`, parsuje frontmatter YAML plus treść w markdown i produkuje słownik kluczowany według nazwy umiejętności. Następnie symuluje pętlę agenta, która wywołuje `release-notes-writer` po nazwie.

Na co zwrócić uwagę:

- Frontmatter YAML parsowany minimalistycznym parserem stdlib (brak zależności `pyyaml`).
- Treść umiejętności przechowywana dosłownie; agent dołącza ją do promptu systemowego przy wywołaniu.
- Progresywne ujawnianie zademonstrowane przez funkcję `read_subresource`, która pobiera wskazane pliki na żądanie.

## Ship It

Ta lekcja produkuje `outputs/skill-agent-bundle.md`. Mając przepływ pracy, umiejętność produkuje połączony pakiet SKILL.md + AGENTS.md + blueprint serwera MCP, przenośny między agentami.

## Exercises

1. Uruchom `code/main.py`. Dodaj drugą umiejętność w `skills/` i potwierdź, że ładowarka ją wykrywa.

2. Napisz AGENTS.md dla tego repozytorium kursu. Uwzględnij polecenia testowania, konwencje stylu i model mentalny Fazy 13.

3. Przepisz wieloetapowy przepływ pracy z wewnętrznej dokumentacji twojego zespołu do SKILL.md. Zweryfikuj, że ładuje się w Claude Code.

4. Ręcznie przetłumacz umiejętność na natywne formaty reguł Cursor i Codex. Policz różnice między formatami — to jest powierzchnia translacji, którą automatyzuje SkillKit.

5. Przeczytaj wpis na blogu Anthropic Agent Skills. Zidentyfikuj jedną funkcję w Claude Agent SDK, której ładowarka tej lekcji nie obejmuje. (Podpowiedź: pod-wywołanie agenta.)

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SKILL.md | "Plik umiejętności" | Frontmatter YAML plus treść markdown, ładowany przez środowisko agenta |
| AGENTS.md | "Kontekst agenta w katalogu głównym repo" | Plik konwencji na poziomie projektu czytany na starcie sesji |
| Progressive disclosure | "Leniwie ładowane pod-zasoby" | Treść umiejętności odwołuje się do plików pobieranych tylko w razie potrzeby |
| Frontmatter | "Blok YAML na górze" | Metadane (nazwa, opis) w ogranicznikach `---` |
| Claude Agent SDK | "Środowisko umiejętności Anthropic" | `@anthropic-ai/claude-agent-sdk`, ładuje umiejętności i kieruje |
| OpenAI Apps SDK | "MCP + metadane widżetów" | Powierzchnia programistyczna OpenAI zbudowana na MCP plus haki UI ChatGPT |
| Skill discovery | "Skanowanie systemu plików" | Przechodzenie znanych katalogów w poszukiwaniu SKILL.md, kluczowanie według nazwy |
| Cross-agent portability | "Jedna umiejętność, wielu agentów" | Tłumaczenie jednego SKILL.md na 32+ agentów przez narzędzia typu SkillKit |
| Agent Skill | "Przenośne know-how" | Szablon zadania wielokrotnego użytku poza koncepcją narzędzi MCP |
| Apps SDK | "MCP plus UI ChatGPT" | Connectors i Custom GPTs ujednolicone na MCP |

## Further Reading

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — uruchomienie w grudniu 2025
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — dokumentacja formatu SKILL.md
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — platforma programistyczna oparta na MCP dla ChatGPT
- [agents.md](https://agents.md/) — format AGENTS.md i lista adopcji
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — oficjalne przykłady umiejętności