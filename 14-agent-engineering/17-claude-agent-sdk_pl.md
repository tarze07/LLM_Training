# Claude Agent SDK: Podagenci i Magazyn Sesji

> Claude Agent SDK to biblioteczna forma szkieletu Claude Code. Wbudowane narzędzia, podagenci do izolacji kontekstu, haki, propagacja kontekstu śledzenia W3C, parytet magazynu sesji. Claude Managed Agents to hostowana alternatywa dla długotrwałej pracy asynchronicznej.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 minutes

## Learning Objectives

- Wyjaśnij różnicę między Anthropic Client SDK (surowe API) a Claude Agent SDK (kształt szkieletu).
- Opisz podagentów — parallelizację i izolację kontekstu — oraz kiedy po nie sięgać.
- Wymień powierzchnię magazynu sesji Python SDK (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`) oraz rolę `--session-mirror`.
- Zaimplementuj szkielet w stdlib z wbudowanymi narzędziami, uruchamianiem podagentów z izolowanym kontekstem, hakami cyklu życia i magazynem sesji.

## Problem

Surowe API LLM daje ci jedną rundę. Produkcyjny agent potrzebuje wykonywania narzędzi, serwerów MCP, haków cyklu życia, uruchamiania podagentów, trwałości sesji, propagacji śledzenia. Claude Agent SDK dostarcza ten kształt jako bibliotekę — ten sam szkielet, którego używa Claude Code, udostępniony dla niestandardowych agentów.

## Koncepcja

### Client SDK a Agent SDK

- **Client SDK (`anthropic`).** Surowe API Messages. Ty zarządzasz pętlą, narzędziami i stanem.
- **Agent SDK (`claude-agent-sdk`).** Wbudowane wykonywanie narzędzi, połączenia MCP, haki, uruchamianie podagentów, magazyn sesji. Pętla Claude Code jako biblioteka.

### Wbudowane narzędzia

SDK dostarcza 10+ narzędzi gotowych do użycia: odczyt/zapis plików, shell, grep, glob, pobieranie stron WWW i inne. Niestandardowe narzędzia rejestruje się przez standardowy interfejs schematu narzędzi.

### Podagenci

Dwa cele udokumentowane przez Anthropic:

1. **Równoległość.** Wykonuj niezależną pracę współbieżnie. „Znajdź plik testowy dla każdego z tych 20 modułów" to 20 równoległych zadań podagentów.
2. **Izolacja kontekstu.** Podagenci używają własnego okna kontekstu; tylko wyniki wracają do orkiestratora. Budżet orkiestratora jest zachowany.

Najnowsze dodatki Python SDK: `list_subagents()`, `get_subagent_messages()` do odczytu transkryptów podagentów.

### Magazyn sesji

Parytet protokołu z TypeScript:

- `append(session_id, message)` — dodaj turę.
- `load(session_id)` — przywróć konwersację.
- `list_sessions()` — wylicz.
- `delete(session_id)` — z kaskadą do sesji podagentów.
- `list_subkeys(session_id)` — wypisz klucze podagentów.

`--session-mirror` (flaga CLI) zapisuje transkrypt do zewnętrznego pliku w miarę strumieniowania, do debugowania.

### Haki

Haki cyklu życia, które możesz zarejestrować:

- `PreToolUse`, `PostToolUse` — bramkuj lub audytuj wywołania narzędzi.
- `SessionStart`, `SessionEnd` — konfiguruj i zamykaj.
- `UserPromptSubmit` — działaj na wejściu użytkownika zanim model je zobaczy.
- `PreCompact` — uruchom przed kompakcją kontekstu.
- `Stop` — sprzątanie przy wyjściu agenta.
- `Notification` — alerty kanałem bocznym.

Haki to sposób, w jaki pro-workflow (odniesienie do programu Phase 14) i podobne systemy dodają zachowania przekrojowe.

### Kontekst śledzenia W3C

Spany OTel aktywne u wywołującego propagują się do podprocesu CLI przez nagłówki kontekstu śledzenia W3C. Cały ślad wieloprocesowy pojawia się jako jeden ślad w twoim backendzie.

### Claude Managed Agents

Hostowana alternatywa (nagłówek beta `managed-agents-2026-04-01`). Długotrwała praca asynchroniczna, wbudowane buforowanie promptów, wbudowana kompakcja. Wymiana kontroli na zarządzaną infrastrukturę.

### Gdzie ten wzorzec zawodzi

- **Nadmierne uruchamianie podagentów.** Uruchamianie 100 podagentów dla 100 drobnych zadań. Narzut dominuje. Lepiej grupować.
- **Rozrost haków.** Każdy zespół dodaje haki; czas uruchamiania rośnie. Przeglądaj haki kwartalnie.
- **Rozdęcie sesji.** Sesje kumulują się; rozmiar rośnie. Używaj `list_sessions` + polityki wygasania.

## Build It

`code/main.py` implementuje kształt SDK w stdlib:

- `Tool`, `ToolRegistry` z wbudowanymi `read_file`, `write_file`, `list_dir`.
- `Subagent` — prywatny kontekst, izolowane uruchomienie, zwracane wyniki.
- `SessionStore` — append, load, list, delete, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- Demo: główny agent uruchamia 3 podagentów równolegle (każdy izolowany), agreguje wyniki, utrwala sesję.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje izolację kontekstu podagentów (rozmiar kontekstu orkiestratora pozostaje ograniczony), wykonywanie haków i trwałość sesji.

## Use It

- **Claude Agent SDK** dla produktów zorientowanych na Claude, które chcą kształtu szkieletu Claude Code.
- **Claude Managed Agents** dla hostowanej długotrwałej pracy asynchronicznej.
- **OpenAI Agents SDK** (Lekcja 16) dla odpowiedników zorientowanych na OpenAI.
- **LangGraph + niestandardowe narzędzia** jeśli wolisz maszynę stanów w kształcie grafu.

## Ship It

`outputs/skill-claude-agent-scaffold.md` buduje szkielet aplikacji Claude Agent SDK z podagentami, hakami, magazynem sesji, podłączeniem serwera MCP i propagacją kontekstu śledzenia W3C.

## Ćwiczenia

1. Dodaj kreator podagentów, który grupuje 20 zadań w partie po 5 równoległych podagentów. Zmierz rozmiar kontekstu orkiestratora w porównaniu do jednego na zadanie.
2. Zaimplementuj hak `PreToolUse`, który ogranicza szybkość wywołań `write_file` (5 na minutę na sesję). Prześledź zachowanie.
3. Podłącz `list_subkeys` do renderowania drzewa podagentów. Jak wygląda głębokie zagnieżdżenie?
4. Przenieś zabawkę do prawdziwego pakietu Python `claude-agent-sdk`. Co zmienia się w rejestracji narzędzi?
5. Przeczytaj dokumentację Claude Managed Agents. Kiedy przeszedłbyś z samodzielnego hostowania na zarządzane?

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| Agent SDK | „Claude Code jako biblioteka" | Kształt szkieletu: narzędzia, MCP, haki, podagenci, magazyn sesji |
| Podagent | „Agent potomny" | Osobny kontekst, własny budżet; wyniki wypływają w górę |
| Magazyn sesji | „Baza danych konwersacji" | Utrwalaj, wczytuj, wyliczaj, usuwaj tury z kaskadą podagentów |
| Hak | „Callback cyklu życia" | Przed/po narzędziu, sesji, zgłoszeniu promptu, kompakcji, zatrzymaniu |
| Kontekst śledzenia W3C | „Ślad międzyprocesowy" | Span rodzica propaguje się do podprocesu CLI |
| Managed Agents | „Hostowany szkielet" | Długotrwała praca asynchroniczna hostowana przez Anthropic |
| `--session-mirror` | „Lustrzany transkrypt" | Zapisuje tury sesji do zewnętrznego pliku w miarę strumieniowania |
| Serwer MCP | „Powierzchnia narzędzi" | Zewnętrzne źródło narzędzi/zasobów podłączone do agenta |

## Dalsza lektura

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — biblioteczna forma Claude Code
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — wzorce produkcyjne
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hostowana alternatywa
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — odpowiednik