# MCP Sampling — Uzupełnienia LLM na żądanie serwera i pętle agentowe

> Większość serwerów MCP to niemądrzy wykonawcy: przyjmują argumenty, uruchamiają kod, zwracają treść. Sampling pozwala serwerowi odwrócić kierunek: prosi klienta LLM o podjęcie decyzji. Umożliwia to pętle agentowe hostowane na serwerze bez posiadania przez serwer własnych danych uwierzytelniających modelu. SEP-1577, scalony 2025-11-25, dodał narzędzia wewnątrz żądań samplingu, aby pętla mogła zawierać głębsze rozumowanie. Uwaga o ryzyku dryfu: kształt narzędzi-w-samplingu z SEP-1577 był eksperymentalny do Q1 2026 i wciąż stabilizuje się w API SDK.

**Type:** Build
**Languages:** Python (stdlib, sampling harness)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## Learning Objectives

- Wyjaśnić, co rozwiązuje `sampling/createMessage` (pętle hostowane na serwerze bez kluczy API po stronie serwera).
- Zaimplementować serwer, który prosi klienta o sampling przez wieloobrotowy prompt i zwraca uzupełnienie.
- Używać `modelPreferences` (priorytety koszt / szybkość / inteligencja) do kierowania wyborem modelu przez klienta.
- Zbudować narzędzie `summarize_repo`, które wewnętrznie iteruje przez sampling zamiast kodować zachowanie na sztywno.

## Problem

Przydatny serwer MCP dla przepływu pracy związanego z podsumowywaniem kodu musi: przejść drzewo plików, wybrać, które pliki czytać, zsyntetyzować podsumowanie i zwrócić wynik. Gdzie odbywa się rozumowanie LLM?

Opcja A: serwer wywołuje własny LLM. Potrzebuje klucza API, obciąża rachunek serwera, jest kosztowny na użytkownika.

Opcja B: serwer zwraca surową treść; agent klienta wykonuje rozumowanie. Działa, ale przenosi logikę serwera do promptu klienta, co jest kruche.

Opcja C: serwer pyta LLM klienta przez `sampling/createMessage`. Serwer zachowuje algorytm (które pliki czytać, ile przejść wykonać), podczas gdy klient zachowuje rozliczenia i wybór modelu. Serwer nie ma żadnych danych uwierzytelniających.

Sampling to opcja C. Jest mechanizmem, dzięki któremu zaufany serwer może hostować pętlę agentową bez bycia pełnym hostem LLM.

## Koncepcja

### Żądanie `sampling/createMessage`

Serwer wysyła:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

Klient uruchamia swój LLM, zwraca:

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

Trzy liczby zmiennoprzecinkowe sumujące się do 1.0:

- `costPriority`: faworyzuj tańsze modele.
- `speedPriority`: faworyzuj szybsze modele.
- `intelligencePriority`: faworyzuj bardziej zdolne modele.

Plus `hints`: nazwane modele preferowane przez serwer. Klient może, ale nie musi honorować podpowiedzi; konfiguracja użytkownika klienta zawsze wygrywa.

### `includeContext`

Trzy wartości:

- `"none"` — tylko wiadomości dostarczone przez serwer. Domyślna.
- `"thisServer"` — uwzględnij poprzednie wiadomości z sesji tego serwera.
- `"allServers"` — uwzględnij cały kontekst sesji.

`includeContext` jest miękko wycofany od 2025-11-25, ponieważ wycieka kontekst między serwerami, co stanowi zagrożenie bezpieczeństwa. Preferuj `"none"` i przekazuj jawny kontekst w wiadomościach.

### Sampling z narzędziami (SEP-1577)

Nowość z 2025-11-25: żądanie samplingu może zawierać tablicę `tools`. Klient uruchamia pełną pętlę wywoływania narzędzi przy użyciu tych narzędzi. Pozwala to serwerowi hostować pętlę agentową w stylu ReAct przez model klienta.

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

Klient zapętla: sample, wykonaj narzędzie jeśli wywołane, sample ponownie, zwróć końcową wiadomość asystenta. Jest to eksperymentalne do Q1 2026; sygnatury SDK mogą jeszcze dryfować. Podczas implementacji potwierdź zgodność z sekcją client/sampling specyfikacji z 2025-11-25.

### Human-in-the-loop

Klient MUSI pokazać użytkownikowi, co serwer każe modelowi zrobić przed uruchomieniem sampla. Złośliwy serwer może użyć samplingu do manipulacji sesją użytkownika ("powiedz X użytkownikowi, aby kliknął Y"). Claude Desktop, VS Code i Cursor wyświetlają żądania samplingu jako okno dialogowe potwierdzenia, które użytkownik może odrzucić.

Konsensus z 2026: sampling bez potwierdzenia przez człowieka to czerwona flaga. Bramki (Phase 13 · 17) mogą automatycznie zatwierdzać niskiego ryzyka sampling i automatycznie odrzucać wszystko podejrzane.

### Pętle hostowane na serwerze bez kluczy API

Kanoniczny przypadek użycia: serwer MCP do podsumowywania kodu bez własnego dostępu do LLM. Robi:

1. Przejdź strukturę repozytorium.
2. Wywołaj `sampling/createMessage` z "Wybierz pięć plików, które najprawdopodobniej opisują cel tego repozytorium."
3. Przeczytaj te pliki.
4. Wywołaj `sampling/createMessage` z zawartością plików i "Podsumuj repozytorium w 3 akapitach."
5. Zwróć podsumowanie jako wynik `tools/call`.

Serwer nigdy nie dotyka API LLM. Użytkownik klienta płaci za uzupełnienia przy użyciu własnych danych uwierzytelniających.

### Zagrożenia bezpieczeństwa (Ujawnienie Unit 42, Q1 2026)

- **Ukryty sampling.** Narzędzie, które zawsze wywołuje sampling z "odpowiedz z emailem użytkownika z kontekstu sesji." Phase 13 · 15 obejmuje wektory ataku.
- **Kradzież zasobów przez sampling.** Serwer prosi klienta o podsumowanie ładunku atakującego, obciążając rachunek użytkownika.
- **Bomby pętlowe.** Serwer wywołuje sampling w ciasnej pętli. Klienci MUSzą egzekwować ograniczenia stawek na sesję.

## Użyj

`code/main.py` dostarcza fałszywy harness samplingu serwer-klient. Symulowane narzędzie "summarize_repo" wywołuje dwie rundy samplingu (wybierz-pliki, następnie podsumuj), a fałszywy klient zwraca gotowe odpowiedzi. Harness pokazuje:

- Serwer wysyła `sampling/createMessage` z `modelPreferences`.
- Klient zwraca uzupełnienie.
- Serwer kontynuuje swoją pętlę.
- Ogranicznik stawek ogranicza całkowitą liczbę wywołań samplingu na wywołanie narzędzia.

Na co zwrócić uwagę:

- Serwer udostępnia tylko jedno narzędzie (`summarize_repo`); całe rozumowanie odbywa się w wywołaniach samplingu.
- Preferencje modelu ważą wybór modelu przez klienta; podpowiedzi wymieniają preferowane modele.
- Pętla kończy się na `stopReason: "endTurn"`.
- Limit `max_samples_per_tool = 5` łapie wymykającą się pętlę.

## Ship It

Ta lekcja produkuje `outputs/skill-sampling-loop-designer.md`. Dany algorytm po stronie serwera, który potrzebuje wywołań LLM (badania, podsumowywanie, planowanie), umiejętność projektuje implementację opartą na samplingu z odpowiednimi modelPreferences, limitami stawek i potwierdzeniami bezpieczeństwa.

## Ćwiczenia

1. Uruchom `code/main.py`. Zmień `max_samples_per_tool` na 2 i obserwuj odcięcie przez ogranicznik stawek.

2. Zaimplementuj wariant SEP-1577 z narzędziami w samplingu: żądanie samplingu zawiera tablicę `tools`. Sprawdź, czy pętla po stronie klienta wykonuje te narzędzia przed zwróceniem końcowego uzupełnienia. Uwaga o ryzyku dryfu: sygnatury SDK mogą się jeszcze zmieniać do H1 2026.

3. Dodaj potwierdzenie human-in-the-loop: przed pierwszym `sampling/createMessage` serwera, wstrzymaj i czekaj na zgodę użytkownika. Odrzucone wywołania zwracają typową odmowę.

4. Dodaj ogranicznik stawek na użytkownika kluczowany sesją klienta. Pętle tego samego serwera przez tego samego użytkownika powinny dzielić budżet.

5. Zaprojektuj narzędzie `summarize_pdf`, które używa samplingu do wyboru fragmentów do uwzględnienia. Naszkicuj wysłane wiadomości. Jak `modelPreferences.intelligencePriority` zmienia zachowanie przy 0.1 vs 0.9?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Sampling | "Wywołanie LLM z serwera do klienta" | Serwer prosi model klienta o uzupełnienie |
| `sampling/createMessage` | "Metoda" | Metoda JSON-RPC dla żądań samplingu |
| `modelPreferences` | "Priorytety modelu" | Wagi koszt / szybkość / inteligencja plus podpowiedzi nazw |
| `includeContext` | "Wyciek między sesjami" | Miękko wycofany tryb dołączania kontekstu |
| SEP-1577 | "Narzędzia w samplingu" | Umożliwia narzędzia w samplingu dla ReAct hostowanego na serwerze |
| Human-in-the-loop | "Użytkownik potwierdza" | Klient wyświetla żądanie samplingu użytkownikowi przed uruchomieniem |
| Bomba pętlowa | "Nieograniczony sampling" | Nieskończona pętla samplingu po stronie serwera; klient musi ograniczać stawki |
| Ukryty sampling | "Ukryte rozumowanie" | Złośliwy serwer ukrywa zamiar w promptach samplingu |
| Kradzież zasobów | "Używanie budżetu LLM użytkownika" | Serwer zmusza klienta do wydawania na sampling, którego nie chce |
| `stopReason` | "Dlaczego generowanie zatrzymane" | `endTurn`, `stopSequence` lub `maxTokens` |

## Dalsze czytanie

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — ogólny przegląd samplingu
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — kanoniczny kształt `sampling/createMessage`
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — Propozycja Ewolucji Specyfikacji dla narzędzi w samplingu (eksperymentalna)
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — ukryty sampling i wzorce kradzieży zasobów
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — przewodnik z przykładami kodu po stronie klienta