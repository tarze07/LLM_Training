# Korzenie i Elicytacja — Zakresowanie i Wprowadzanie Danych w Trakcie Działania

> Zakodowane na sztywno ścieżki przestają działać, gdy użytkownik otworzy inny projekt. Wstępnie wypełnione argumenty narzędzi przestają działać, gdy użytkownik nie określi wystarczających informacji. Korzenie (roots) zakresowują serwer do zestawu URI kontrolowanych przez użytkownika; elicytacja wstrzymuje się w trakcie wywołania narzędzia, aby zapytać użytkownika o ustrukturyzowane dane wejściowe przez formularz lub URL. Dwa prymitywy klienckie, dwa rozwiązania typowych trybów awarii MCP. SEP-1036 (elicytacja w trybie URL, 2025-11-25) jest eksperymentalny do H1 2026 — sprawdź wersje SDK przed poleganiem na nim.

**Type:** Build
**Languages:** Python (stdlib, roots + elicitation demo)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## Learning Objectives

- Zadeklarować `roots` i odpowiadać na `notifications/roots/list_changed`.
- Ograniczyć operacje na plikach serwera do URI wewnątrz zadeklarowanego zestawu korzeni.
- Użyć `elicitation/create`, aby zapytać użytkownika o potwierdzenie lub ustrukturyzowane dane wejściowe w trakcie wywołania narzędzia.
- Wybrać między trybem formularza a trybem URL elicytacji (ten drugi jest eksperymentalny; odnotowano ryzyko dryfu).

## Problem

Dwie konkretne awarie, które serwer MCP notatek napotyka w produkcji.

**Załamane założenie ścieżki.** Serwer jest napisany względem `~/notes`. Użytkownik na innym komputerze z notatkami w `~/Documents/Notes` otrzymuje wywołanie narzędzia, które cicho zawodzi (nie znaleziono pliku) lub co gorsza, zapisuje w złym miejscu.

**Brakujący argument, który użytkownik by znał.** Użytkownik pyta "usuń stary raport TPS". Model wywołuje `notes_delete(title: "TPS report")`, ale są trzy pasujące notatki z 2023, 2024 i 2025. Narzędzie nie może zgadnąć. Zakończenie "niejednoznaczne" jest irytujące; uruchomienie na wszystkich trzech jest katastrofalne.

Korzenie rozwiązują pierwszy problem: klient deklaruje przy `initialize` zestaw URI, których serwer może dotykać. Elicytacja rozwiązuje drugi: serwer wstrzymuje wywołanie narzędzia i wysyła `elicitation/create`, aby zapytać użytkownika, którą wybrać.

## Koncepcja

### Korzenie (Roots)

Klient deklaruje listę korzeni przy `initialize`:

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

Serwer może następnie wywołać `roots/list`:

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

Serwery MUSZĄ traktować korzenie jako granicę: każdy odczyt lub zapis pliku poza zestawem korzeni jest odrzucany. Nie jest to egzekwowane przez klienta (serwer to wciąż kod, któremu użytkownik zaufał), ale serwery zgodne ze specyfikacją honorują to.

Gdy użytkownik dodaje lub usuwa korzeń, klient wysyła `notifications/roots/list_changed`. Serwer ponownie wywołuje `roots/list` i aktualizuje swoją granicę.

### Dlaczego korzenie są prymitywem klienckim

Korzenie są deklarowane przez klienta, ponieważ reprezentują model zgody użytkownika. Użytkownik powiedział Claude Desktop "daj temu serwerowi notatek dostęp do tych dwóch katalogów". Serwer nie może poszerzyć tego zakresu.

### Elicytacja: domyślny tryb formularza

`elicitation/create` przyjmuje schemat formularza plus naturalnojęzykowy prompt:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Usunąć 'TPS report'? Pasuje wiele notatek; wybierz jedną.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

Klient renderuje formularz, zbiera odpowiedź użytkownika, zwraca:

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

Trzy możliwe akcje: `accept` (użytkownik wypełnił), `decline` (użytkownik zamknął), `cancel` (użytkownik przerwał całe wywołanie narzędzia).

Schematy formularzy są płaskie — zagnieżdżone obiekty nie są obsługiwane w v1. SDK zazwyczaj odrzucają wszystko bardziej złożone niż pojedyncza warstwa.

### Elicytacja: tryb URL (SEP-1036, eksperymentalny)

Nowość z 2025-11-25. Zamiast schematu serwer wysyła URL:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Zaloguj się do GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

Klient otwiera URL w przeglądarce, czeka na zakończenie, zwraca, gdy użytkownik wróci. Przydatne dla przepływów OAuth, autoryzacji płatności i podpisywania dokumentów, gdzie formularz jest niewystarczający.

Uwaga o ryzyku dryfu: kształt odpowiedzi SEP-1036 wciąż się stabilizuje; niektóre SDK zwracają URL callbacku, inne zwracają token zakończenia. Przeczytaj notatki wydania swojego SDK przed użyciem trybu URL w produkcji.

### Kiedy elicytacja jest właściwym narzędziem

- Potwierdzenie użytkownika przed destrukcyjnymi działaniami (wskaźnik destrukcyjności + elicytacja).
- Ujednoznacznienie (wybierz jeden z N pasujących).
- Konfiguracja przy pierwszym uruchomieniu (klucze API, katalogi, preferencje).
- Przepływy w stylu OAuth (tryb URL).

### Kiedy elicytacja jest niewłaściwa

- Wypełnianie wymaganych argumentów narzędzia, o które model mógłby zapytać w prozie. Użyj normalnego ponownego promptu, a nie okna elicytacji.
- Wywołania o wysokiej częstotliwości. Elicytacja przerywa rozmowę; nie uruchamiaj jej wewnątrz pętli.
- Czegokolwiek, co serwer może zweryfikować po fakcie. Zweryfikuj, zwróć błąd, pozwól modelowi zapytać użytkownika w tekście.

### Pomost human-in-the-loop

Elicytacja i sampling razem umożliwiają model MCP "human-in-the-loop". Pętla agentowa serwera może wstrzymać się na wprowadzenie danych przez użytkownika (elicytacja) lub rozumowanie modelu (sampling). Phase 13 · 11 omawiała sampling; ta lekcja obejmuje elicytację. Połącz je, aby uzyskać pełną kontrolę w trakcie pętli.

## Użyj

`code/main.py` rozszerza serwer notatek o:

- Odpowiedź `roots/list`, którą serwer ponownie odpytał po powiadomieniach o zmianie listy korzeni.
- Narzędzie `notes_delete`, które używa `elicitation/create` do ujednoznacznienia, gdy pasuje wiele notatek.
- Narzędzie `notes_setup`, które używa elicytacji w trybie URL do otwarcia strony konfiguracyjnej przy pierwszym uruchomieniu (symulowane).
- Sprawdzenie granicy, które odmawia operacji na URI poza zadeklarowanymi korzeniami.

Demo uruchamia trzy scenariusze: ścieżka szczęśliwa (jeden pasujący), ujednoznacznienie (trzy pasujące, elicytacja uruchamiana), zapis poza korzeniem (odrzucony).

## Ship It

Ta lekcja produkuje `outputs/skill-elicitation-form-designer.md`. Dla narzędzia, które może potrzebować potwierdzenia użytkownika lub ujednoznacznienia, umiejętność projektuje schemat formularza elicytacji i szablon wiadomości.

## Ćwiczenia

1. Uruchom `code/main.py`. Wyzwól ścieżkę ujednoznacznienia; potwierdź, że symulowana odpowiedź użytkownika jest kierowana z powrotem do narzędzia.

2. Dodaj nowe narzędzie `notes_archive`, które wymaga potwierdzenia elicytacji za każdym razem (wskaźnik destrukcyjności). Sprawdź UX: jak to się porównuje do ponownego pytania modelu w tekście?

3. Zaimplementuj elicytację w trybie URL dla przepływu OAuth przy pierwszym uruchomieniu. Zwróć uwagę na ryzyko dryfu i dodaj strażnika wersji SDK.

4. Rozszerz obsługę `roots/list`: gdy nadejdzie powiadomienie, serwer powinien atomowo ponownie odczytać i zeskanować otwarte uchwyty plików, które mogą być teraz poza zakresem.

5. Przeczytaj wątek dyskusji SEP-1036 na GitHubie. Zidentyfikuj jedno otwarte pytanie, które wpływa na to, jak serwery powinny obsługiwać callbacki w trybie URL.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Korzeń (Root) | "Granica zgody" | URI, którego klient pozwolił serwerowi dotykać |
| `roots/list` | "Serwer pyta o zakres" | Klient zwraca bieżący zestaw korzeni |
| `notifications/roots/list_changed` | "Użytkownik zmienił zakres" | Klient sygnalizuje mutację zestawu korzeni |
| Elicytacja | "Zapytaj użytkownika w trakcie" | Żądanie zainicjowane przez serwer dla ustrukturyzowanych danych od użytkownika |
| `elicitation/create` | "Metoda" | Metoda JSON-RPC dla żądań elicytacji |
| Tryb formularza | "Formularz sterowany schematem" | Płaski JSON Schema renderowany jako formularz w UI klienta |
| Tryb URL | "Przekierowanie przeglądarki" | SEP-1036 eksperymentalny; otwiera URL i czeka |
| `accept` / `decline` / `cancel` | "Wyniki odpowiedzi użytkownika" | Trzy gałęzie obsługiwane przez serwer |
| Ujednoznacznienie | "Wybierz jeden" | Typowy przypadek użycia elicytacji, gdy narzędzie ma N kandydatów |
| Płaski formularz | "Tylko właściwości najwyższego poziomu" | Schematy elicytacji nie mogą być zagnieżdżone |

## Dalsze czytanie

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — kanoniczny opis korzeni
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — kanoniczny opis elicytacji
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — przewodnik po dodatkach z 2025-11-25
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — propozycja elicytacji w trybie URL (eksperymentalna, ryzyko dryfu)
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — przewodnik UX