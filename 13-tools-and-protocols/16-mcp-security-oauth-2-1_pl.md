# MCP Security II — OAuth 2.1, Resource Indicators, Zakresy Przyrostowe

> Zdalne serwery MCP wymagają autoryzacji, nie tylko uwierzytelniania. Specyfikacja z 2025-11-25 jest zgodna z OAuth 2.1 + PKCE + resource indicators (RFC 8707) + metadanymi zasobów chronionych (RFC 9728). SEP-835 dodaje przyrostową zgodę zakresu z autoryzacją krokową na 403 WWW-Authenticate. Ta lekcja implementuje przepływ krokowy jako maszynę stanów, abyś mógł zobaczyć każdy krok.

**Type:** Build
**Languages:** Python (stdlib, symulator maszyny stanów OAuth)
**Prerequisites:** Phase 13 · 09 (transporty), Phase 13 · 15 (security I)
**Time:** ~75 minut

## Learning Objectives

- Rozróżnić obowiązki serwera zasobów od serwera autoryzacyjnego.
- Przejść przez przepływ kodu autoryzacyjnego OAuth 2.1 chronionego PKCE.
- Użyć `resource` (RFC 8707) i metadanych zasobów chronionych (RFC 9728) do zapobiegania atakom confused-deputy.
- Zaimplementować autoryzację krokową: serwer odpowiada 403 z WWW-Authenticate prosząc o wyższy zakres; klient ponownie wyświetla zgodę użytkownika i ponawia próbę.

## Problem

Wczesne MCP (przed 2025) dostarczało zdalne serwery z ad-hoc kluczami API lub nawet bez autoryzacji. Specyfikacja z 2025-11-25 zamyka tę lukę za pomocą pełnego profilu OAuth 2.1.

Trzy potrzeby ze świata rzeczywistego:

- **Zwykłe zdalne serwery.** Użytkownik instaluje zdalny serwer MCP, który uzyskuje dostęp do jego Notion / GitHub / Gmail. OAuth 2.1 z PKCE jest odpowiednim rozwiązaniem.
- **Eskalacja zakresu.** Serwer notatek, któremu przyznano `notes:read`, może później potrzebować `notes:write` dla konkretnej akcji. Zamiast powtarzać cały przepływ, autoryzacja krokowa (SEP-835) prosi o dodatkowy zakres.
- **Zapobieganie confused deputy.** Klient posiada token z ograniczeniem audytorium dla Serwera A. Serwer A jest złośliwy i próbuje przedstawić token Serwerowi B. Resource indicators (RFC 8707) przypinają token do zamierzonego audytorium.

OAuth 2.1 nie jest nowy. Nowy jest profil MCP: wymagane konkretne przepływy (tylko authorization code + PKCE; brak implicit, brak client credentials domyślnie), obowiązkowe resource indicators przy każdym żądaniu tokena oraz publikowane metadane zasobów chronionych, aby klienci wiedzieli, dokąd się udać.

## Koncepcja

### Role

- **Client.** Klient MCP (Claude Desktop, Cursor itp.).
- **Resource server.** Serwer MCP (notatki, GitHub, Postgres, cokolwiek).
- **Authorization server.** Wydaje tokeny. Może być tą samą usługą co serwer zasobów lub osobnym IdP (Auth0, Keycloak, Cognito).

W profilu MCP serwer zasobów i serwer autoryzacyjny MOGĄ być tym samym hostem, ale POWINNY być rozróżniane przez URL-e.

### Authorization code + PKCE

Przepływ:

1. Klient generuje `code_verifier` (losowy) i `code_challenge` (SHA256).
2. Klient przekierowuje użytkownika do `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`.
3. Użytkownik wyraża zgodę. Serwer autoryzacyjny przekierowuje do `redirect_uri?code=...`.
4. Klient wysyła POST do `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`.
5. Serwer autoryzacyjny weryfikuje hash weryfikatora względem zapisanego wyzwania i wydaje token dostępu.
6. Klient używa tokena: `Authorization: Bearer ...` przy każdym żądaniu do serwera zasobów.

PKCE zapobiega atakom przez przechwycenie kodu autoryzacyjnego. Resource indicators zapobiegają ważności tokena w innym miejscu.

### Metadane zasobów chronionych (RFC 9728)

Serwer zasobów publikuje dokument `.well-known/oauth-protected-resource`:

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

Klient odkrywa serwer autoryzacyjny z serwera zasobów. Zmniejsza to konfigurację — klient potrzebuje tylko URL-a zasobu.

### Resource indicators (RFC 8707)

Parametr `resource` w żądaniu tokena przypina zamierzone audytorium tokena. Wydany token zawiera `aud: "https://notes.example.com"`. Inny serwer MCP otrzymujący ten token sprawdza `aud` i odrzuca go.

### Model zakresów

Zakresy są ciągami znaków oddzielonymi spacjami. Typowe konwencje MCP:

- `notes:read`, `notes:write`, `notes:delete`
- `admin:*` dla uprawnień administracyjnych (używaj oszczędnie)
- `profile:read` dla tożsamości

Wybór zakresu powinien kierować się zasadą najmniejszych uprawnień: żądaj tego, czego potrzebujesz teraz, zwiększaj, gdy potrzebujesz więcej.

### Autoryzacja krokowa (SEP-835)

Użytkownik przyznaje `notes:read`. Później prosi agenta o usunięcie notatki. Serwer odpowiada:

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

Klient widzi błąd insufficient_scope, wyświetla użytkownikowi okno zgody na dodatkowy zakres, wykonuje mini przepływ OAuth dla niego, ponawia żądanie z nowym tokenem.

### Walidacja audytorium tokena

Każde żądanie: serwer sprawdza `token.aud == self.resource_url`. Niezgodność = 401. To zatrzymuje ponowne użycie tokena między serwerami.

### Krótkotrwałe tokeny i rotacja

Tokeny dostępu POWINNY być krótkotrwałe (domyślnie 1 godzina). Tokeny odświeżania rotują przy każdym odświeżeniu. Klient obsługuje ciche odświeżanie w tle.

### Brak przekazywania tokena

Serwery próbkujące (Phase 13 · 11) NIE MOGĄ przekazywać tokena klienta do innych usług. Żądanie próbkujące jest granicą.

### Zapobieganie confused deputy

Token jest związany z `aud`. Klient jest związany z `client_id`. Każde żądanie walidowane względem obu. Specyfikacja wyraźnie zakazuje starego wzorca "przekaż token", który był powszechny w ekosystemach zdalnych narzędzi przed MCP.

### Odkrywanie ID klienta

Każdy klient MCP publikuje swoje metadane pod stałym URL-em. Serwery autoryzacyjne mogą pobrać dokument metadanych klienta, aby odkryć URI przekierowań i dane kontaktowe. Eliminuje to ręczną rejestrację klienta.

### Bramki i OAuth

Phase 13 · 17 pokazuje, jak bramka korporacyjna obsługuje OAuth: bramka przechowuje poświadczenia dla serwerów nadrzędnych, tokeny dla klienta są wydawane przez bramkę, a nadrzędne tokeny nigdy nie opuszczają bramki. To odwraca model zaufania — użytkownicy uwierzytelniają się w bramce raz; bramka obsługuje N autoryzacji serwerów.

## Użyj tego

`code/main.py` symuluje pełny przepływ krokowy OAuth 2.1 jako maszynę stanów. Implementuje:

- Generowanie PKCE code-verifier / challenge.
- Przepływ kodu autoryzacyjnego z resource indicator.
- Endpoint metadanych zasobów chronionych.
- Walidację tokena z kontrolą audytorium.
- Autoryzację krokową na `insufficient_scope`.

Brak serwera HTTP w tej lekcji; maszyna stanów działa w pamięci, abyś mógł prześledzić każdy krok. Lekcja o bramkach w Phase 13 · 17 łączy to z rzeczywistym transportem.

## Ship It

Ta lekcja produkuje `outputs/skill-oauth-scope-planner.md`. Dla zdalnego serwera MCP z narzędziami, skill projektuje zestaw zakresów, reguły przypinania i politykę autoryzacji krokowej.

## Ćwiczenia

1. Uruchom `code/main.py`. Prześledź dwuzakresowy przepływ krokowy. Zwróć uwagę, które kroki powtarzają się przy autoryzacji krokowej.

2. Dodaj rotację tokena odświeżania: każde odświeżenie wydaje nowy token odświeżania i unieważnia stary. Zasymuluj skradziony token odświeżania użyty po rotacji i potwierdź, że zawodzi.

3. Zaimplementuj endpoint metadanych zasobów chronionych jako prawdziwą odpowiedź HTTP za pomocą stdlib http.server. Odwzoruj endpoint /mcp z Lekcji 09.

4. Zaprojektuj hierarchię zakresów dla serwera MCP GitHub: odczyt repozytorium, zapis PR, zatwierdzanie PR, scalanie PR, admin. Użyj autoryzacji krokowej między każdym poziomem.

5. Przeczytaj RFC 8707 i RFC 9728. Zidentyfikuj jedno pole w 9728, które MCP używa inaczej niż przykład z RFC. (Wskazówka: dotyczy `scopes_supported`.)

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| OAuth 2.1 | "Nowoczesne OAuth" | Skonsolidowane RFC, które wymusza PKCE i zabrania przepływu implicit |
| PKCE | "Proof-of-possession" | Weryfikator kodu + wyzwanie udaremniające przechwycenie kodu autoryzacyjnego |
| Resource indicator | "Audytorium tokena" | Parametr `resource` z RFC 8707 przypinający token do jednego serwera |
| Protected-resource metadata | "Dokument odkrywania" | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization | "Przyrostowa zgoda" | Przepływ SEP-835 do dodawania zakresów na żądanie |
| `insufficient_scope` | "403 z WWW-Authenticate" | Sygnał serwera do ponownej zgody na większy zakres |
| Confused deputy | "Ponowne użycie tokena między usługami" | Atak, w którym zaufany posiadacz niewłaściwie przekazuje token dalej |
| Short-lived token | "TTL tokena dostępu" | Nośnik, który szybko wygasa; token odświeżania odnawia |
| Scope hierarchy | "Stos najmniejszych uprawnień" | Stopniowany zestaw zakresów z autoryzacją krokową między poziomami |
| Client ID metadata | "Dokument odkrywania klienta" | URL, pod którym klient publikuje własne metadane OAuth |

## Dalsza lektura

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — kanoniczny profil OAuth MCP
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — omówienie zmian z 2025-11-25
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — RFC przypinania audytorium
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — RFC dokumentu odkrywania
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — praktyczne omówienie przepływu krokowego
