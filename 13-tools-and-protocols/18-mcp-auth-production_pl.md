# MCP Auth w Produkcji — Rejestracja, Odświeżanie JWKS, Tokeny z Przypiętym Audytorium

> Lekcja 16 uruchomiła maszynę stanów OAuth 2.1 w pamięci. W 2026 roku każdy serwer MCP wdrażany w prawdziwej organizacji działa z produkcyjnym uwierzytelnianiem: rejestracją klienta skalowalną dla nieograniczonej populacji klientów (Client ID Metadata Documents jako pierwszy wybór, dynamiczna rejestracja klienta jako kompatybilny wstecz fallback), odkrywaniem metadanych serwera autoryzacyjnego (RFC 8414 *lub* OpenID Connect Discovery), odświeżaniem pamięci podręcznej JWKS, która nie psuje walidacji tokena o 3 nad ranem, oraz tokenami z przypiętym audytorium, które odmawiają powtórzenia między zasobami. Ta lekcja modeluje pełną powierzchnię z trzema rolami — serwerem autoryzacyjnym, serwerem zasobów (serwer MCP) i klientem — abyś mógł prześledzić każdy krok od odkrycia do zweryfikowanego wywołania narzędzia.
>
> **Uwaga do specyfikacji (2025-11-25):** specyfikacja autoryzacji MCP z listopada 2025 zdegradowała Dynamic Client Registration z `SHOULD` do `MAY` i ustanowiła **Client ID Metadata Documents (CIMD)** jako zalecany domyślny mechanizm rejestracji. Ta lekcja uczy obu, w kolejności priorytetów specyfikacji, a kod zachowuje DCR dla celów demonstracyjnych, ponieważ jest w pełni samodzielny w jednym procesie.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16 (maszyna stanów OAuth 2.1), Phase 13 · 17 (bramki)
**Time:** ~90 minut

## Learning Objectives

- Odkryć serwer autoryzacyjny poprzez metadane RFC 8414 i zweryfikować kontrakt.
- Zaimplementować dynamiczną rejestrację klienta RFC 7591, aby klienci MCP rejestrowali się bez interwencji administratora.
- Buforować i odświeżać klucze JWKS zgodnie z harmonogramem, aby weryfikacja podpisu przetrwała rotację kluczy.
- Przypiąć tokeny do pojedynczego zasobu MCP za pomocą resource indicators RFC 8707 i odmówić ponownego użycia w ataku confused-deputy.
- Rozdzielić czysto trzy role — serwer autoryzacyjny, serwer zasobów, klient — tak aby każdy wymuszał tylko należące do niego kontrole.
- Odczytać macierz możliwości IdP i odmówić wdrożenia, gdy IdP nie może spełnić profilu autoryzacyjnego MCP.

## Problem

Symulator z Lekcji 16 uruchamia OAuth 2.1 w pamięci. Produkcja ma trzy luki operacyjne, których symulator tylko pamięciowy nie widzi.

Pierwszą luką jest rejestracja. Prawdziwa organizacja uruchamia setki serwerów MCP i tysiące klientów MCP. Operatorzy nie rejestrują ręcznie każdego użytkownika Cursor jako klienta OAuth. Specyfikacja z 2025-11-25 daje klientom hierarchię priorytetów rozwiązania tego: użyj wcześniej zarejestrowanego `client_id` jeśli go masz, w przeciwnym razie użyj **Client ID Metadata Document** (klient identyfikuje się URL-em HTTPS, który kontroluje, a serwer autoryzacyjny *pobiera* metadane), w przeciwnym razie użyj **dynamicznej rejestracji klienta RFC 7591** (klient *wysyła* `POST /register` i otrzymuje `client_id` na miejscu), w przeciwnym razie poproś użytkownika. CIMD jest zalecanym domyślnym rozwiązaniem, ponieważ całkowicie eliminuje rejestrację na serwer przy jednoczesnym zachowaniu modelu zaufania zakorzenionego w DNS; DCR jest zachowane dla kompatybilności wstecznej. Oba odkrywają swoje punkty wejścia z metadanych serwera autoryzacyjnego: `client_id_metadata_document_supported` dla CIMD, `registration_endpoint` dla DCR.

Drugą luką jest rotacja kluczy. Walidacja JWT zależy od kluczy podpisu serwera autoryzacyjnego, publikowanych jako zestaw kluczy webowych JSON (JWKS). Serwer autoryzacyjny rotuje je zgodnie z harmonogramem (często co godzinę, czasami szybciej w przypadku incydentu). Serwer MCP, który pobiera JWKS raz przy starcie, działa poprawnie do momentu rotacji — wtedy każde żądanie zawodzi aż do restartu. Produkcja implementuje JWKS jako buforowaną wartość z zadaniem odświeżania, które nadpisuje pamięć podręczną przed wygaśnięciem poprzednich kluczy, plus zapasowe pobranie na chybienie w pamięci podręcznej na wypadek, gdy pojawi się token podpisany kluczem nowszym niż pamięć podręczna.

Trzecią luką jest wiązanie audytorium. Lekcja 16 wprowadziła resource indicators RFC 8707. W produkcji ten wskaźnik staje się twardą kontrolą roszczeń przy każdym żądaniu. Serwer MCP porównuje `token.aud` z własnym kanonicznym URL-em zasobu i odrzuca niezgodności z HTTP 401. To jedyna obrona na poziomie protokołu przed nadrzędnym serwerem MCP (lub złośliwym klientem posiadającym token przeznaczony dla jednego serwera) odtwarzającym ten token przeciwko innemu serwerowi w tej samej siatce zaufania.

Ta lekcja mapuje każdą lukę na konkretny element powierzchni. Dokument metadanych to endpoint HTTP. Odświeżanie pamięci podręcznej JWKS to zaplanowane zadanie plus pamięć klucz-wartość. Walidacja JWT to procedura, którą serwer zasobów uruchamia przed wysłaniem dowolnego narzędzia. Utrzymuj trzy role oddzielone, a każda wymusza tylko kontrole, które do niej należą: serwer autoryzacyjny wydaje i rotuje klucze, serwer zasobów buforuje i waliduje, klient odkrywa i rejestruje się.

## Koncepcja

### RFC 8414 — Metadane Serwera Autoryzacyjnego OAuth

Dokument pod `/.well-known/oauth-authorization-server` opisuje wszystko, czego potrzebuje klient:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

Klient, któremu podano URL zasobu MCP, łączy odkrywanie: `oauth-protected-resource` z RFC 9728 (dokument serwera zasobów) podaje wystawcę, następnie `oauth-authorization-server` (to RFC) podaje każdy endpoint. Klient nigdy nie koduje na stałe URL-a autoryzacji.

Kontrakt, który weryfikujesz przed zaufaniem IdP dla MCP:

- `code_challenge_methods_supported` zawiera `S256` (PKCE zgodnie z RFC 7636). Specyfikacja jest wyraźna: jeśli to pole jest **nieobecne**, serwer autoryzacyjny nie obsługuje PKCE, a klient **MUSI** odmówić kontynuacji.
- `grant_types_supported` zawiera `authorization_code` i odrzuca `password` oraz `implicit`.
- Co najmniej jedna ścieżka rejestracji jest reklamowana: `client_id_metadata_document_supported: true` (CIMD, preferowane) **lub** `registration_endpoint` (RFC 7591 DCR, fallback). Każda z nich spełnia kontrakt; nie wymagasz już twardo DCR.
- `response_types_supported` to dokładnie `["code"]` dla OAuth 2.1.

Jeśli brakuje `S256`, serwer MCP odmawia wdrożenia względem tego IdP — nie ma trybu zdegradowanego dla PKCE. Jeśli *żadna* ścieżka rejestracji nie jest reklamowana i nie masz wcześniej zarejestrowanego `client_id`, również nie możesz się zarejestrować; manifest wdrożenia jest błędny, nie kod.

### RFC 9728 (powtórka) — Metadane Zasobu Chronionego

Lekcja 16 omówiła RFC 9728. Różnica w produkcji: ten dokument jest jedynym miejscem, w którym klient szuka serwerów autoryzacyjnych zaufanych przez *ten* serwer MCP. Pojedynczy serwer MCP może akceptować tokeny z wielu IdP (jeden dla pracowników, jeden dla partnerów). RFC 9728 deklaruje ten zestaw; RFC 8414 dokumentuje, co każdy IdP obsługuje.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents (zalecane domyślne rozwiązanie)

CIMD odwraca rejestrację z *push* na *pull*. Zamiast prosić serwer autoryzacyjny o wygenerowanie `client_id`, klient używa URL-a HTTPS, który kontroluje, **jako** swojego `client_id`. URL rozwiązuje się do dokumentu JSON z metadanymi; serwer autoryzacyjny pobiera go na żądanie podczas przepływu OAuth. Zaufanie jest zakorzenione w DNS: jeśli operator serwera ufa `app.example.com`, ufa klientowi obsługiwanemu z `https://app.example.com/client.json`. Brak rundy rejestracji, brak przestrzeni nazw `client_id` do wyczerpania, brak stanu na serwer do synchronizacji.

Dokument metadanych hostowany przez klienta:

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

Wartość `client_id` w dokumencie **MUSI** być równa URL-owi, z którego jest obsługiwana (serwer autoryzacyjny to weryfikuje; niezgodności są odrzucane). Serwer autoryzacyjny reklamuje wsparcie przez `client_id_metadata_document_supported: true` w swoich metadanych RFC 8414.

Dwa fakty bezpieczeństwa, które specyfikacja wyraźnie podkreśla:

- **SSRF.** Serwer autoryzacyjny pobiera URL dostarczony przez atakującego. Musi bronić się przed server-side request forgery (brak pobrań do wewnętrznych/admin endpointów).
- **Podszywanie się pod localhost.** Samo CIMD nie może powstrzymać lokalnego atakującego przed przejęciem URL-a metadanych legalnego klienta i związaniem dowolnego przekierowania `localhost`. Serwer autoryzacyjny **MUSI** wyraźnie wyświetlić nazwę hosta URI przekierowania podczas wyrażania zgody i **POWINIEN** ostrzegać przy przekierowaniach tylko do `localhost`.

Ponieważ CIMD nie potrzebuje stanu po stronie serwera, nie ma potrzeby uruchamiania rejestratora tak, jak wymaga tego DCR. Strona klienta jest tylko do odczytu: udostępnij swój dokument metadanych ze statycznego endpointu HTTPS i pozwól serwerowi autoryzacyjnemu go pobrać.

### RFC 7591 — Dynamiczna Rejestracja Klienta (fallback / kompatybilność wsteczna)

DCR jest teraz `MAY`, zachowane dla kompatybilności wstecznej z wdrożeniami sprzed 2025-11-25 i IdP, które jeszcze nie obsługują CIMD. Bez niego (i bez CIMD lub pre-rejestracji) każdy klient MCP (Cursor, Claude Desktop, niestandardowy agent) potrzebuje wymiany pozapasmowej z administratorem IdP. Z DCR klient wysyła:

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Serwer odpowiada z `client_id` i `registration_access_token` do późniejszych aktualizacji:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` jest właściwym domyślnym ustawieniem dla klientów MCP działających na urządzeniu użytkownika. Otrzymują tylko `client_id` — bez `client_secret` do wycieku. PKCE zapewnia proof-of-possession, którego potrzebują klienci publiczni.

Trzy pułapki produkcyjne:

- Endpoint rejestracji musi ograniczać szybkość według źródłowego IP. Bez tego wrogi aktor może zaskryptować miliony fałszywych rejestracji i wyczerpać przestrzeń nazw `client_id`. Uruchom kontrolę ograniczenia szybkości przed przetworzeniem żądania przez rejestrator.
- `software_statement` (podpisany JWT poręczający za klienta) jest wymagany przez niektóre korporacyjne IdP. Mock lekcji go pomija; produkcja wymaga kroku weryfikacji, który odrzuca niepodpisane rejestracje z innych niż localhost URI przekierowań.
- `registration_access_token` musi być przechowywany jako hash, nie jako czysty tekst. Kradzież tego tokena oznacza, że atakujący może przepisać URI przekierowań klienta.

### RFC 8707 (powtórka) — Resource Indicators

Lekcja 16 ustaliła kształt. Reguła produkcyjna: każde żądanie tokena zawiera `resource=<kanoniczny-url-mcp>`, a serwer MCP weryfikuje `token.aud` względem własnego URL-a zasobu przy każdym wywołaniu. Kanoniczny URI to *najbardziej szczegółowy* identyfikator serwera: używa małych liter w schemacie i hoście, bez fragmentu, i konwencjonalnie bez końcowego ukośnika. Komponent ścieżki **nie** jest usuwany według reguły — specyfikacja go zachowuje, gdy jest potrzebny do identyfikacji indywidualnego serwera MCP. `https://mcp.example.com`, `https://mcp.example.com/mcp`, `https://mcp.example.com:8443` i `https://mcp.example.com/server/mcp` są prawidłowymi kanonicznymi URI. Wybierz jeden na serwer i przypnij `aud` dokładnie do niego. (Mock tej lekcji używa audytoriów z samym hostem, takich jak `https://notes.example.com` dla zwięzłości; wdrożenie, które współdzieli kilka serwerów MCP pod jednym originem, odróżnia je ścieżką.)

### RFC 7636 (powtórka) — PKCE

PKCE jest obowiązkowe w OAuth 2.1. Przepływ kodu autoryzacyjnego lekcji zawsze zawiera `code_challenge` i `code_verifier`. Serwer odrzuca każde żądanie tokena bez weryfikatora lub z weryfikatorem, który nie haszuje do zapisanego wyzwania.

### Profil Autoryzacyjny MCP Spec 2025-11-25

Specyfikacja MCP (2025-11-25) jest precyzyjna co do tego, co musi robić warstwa autoryzacji serwera MCP:

- Zaimplementować metadane zasobu chronionego RFC 9728 i udostępnić ich lokalizację poprzez nagłówek `WWW-Authenticate: Bearer resource_metadata="..."` na 401 **lub** dobrze znany URI `/.well-known/oauth-protected-resource` (SEP-985 uczynił nagłówek opcjonalnym z dobrze znanym fallbackiem). Pole `authorization_servers` w metadanych **MUSI** wymieniać co najmniej jeden serwer.
- Akceptować tokeny tylko przez `Authorization: Bearer ...` przy **każdym** żądaniu — nigdy w ciągu zapytania, nigdy walidowane tylko na początku sesji.
- Walidować `aud`, `iss`, `exp` i wymagane zakresy na żądanie. Serwer **MUSI** zweryfikować, że token został wydany specjalnie dla niego (audytorium); brakujący lub niezgodny `aud` jest odrzucany, nigdy traktowany jako wieloznaczny.
- Na 401/403 zwracać `WWW-Authenticate: Bearer` z `error=...`, parametr `resource_metadata="<URL-PRM>"` (URL dokumentu metadanych, *nie* sam zasób) oraz `scope="..."` na `insufficient_scope` (403). Uwaga: parametr to `resource_metadata`, wskaźnik odkrywania — nie ma parametru `resource` w wyzwaniu.
- Odkrywanie serwera autoryzacyjnego akceptuje **albo** metadane OAuth RFC 8414 **albo** OpenID Connect Discovery 1.0; klienci muszą spróbować obu dobrze znanych sufiksów w kolejności priorytetów.
- Klient (nie serwer) broni się przed **atakami mix-up**: zapisuje oczekiwanego `issuer` przed przekierowaniem i waliduje parametr odpowiedzi autoryzacyjnej `iss` (RFC 9207) przed realizacją kodu. Samo PKCE nie powstrzymuje mix-up, ponieważ klient przekazuje swój `code_verifier` do dowolnego endpointu tokena, do którego został skierowany.

Projekt OAuth 2.1 jest podłożem; RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD są powierzchnią; specyfikacja MCP jest profilem.

### Macierz możliwości IdP

Nie każdy IdP obsługuje pełny profil MCP. Poniższa macierz dokumentuje faktyczne możliwości według specyfikacji z 2025-11-25. Jest to *brama wdrożeniowa*, nie rekomendacja.

CIMD pojawiło się w specyfikacji z 2025-11-25, a bazowy projekt OAuth został przyjęty dopiero w październiku 2025, więc wsparcie dostawców wciąż nadchodzi — traktuj "CIMD" poniżej jako "gdzie to jest dzisiaj, zweryfikuj w swojej dzierżawie", a nie stałe oświadczenie.

| Kategoria IdP | AS metadata (8414/OIDC) | CIMD | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | Uwagi |
|---|---|---|---|---|---|---|
| Samodzielnie hostowane (Keycloak) | tak | pojawia się | tak | tak (od 24.x) | tak | Referencyjny IdP dla profilu MCP w tej lekcji; pełna ścieżka DCR od końca do końca, CIMD śledzący nową specyfikację. |
| Korporacyjne SSO (Microsoft Entra ID) | tak | pojawia się | tak (wyższe poziomy) | tak | tak | Dostępność DCR różni się w zależności od poziomu dzierżawy; zweryfikuj w docelowej dzierżawie przed wdrożeniem. |
| Korporacyjne SSO (Okta) | tak | pojawia się | tak (Okta CIC / Auth0) | tak | tak | DCR dostępne na Auth0 (obecnie Okta CIC); klasyczne orgi Okta wymagają administracyjnej pre-rejestracji. |
| IdP logowania społecznościowego (ogólne) | różnie | nie | rzadko | rzadko | tak | Większość IdP społecznościowych traktuje klientów jako statycznych partnerów; brak samoobsługowej rejestracji. Używaj tylko jako źródła tożsamości, nałóż własny serwer autoryzacyjny świadomy MCP na wierzchu. |
| Niestandardowe / własnej roboty | zależy | zależy | zależy | zależy | zależy | Jeśli dostarczasz własne, dostarcz pełny profil i preferuj CIMD. Pominięcie PKCE lub wiązania audytorium łamie kontrakt autoryzacyjny MCP. |

Zasada odmowy dla manifestu wdrożeniowego: jeśli wybrany IdP nie wymienia `S256` w `code_challenge_methods_supported`, serwer MCP odmawia uruchomienia — PKCE nie ma trybu zdegradowanego. Rejestracja jest łagodniejszą bramą: potrzebujesz *jednej* działającej ścieżki (wcześniej zarejestrowanego `client_id`, `client_id_metadata_document_supported: true` lub `registration_endpoint`). Brak DCR nie jest już sam w sobie wyzwalaczem odmowy, ponieważ CIMD lub pre-rejestracja mogą to pokryć.

### Wzorzec odświeżania JWKS (rotacja na AS, odświeżanie na serwerze zasobów)

Rozdzielaj dwa czasowniki, ponieważ ich mylenie jest prawdziwym błędem produkcyjnym:

- **Rotacja** to to, co robi *serwer autoryzacyjny*: tworzy nowy klucz podpisu, publikuje go w JWKS, później wycofuje stary. Serwer zasobów nie ma w tym udziału i nie może tego zrobić — nie posiada prywatnych kluczy IdP.
- **Odświeżanie** to to, co robi *serwer zasobów*: ponownie `GET`-uje opublikowany JWKS do swojej pamięci podręcznej. To jedyna akcja JWKS, jaką kiedykolwiek wykonuje serwer zasobów.

Tryb awarii produkcyjnej to nieaktualna pamięć podręczna. Rozwiąż to za pomocą zaplanowanego zadania odświeżania plus pamięci klucz-wartość. Serwer zasobów uruchamia zadanie (cron, timer, cokolwiek oferuje twoje środowisko), które w stałym interwale pobiera `<issuer>/.well-known/jwks.json` i nadpisuje `cache[issuer] = {keys, fetched_at}`. Walidator czyta z tej pamięci podręcznej. Token, którego `kid` brakuje w pamięci podręcznej, wyzwala **jedno** synchroniczne odświeżenie jako fallback, a następnie ponownie sprawdza. To obsługuje dwa przypadki jednocześnie: zaplanowane odświeżenie i okna nakładania się kluczy, gdzie token podpisany nowym kluczem pojawia się przed następnym zaplanowanym odświeżeniem.

**Fallback musi być ponownym pobraniem, nigdy rotacją.** Jeśli podłączysz ścieżkę chybienia w pamięci podręcznej do rotacji i tworzenia, dwie rzeczy się psują: (1) utworzenie nowego klucza produkuje `kid`, który *nadal* nie pasuje do tokena, więc wyszukiwanie i tak zawodzi; oraz (2) atakujący, który wysyła tokeny z losowymi wartościami `kid`, wymusza nieograniczoną serię tworzeń kluczy — samospowodowany DoS. Ponowne pobranie jest idempotentne, więc fałszywy `kid` kosztuje co najwyżej jedno zmarnowane pobranie.

Kształt pamięci podręcznej:

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

Dwa klucze jednocześnie to stan ustalony. Serwery autoryzacyjne rotują, wprowadzając następny klucz (`k_2026_04`) przed wycofaniem poprzedniego (`k_2026_03`), więc tokeny wydane pod starym kluczem pozostają ważne do ich wygaśnięcia. Pamięć podręczna przechowuje sumę; walidator wybiera po `kid`.

### Procedura walidacji

Serwer MCP uruchamia walidację przed wysłaniem dowolnego narzędzia. Kształt używany w `code/main.py`:

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` dekoduje JWT, znajduje klucz podpisu z pamięci podręcznej JWKS (odświeżając raz przy chybieniu), weryfikuje podpis, a następnie sprawdza `iss` względem listy dozwolonych, `aud` względem kanonicznego zasobu tego serwera, `exp` i wymagany zakres — zwracając wyzwanie `WWW-Authenticate` przy pierwszym błędzie. Utrzymanie tego jako pojedynczej procedury na serwerze zasobów oznacza, że każdy punkt wejścia (każde wywołanie narzędzia, każdy transport) przechodzi przez te same kontrole; nie ma ścieżki, która dociera do narzędzia bez walidacji.

### Przykład odtworzenia audytorium (ograniczenie uprawnień tokena dostępu)

Serwer A (`notes.example.com`) i Serwer B (`tasks.example.com`) oba rejestrują się względem tego samego serwera autoryzacyjnego. Serwer A jest skompromitowany. Atakujący bierze token notatek użytkownika i odtwarza go przeciwko Serwerowi B.

Walidator Serwera B:

1. Dekoduje JWT, pobiera JWKS po `kid`, weryfikuje podpis.
2. Sprawdza `iss` względem `authorization_servers` z metadanych zasobu chronionego. (Zaliczony — ten sam IdP.)
3. Sprawdza `aud == "https://tasks.example.com"`. (Niezaliczony — `aud` tokena to `https://notes.example.com`.)
4. Zwraca 401 z `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`.

Roszczenie audytorium jest jedyną obroną przed tym atakiem na poziomie protokołu. Pomijanie go dla wydajności to najczęstszy błąd produkcyjny; walidator musi działać przy każdym żądaniu, nie tylko na początku sesji. Specyfikacja nazywa to **ograniczeniem uprawnień tokena dostępu**: serwer MCP `MUSI` odrzucić każdy token, który nie wymienia go w audytorium.

> **Uwaga terminologiczna.** Specyfikacja rezerwuje termin *confused deputy* dla powiązanego, ale odrębnego problemu: serwer MCP działający jako **proxy** OAuth do API strony trzeciej, używający statycznego ID klienta, który przekazuje token bez uzyskiwania zgody użytkownika na klienta. Wiązanie audytorium naprawia odtworzenie powyżej; poprawka confused deputy to zgoda na klienta **plus** nigdy nieprzekazywanie przychodzącego tokena do nadrzędnych API (serwer MCP `MUSI` uzyskać własny oddzielny nadrzędny token).

### Ataki mix-up (obrona po stronie klienta, której serwer nie może zapewnić)

Klient rozmawia z wieloma serwerami autoryzacyjnymi w ciągu swojego życia. Złośliwy AS może próbować sprawić, by klient zrealizował kod autoryzacyjny uczciwego AS na endpointcie tokena atakującego. Wiązanie audytorium tutaj nie pomaga — atak ma miejsce, zanim jakikolwiek token istnieje. Obrona żyje po stronie klienta (RFC 9207):

1. Przed przekierowaniem klient zapisuje oczekiwanego `issuer` ze zweryfikowanych metadanych AS.
2. Przy odpowiedzi autoryzacyjnej klient porównuje zwrócony parametr `iss` z zapisanym wystawcą (proste porównanie ciągów, bez normalizacji) przed wysłaniem kodu gdziekolwiek.
3. Niezgodność (lub brak `iss`, gdy AS reklamował `authorization_response_iss_parameter_supported`) → odrzuć i nie wyświetlaj nawet pól `error`.

Samo PKCE nie powstrzymuje mix-up, ponieważ klient przekazuje swój `code_verifier` do dowolnego endpointu tokena, do którego został skierowany. Dlatego specyfikacja zapisuje wystawcę na żądanie obok weryfikatora PKCE i `state`.

### Tryby awarii

- **Nieaktualny JWKS.** Walidator odrzuca ważne tokeny po rotacji klucza przez AS. Poprawką jest wzór cron-refresh + cache-miss-refetch powyżej. Nigdy nie buforuj JWKS bez zadania odświeżania.
- **Rotacja jako fallback.** Podłączenie ścieżki chybienia w pamięci podręcznej do rotacji i tworzenia zamiast ponownego pobrania to prawdziwy błąd: nigdy nie produkuje brakującego `kid` i zamienia wartości `kid` kontrolowane przez atakującego w DoS tworzenia kluczy. Fallback musi być idempotentnym `refresh-jwks`.
- **Brak roszczenia `aud`.** Niektóre IdP domyślnie pomijają `aud`, chyba że `resource` jest obecny w żądaniu tokena. Walidator musi odrzucać tokeny z brakującym `aud`, nie traktować braku jako wieloznacznego.
- **Mix-up przez brak kontroli `iss`.** Klient, który nie waliduje parametru odpowiedzi autoryzacyjnej `iss` (RFC 9207) względem wystawcy zapisanego przed przekierowaniem, może zostać skierowany do realizacji kodu uczciwego AS na endpointcie tokena atakującego. To błąd po stronie klienta; serwer zasobów nie może go skompensować.
- **Wyścig o uaktualnienie zakresu.** Dwa równoczesne przepływy krokowe dla tego samego użytkownika mogą oba się powieść i wyprodukować dwa tokeny dostępu z różnymi zakresami. Walidator musi użyć tokena przedstawionego w żądaniu, a nie szukać "bieżącego zakresu użytkownika" — to tworzy okno TOCTOU.
- **Kradzież tokena rejestracji.** Wyciekły `registration_access_token` pozwala atakującemu przepisać URI przekierowań. Haszuj je w spoczynku; wymagaj od klienta przedstawienia czystego tekstu przy każdej aktualizacji; rotuj przy podejrzeniu.
- **Niespinnedowany `iss`.** Walidator, który akceptuje dowolny `iss`, pozwala atakującemu uruchomić własny serwer autoryzacyjny, zarejestrować klienta dla docelowego audytorium i wydawać tokeny. Lista `authorization_servers` w metadanych zasobu chronionego jest listą dozwolonych; wymuszaj ją.

## Użyj tego

`code/main.py` przeprowadza przez pełny przepływ produkcyjny ze stdlib Python i trzema rolami — `AuthorizationServer`, `ResourceServer` i `Client`. Przepływ:

1. Serwer autoryzacyjny publikuje metadane RFC 8414 pod `/.well-known/oauth-authorization-server`.
2. Klient MCP wywołuje endpoint metadanych i sprawdza swoje opcje rejestracji (`client_id_metadata_document_supported` dla CIMD, `registration_endpoint` dla DCR) oraz wsparcie PKCE `S256`.
3. Przykład idzie ścieżką DCR: klient wysyła POST do `/register` (RFC 7591) i otrzymuje `client_id`. (Klient CIMD zamiast tego przedstawiłby własny URL HTTPS `client_id` i pominąłby ten krok.)
4. Klient MCP uruchamia przepływ kodu autoryzacyjnego chronionego PKCE (RFC 7636) ze wskaźnikiem `resource` (RFC 8707).
5. Klient MCP wywołuje narzędzie na serwerze MCP z `Authorization: Bearer ...`.
6. Serwer MCP uruchamia `validate`, znajdując klucz podpisu z pamięci podręcznej JWKS.
7. IdP rotuje klucz; zaplanowane odświeżanie ponownie pobiera JWKS do pamięci podręcznej.
8. Następne wywołanie waliduje względem odświeżonych kluczy bez restartu, a poprzedni token wciąż jest ważny w oknie nakładania.
9. Próba odtworzenia audytorium względem innego zasobu MCP otrzymuje 401 z `audience mismatch` i wskaźnikiem `resource_metadata`.

JWT używa tutaj HS256 ze współdzielonym sekretem (aby lekcja działała tylko na stdlib). Produkcja używa RS256 lub EdDSA ze wzorcem JWKS powyżej; logika walidacji jest poza tym identyczna. Ponieważ IdP i serwer zasobów żyją w jednym procesie, `refresh_jwks` czyta listę kluczy serwera autoryzacyjnego bezpośrednio; przez sieć jest to HTTP `GET` do `jwks_uri`.

## Ship It

Ta lekcja produkuje `outputs/skill-mcp-auth.md`. Dla konfiguracji serwera MCP i zestawu możliwości IdP, skill emituje powierzchnię autoryzacyjną do uruchomienia — metadane zasobu chronionego, ścieżkę rejestracji do użycia (CIMD, pre-rejestracja lub fallback DCR), harmonogram odświeżania JWKS, mapowanie zakresów i reguły odmowy do zastosowania, gdy IdP nie obsługuje pełnego profilu RFC.

## Ćwiczenia

1. Uruchom `code/main.py`. Prześledź przepływ. Zwróć uwagę, jak IdP rotuje klucz w kroku 6, zaplanowane `refresh_jwks` ponownie pobiera opublikowany zestaw, a zarówno stary token (okno nakładania), jak i świeży token walidują bez restartu.

2. Dodaj nowy IdP do listy `authorization_servers` w metadanych zasobu chronionego. Wydaj token podpisany przez nowy IdP i potwierdź, że walidator go akceptuje. Wydaj token podpisany przez niewymieniony IdP i potwierdź, że walidator odrzuca z `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`.

3. Dodaj kontrolę ograniczenia szybkości do `register_client`, która uruchamia się przed przyjęciem żądania przez rejestrator. Użyj zasobnika tokenowego na źródłowy IP przechowywanego w małym słowniku kluczowanym przez IP.

4. Przeczytaj RFC 7591 i zidentyfikuj dwa pola, których handler `/register` lekcji nie waliduje. Dodaj walidację. (Wskazówka: `software_statement` i schemat URI `redirect_uris`.)

5. Dodaj ścieżkę Client ID Metadata Document. Udostępnij `client.json`, którego `client_id` równa się własnemu URL-owi, i pozwól serwerowi autoryzacyjnemu go pobrać i zweryfikować (odrzuć, jeśli `client_id` ≠ URL). Potwierdź, że klient CIMD rejestruje się bez wywołania `register_client`.

6. Udowodnij poprawkę DoS. Wyślij do walidatora token z losowym `kid` i potwierdź, że `refresh_jwks` uruchamia się co najwyżej raz, a liczba kluczy serwera autoryzacyjnego nie rośnie. Następnie celowo przełącz fallback na rotację i twórz i obserwuj wzrost liczby kluczy na każdy fałszywy token — przywróć ponowne pobranie później.

7. Zaimplementuj kontrolę `iss` RFC 9207 po stronie klienta z sekcji o mix-up: zapisz oczekiwanego wystawcę przed żądaniem autoryzacji, a następnie odrzuć odpowiedź autoryzacyjną, której `iss` nie pasuje.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| ASM | "Dokument metadanych OAuth" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "URL metadanych klienta" | Client ID Metadata Document — URL HTTPS używany jako `client_id`; AS pobiera JSON. Zalecane domyślne rozwiązanie od 2025-11-25 |
| DCR | "Samoobsługowa rejestracja klienta" | Przepływ RFC 7591 `POST /register`; zdegradowany do `MAY` fallback w 2025-11-25 |
| JWKS | "Klucze publiczne do walidacji JWT" | Zestaw kluczy webowych JSON, pobierany z `jwks_uri`, indeksowany przez `kid` |
| Rotacja vs odświeżanie | "Aktualizowanie kluczy" | *Rotacja* = AS tworzy/wycofuje klucze podpisu; *odświeżanie* = serwer zasobów ponownie pobiera opublikowany zestaw. Serwery zasobów tylko odświeżają |
| Resource indicator | "Parametr audytorium" | Parametr `resource` RFC 8707 przypinający token do jednego serwera |
| Roszczenie `aud` | "Audytorium" | Roszczenie JWT, które walidator porównuje z kanonicznym URL-em zasobu |
| Odtworzenie audytorium | "Odtworzenie tokena" | Token wydany dla Serwera A przedstawiony Serwerowi B; bronione przez walidację audytorium (spec: ograniczenie uprawnień tokena dostępu) |
| Confused deputy | "Niewłaściwe użycie tokena proxy" | Proxy MCP ze statycznym ID klienta przekazujące token bez zgody na klienta; odrębne od odtworzenia audytorium |
| Atak mix-up | "Zły endpoint tokena" | Klient skierowany do realizacji kodu uczciwego AS na endpointcie atakującego; bronione po stronie klienta przez `iss` RFC 9207 |
| Lista dozwolonych `iss` | "Zaufane serwery autoryzacyjne" | Zestaw wymieniony w `authorization_servers` metadanych zasobu chronionego |
| `resource_metadata` | "Gdzie znaleźć dokument PRM" | Parametr `WWW-Authenticate` wskazujący URL metadanych RFC 9728 na 401/403 |
| Klient publiczny | "Klient natywny lub przeglądarkowy" | Klient OAuth bez `client_secret`; PKCE kompensuje |
| `WWW-Authenticate` | "Nagłówek odpowiedzi 401/403" | Nosi dyrektywy `Bearer error=...` kierujące odzyskiwaniem klienta |

## Dalsza lektura

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — profil autoryzacyjny MCP, który ta lekcja implementuje
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — co się zmieniło w 2025-11-25 (CIMD, XAA, degradacja DCR)
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — uzasadnienie CIMD-over-DCR
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — kontrakt odkrywania
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR (ścieżka fallback)
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — proof-of-possession klienta publicznego
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — przypinanie audytorium
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — odkrywanie serwera zasobów
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — parametr `iss` broniący przed atakami mix-up
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — skonsolidowane podłoże OAuth
