# Bramki i Rejestry MCP — Korporacyjne Płaszczyzny Kontroli

> Przedsiębiorstwa nie mogą pozwolić każdemu programiście na instalowanie dowolnych serwerów MCP. Bramka centralizuje uwierzytelnianie, RBAC, audyt, ograniczanie szybkości, buforowanie i wykrywanie zatruwania narzędzi, a następnie udostępnia scaloną powierzchnię narzędzi jako pojedynczy endpoint MCP. Oficjalny Rejestr MCP (Anthropic + GitHub + PulseMCP + Microsoft, z weryfikacją przestrzeni nazw) jest kanonicznym źródłem nadrzędnym. Ta lekcja wskazuje, gdzie pasuje bramka, przeprowadza przez minimalną implementację i przedstawia przegląd dostawców w 2026 roku.

**Type:** Learn
**Languages:** Python (stdlib, minimalna bramka)
**Prerequisites:** Phase 13 · 15 (zatruwanie narzędzi), Phase 13 · 16 (OAuth 2.1)
**Time:** ~45 minut

## Learning Objectives

- Wyjaśnić, gdzie znajduje się bramka MCP (między klientami MCP a wieloma zapleczowymi serwerami MCP).
- Zaimplementować pięć obowiązków bramki: uwierzytelnianie, RBAC, audyt, ograniczanie szybkości, polityka.
- Wymusić zatwierdzony manifest skrótów narzędzi na poziomie bramki.
- Odróżnić Oficjalny Rejestr MCP od metarejestrów (Glama, MCPMarket, MCP.so, Smithery, LobeHub).

## Problem

Firma z listy Fortune 500 ma 30 zatwierdzonych serwerów MCP, 5000 programistów, wymagania dotyczące zgodności i audytu oraz zespół bezpieczeństwa, który chce scentralizowanej polityki. Pozwalanie każdemu programiście na instalowanie dowolnych serwerów w ich IDE jest nie do przyjęcia.

Wzorzec bramki:

1. Bramka działa jako pojedynczy endpoint Streamable HTTP, z którym łączą się programiści.
2. Bramka przechowuje poświadczenia dla każdego zapleczowego serwera MCP.
3. Każde żądanie programisty jest uwierzytelniane i ograniczane zakresem przez własne OAuth bramki.
4. Bramka kieruje wywołanie do zapleczowego serwera, stosując politykę.
5. Wszystkie wywołania są rejestrowane do audytu.

Cloudflare MCP Portals, Kong AI Gateway, IBM ContextForge, MintMCP, TrueFoundry, Envoy AI Gateway — wszystkie dostarczyły bramki lub funkcje bramek w latach 2025-2026.

Tymczasem Oficjalny Rejestr MCP został uruchomiony jako kanoniczne źródło nadrzędne: wyselekcjonowany, z weryfikacją przestrzeni nazw, nazwany w formacie odwrotnej notacji DNS, z którego bramka może pobierać dane. Metarejestry (Glama, MCPMarket, MCP.so, Smithery, LobeHub) agregują serwery z wielu źródeł.

## Koncepcja

### Pięć obowiązków bramki

1. **Uwierzytelnianie.** OAuth 2.1 do identyfikacji programisty; mapuje na role użytkowników.
2. **RBAC.** Polityka na użytkownika: które serwery, które narzędzia, które zakresy.
3. **Audyt.** Każde wywołanie rejestrowane z kto, co, kiedy, wynik.
4. **Ograniczanie szybkości.** Limity na użytkownika / narzędzie / serwer zapobiegające nadużyciom.
5. **Polityka.** Odrzucanie zatrutych opisów, wymuszanie Reguły Dwóch, usuwanie PII.

### Bramka jako pojedynczy endpoint

Dla programistów bramka wygląda jak jeden serwer MCP. Wewnętrznie kieruje do N zapleczy. Identyfikatory sesji (Phase 13 · 09) są przepisywane na granicy.

### Przechowywanie poświadczeń

Programiści nigdy nie widzą tokenów zaplecza. Bramka je przechowuje (lub proxy do dostawcy tożsamości, który to robi). Programista z `notes:read` na bramce może przechodnie uzyskać dostęp do serwera MCP notatek z własnymi poświadczeniami zaplecza bramki — ale tylko zgodnie z polityką wiążącą dostęp przechodni.

### Przypinanie skrótów narzędzi na bramce

Bramka przechowuje manifest zatwierdzonych opisów narzędzi (SHA256). Podczas odkrywania pobiera `tools/list` każdego zaplecza, porównuje skróty z manifestem i usuwa każde narzędzie, którego opis uległ zmianie. To scentralizowana obrona przed rug-pull z Phase 13 · 15.

### Polityka jako kod

Zaawansowane bramki wyrażają politykę w OPA/Rego, Kyverno lub Styra. Reguły takie jak "użytkownik `alice` może wywołać `github.open_pr` tylko w repozytoriach w organizacji `acme`" są kodowane deklaratywnie. Proste bramki używają ręcznie napisanego Pythona. Obie formy są poprawne.

### Routing świadomy sesji

Gdy sesja użytkownika zawiera mieszankę serwerów, bramka multipleksuje: pojedyncza sesja MCP programisty zawiera N sesji zaplecza, po jednej na serwer. Powiadomienia z dowolnego zaplecza są kierowane przez bramkę do sesji programisty.

### Scalanie przestrzeni nazw

Bramki scalają przestrzenie nazw narzędzi ze wszystkich zapleczy, zwykle z prefiksem przy kolizji. `github.open_pr`, `notes.search`. To czyni routing jednoznacznym.

### Rejestry

- **Oficjalny Rejestr MCP (`registry.modelcontextprotocol.io`).** Uruchomiony pod opieką Anthropic, GitHub, PulseMCP, Microsoft. Zweryfikowany pod kątem przestrzeni nazw (odwrotny DNS: `io.github.user/server`). Wstępnie filtrowany pod kątem podstawowej jakości.
- **Glama.** Metarejestr zorientowany na wyszukiwanie, agregujący wiele źródeł.
- **MCPMarket.** Katalog o nachyleniu komercyjnym z listami dostawców.
- **MCP.so.** Społecznościowy katalog; otwarte zgłoszenia.
- **Smithery.** Przepływ instalacji w stylu menedżera pakietów.
- **LobeHub.** Rejestr zintegrowany z UI w aplikacji LobeChat.

Bramki korporacyjne domyślnie pobierają z Oficjalnego Rejestru, pozwalają na administracyjnie wyselekcjonowane dodatki z metarejestrów i odrzucają wszystko, co nie jest przypięte.

### Nazewnictwo odwrotnego DNS

Oficjalny Rejestr wymaga nazw w formacie odwrotnego DNS dla publicznych serwerów: `io.github.alice/notes`. Przestrzenie nazw zapobiegają zawłaszczaniu i ułatwiają delegowanie zaufania.

### Przegląd dostawców, kwiecień 2026

| Dostawca | Mocna strona |
|--------|----------|
| Cloudflare MCP Portals | Hostowane na brzegu sieci; zintegrowane OAuth; darmowy poziom |
| Kong AI Gateway | Natywne dla K8s; szczegółowa polityka; logi do OpenTelemetry |
| IBM ContextForge | Korporacyjne IAM; zgodność; eksport audytu |
| TrueFoundry | Zorientowane na DevOps; najpierw metryki |
| MintMCP | Zorientowane na platformę programistyczną |
| Envoy AI Gateway | Otwarte źródło; konfigurowalne filtry |

Phase 17 (infrastruktura produkcyjna) zagłębia się w operacje bramek.

## Użyj tego

`code/main.py` dostarcza minimalną bramkę w ~150 liniach: uwierzytelnia użytkowników przez fikcyjny token Bearer, przechowuje politykę RBAC na użytkownika, kieruje żądania do dwóch zapleczowych serwerów MCP, zapisuje każde wywołanie do dziennika audytu, wymusza limit szybkości i odrzuca każde narzędzie zaplecza, którego skrót opisu nie zgadza się z przypiętym manifestem.

Na co zwrócić uwagę:

- Słownik `RBAC` kluczowany przez `user_id` z dozwolonymi wpisami `server_tool`.
- `AUDIT_LOG` to lista zdarzeń tylko do dołączania.
- Limit szybkości używa zasobnika tokenowego na użytkownika.
- Przypięty manifest to słownik `server::tool -> hash`.

## Ship It

Ta lekcja produkuje `outputs/skill-gateway-bootstrap.md`. Dla korporacyjnego planu MCP (użytkownicy, zaplecza, zgodność), skill produkuje specyfikację konfiguracji bramki.

## Ćwiczenia

1. Uruchom `code/main.py`. Wykonaj wywołanie jako dozwolony użytkownik; następnie jako niedozwolony użytkownik; następnie seria przekraczająca limit szybkości. Zweryfikuj wszystkie trzy scenariusze.

2. Dodaj politykę usuwającą PII z wyników przed zwróceniem do klienta. Użyj prostego regexu dla ciągów w kształcie SSN; zwróć uwagę na lukę (e-maile, numery telefonów).

3. Rozszerz dziennik audytu, aby emitował span-y OpenTelemetry GenAI. Phase 13 · 20 obejmuje dokładne atrybuty.

4. Zaprojektuj politykę RBAC dla 50-osobowego zespołu z pięcioma zapleczami (notes, github, postgres, jira, slack). Kto ma dostęp tylko do odczytu do każdego? Kto ma dostęp do zapisu?

5. Przeczytaj post Cloudflare o korporacyjnym MCP od początku do końca. Zidentyfikuj jedną funkcję, którą Cloudflare dostarcza, a której ta bramka stdlib nie ma.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Bramka | "Proxy MCP" | Serwer centralizujący między klientami a zapleczami |
| Przechowywanie poświadczeń | "Tokeny zaplecza pozostają po stronie serwera" | Programiści nigdy nie widzą nadrzędnych tokenów |
| Routing świadomy sesji | "Sesja wielozapleczowa" | Bramka multipleksuje N sesji zaplecza na sesję programisty |
| Przypinanie skrótów narzędzi | "Zatwierdzony manifest" | SHA256 każdego zatwierdzonego opisu narzędzia; centralnie blokuje rug-pull |
| RBAC | "Polityka na użytkownika" | Kontrola dostępu oparta na rolach dla narzędzi i serwerów |
| Polityka jako kod | "Reguły deklaratywne" | Polityki OPA/Rego, Kyverno, Styra wymuszane na bramce |
| Dziennik audytu | "Kto, co, kiedy" | Lista zdarzeń tylko do dołączania dla zgodności |
| Limit szybkości | "Zasobnik tokenowy na użytkownika" | Limity na minutę zapobiegające nadużyciom |
| Oficjalny Rejestr MCP | "Kanoniczne źródło nadrzędne" | `registry.modelcontextprotocol.io`, zweryfikowana przestrzeń nazw |
| Nazewnictwo odwrotnego DNS | "Przestrzeń nazw rejestru" | Konwencja `io.github.user/server` |

## Dalsza lektura

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — kanoniczne źródło nadrzędne, zweryfikowana przestrzeń nazw
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — wzorzec bramki z OAuth i polityką
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — referencyjna bramka open-source
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — artykuł porównawczy funkcji
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — korporacyjna bramka od IBM
