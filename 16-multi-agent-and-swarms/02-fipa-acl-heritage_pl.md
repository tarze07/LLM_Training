# Dziedzictwo FIPA-ACL i Aktów Mowy

> Przed MCP, przed A2A, był FIPA-ACL. W 2000 roku IEEE Foundation for Intelligent Physical Agents ratyfikowała język komunikacji agentów z dwudziestoma performatywami, dwoma językami treści i zestawem protokołów interakcji — contract net, subscribe/notify, request-when. Zniknął z przemysłu, ponieważ narzut ontologiczny był zbyt ciężki dla sieci, ale odrodzenie systemów wieloagentowych opartych na LLM po cichu implementuje te same pomysły bez formalnej semantyki: kontrakty JSON zastępują performatywy, język naturalny zastępuje ontologie. Ta lekcja traktuje FIPA-ACL poważnie, abyś mógł zobaczyć, które decyzje protokołów z 2026 roku są wynalazkami na nowo, które nowością, a gdzie obecna fala odkryje na nowo problemy, które lata 2000. już rozwiązały.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## Problem

Krajobraz protokołów agentowych w 2026 roku jest zatłoczony: MCP dla narzędzi, A2A dla agentów, ACP dla audytu korporacyjnego, ANP dla zdecentralizowanego zaufania, NLIP dla treści w języku naturalnym, plus CA-MCP i kilkadziesiąt propozycji badawczych. Każda specyfikacja ogłasza się fundamentalną.

Uczciwa ocena jest taka, że większość z nich odkrywa na nowo bardzo konkretne, dwudziestoletnie drzewo decyzyjne. Teoria aktów mowy od Austina (1962) i Searle'a (1969) dała nam „wypowiedzi są działaniami." KQML (1993) przekształcił to w protokół transmisyjny. FIPA-ACL (ratyfikowany 2000) stworzył referencyjną standaryzację: dwadzieścia performatywów, języki treści SL0/SL1, protokoły interakcji dla contract-net i subscribe-notify. JADE i JACK były referencyjnymi platformami Java. Wysiłek wygasł około 2010 roku, ponieważ narzut ontologiczny był zbyt ciężki, a sieć wygrywała.

Kiedy patrzysz na `tools/call` MCP, cykl życia zadań A2A czy współdzielone przechowywanie kontekstu CA-MCP, patrzysz na łagodniejszą, natywnie JSON-ową wersję decyzji FIPA. Znajomość dziedzictwa mówi ci dwie rzeczy: które nowe „innowacje" są w rzeczywistości wynalazkami na nowo i które stare tryby awarii nowe specyfikacje odkryją na nowo.

## Concept

### Akty mowy, w jednym akapicie

Austin zauważył, że niektóre zdania nie opisują świata — one go zmieniają. „Obiecuję." „Proszę." „Ogłaszam." Nazwał je wypowiedziami performatywnymi. Searle sformalizował pięć kategorii: asertywne, dyrektywne, komisywne, ekspresywne, deklaratywne. KQML (Finin i in., 1993) uczynił to operacyjnym dla agentów programowych: wiadomość to performatywa (działanie) plus treść (czego dotyczy działanie). FIPA-ACL oczyścił luki KQML i ustandaryzował dwadzieścia performatyw.

### Dwadzieścia performatyw FIPA (lista częściowa)

| Performatywa | Intencja |
|---|---|
| `inform` | „Mówię ci, że P jest prawdą" |
| `request` | „Proszę cię o zrobienie X" |
| `query-if` | „Czy P jest prawdą?" |
| `query-ref` | „Jaka jest wartość X?" |
| `propose` | „Proponuję, żebyśmy zrobili X" |
| `accept-proposal` | „Akceptuję propozycję" |
| `reject-proposal` | „Odrzucam propozycję" |
| `agree` | „Zgadzam się zrobić X" |
| `refuse` | „Odmawiam zrobienia X" |
| `confirm` | „Potwierdzam, że P jest prawdą" |
| `disconfirm` | „Zaprzeczam P" |
| `not-understood` | „Twoja wiadomość nie została sparsowana" |
| `cfp` | „Zapytanie o propozycje dotyczące X" |
| `subscribe` | „Powiadom mnie, gdy X się zmieni" |
| `cancel` | „Anuluj trwające X" |
| `failure` | „Próbowałem X i nie udało się" |

Pełna lista znajduje się w `fipa00037.pdf` (FIPA ACL Message Structure). Chodzi nie o jej zapamiętanie — chodzi o to, że każda z nich odpowiada prymitywowi, który protokół LLM prędzej czy później dodaje ponownie.

### Kanoniczna wiadomość FIPA-ACL

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

Siedem pól niesie kopertę protokołu; jedno pole (`content`) niesie ładunek. Reszta pól to dokładnie to, co wynajdujesz na nowo za każdym razem, gdy dodajesz ponowienia, wątkowanie i ontologię do protokołu JSON.

### Dwie platformy dziedzictwa

**JADE** (Java Agent DEvelopment framework, 1999–2020s) był najczęściej używanym środowiskiem wykonawczym zgodnym z FIPA. Agenci rozszerzali klasę bazową, wymieniali wiadomości ACL, działali wewnątrz kontenerów i koordynowali za pomocą „zachowań." Biblioteka protokołów interakcji zawierała contract-net, subscribe-notify, request-when i propose-accept.

**JACK** (Agent Oriented Software, komercyjny) kładł nacisk na rozumowanie BDI (Belief-Desire-Intention) na bazie wiadomości FIPA. Bardziej formalny, mniej przyjęty.

Oba podupadły, gdy stos webowy pochłonął przypadki użycia wieloagentowe. MCP i A2A są „kontenerami" wykonawczymi 2026 roku.

### Dlaczego FIPA zniknął

- **Narzut ontologiczny.** FIPA wymagał współdzielonej ontologii do parsowania `content`. Uzgadnianie ontologii to wieloletni proces standaryzacyjny. Sieć po prostu użyła HTTP + JSON.
- **Formalna semantyka, której nikt nie używał.** SL (Semantic Language) dawał rygorystyczne warunki prawdziwości, ale większość systemów produkcyjnych używała treści dowolnej formy i ignorowała formalizm.
- **Zablokowanie narzędziowe.** JADE był tylko dla Javy; JACK był komercyjny. Wielojęzyczne zespoły omijały oba.
- **Internet wygrał stos.** REST, potem JSON-RPC, potem gRPC zastąpiły transport ACL.

### Odrodzenie LLM to FIPA-lite

Porównaj `request` z FIPA do `tools/call` z MCP:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

Ta sama koperta, inna składnia. Obie niosą: kto, kogo, intencję, ładunek, identyfikator korelacji. Żadna nie jest rewolucją względem drugiej — to różne kompromisy w tym samym projekcie.

Przegląd z 2025 roku autorstwa Liu i in. („A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP", arXiv:2505.02279) czyni to pochodne jawnym: MCP odpowiada aktom mowy użycia narzędzi, A2A aktom mowy agent-równorzędny, ACP aktom mowy śladu audytu, ANP rozszerzeniom zdecentralizowanej tożsamości. Nowe specyfikacje są potomkami ACL ze składnią JSON i luźniejszą semantyką.

### Kompromis, wyrażony wprost

**Co FIPA dawał, a nowoczesne specyfikacje pomijają:**

- Formalną semantykę — możesz udowodnić, że `inform` oznacza, że nadawca wierzy w treść.
- Kanoniczny katalog performatyw — nie musisz na nowo argumentować „czy powinniśmy mieć `cancel`?".
- Dziesięciolecia wzorców protokołów interakcji — contract-net, subscribe-notify, propose-accept — ze znanymi właściwościami poprawności.

**Co nowoczesne specyfikacje dają, a FIPA nie dawał:**

- Natywnie JSON-owe ładunki kompatybilne z każdym nowoczesnym narzędziem.
- Treść w języku naturalnym, którą LLMy mogą interpretować bez ręcznie kodowanej ontologii.
- Transport oparty na stosie webowym (HTTP, SSE, WebSocket).
- Odkrywanie możliwości poprzez samoopisujące się dokumenty (MCP `listTools`, A2A Agent Card).

Luźniejsza semantyka intencji dla łatwiejszej implementacji. To jest właśnie ten kompromis.

### Protokoły interakcji warte przeniesienia

FIPA dostarczał ~15 protokołów interakcji. Trzy są warte przeniesienia do systemów wieloagentowych opartych na LLM:

1. **Contract Net Protocol (CNP).** Menedżer wydaje `cfp` (call for proposals); oferenci odpowiadają `propose`; menedżer akceptuje/odrzuca. To kanoniczny wzorzec rynku zadań (Faza 16 · 16 Negocjacje).
2. **Subscribe/Notify.** Subskrybent wysyła `subscribe`; wydawca wysyła `inform`, gdy temat się zmieni. To każda szyna zdarzeń w 2026 roku.
3. **Request-When.** „Zrób X, gdy zajdzie warunek Y." Działanie opóźnione z warunkami wstępnymi. Odpowiednik w 2026 to odroczone zadania w trwałych silnikach przepływu pracy (Faza 16 · 22 Skalowanie Produkcyjne).

Każdy mapuje się czysto na nowoczesne kolejki komunikatów, HTTP + polling lub strumieniowanie SSE.

### Co się psuje, gdy pomijasz ontologię

Bez współdzielonej ontologii agenci wyciągają znaczenie z treści w języku naturalnym. Udokumentowanym w 2026 trybem awarii jest **dryf semantyczny**: dwóch agentów używa tego samego słowa („klient") dla subtelnie różnych koncepcji, agent odbiorcy działa na błędnej interpretacji, żaden walidator schematu tego nie łapie. Wymóg ontologii FIPA odrzuciłby wiadomość na etapie parsowania.

Mitygacje bez pełnej ontologii:

- JSON Schema na `content` — odrzuca błędy strukturalne na przewodzie.
- Typowane artefakty (A2A) — odrzuca niewłaściwą modalność.
- Jawna performatywa w kopercie — czyni intencję jednoznaczną nawet gdy treść jest w języku naturalnym.

### Specyfikacje 2026, odwzorowane na dziedzictwo aktów mowy

| Nowoczesna specyfikacja | Odpowiednik FIPA | Co zachowuje | Co pomija |
|---|---|---|---|
| MCP `tools/call` | `request` | jawna intencja, identyfikator korelacji | formalną semantykę, ontologię |
| MCP `resources/read` | `query-ref` | jawna intencja, identyfikator korelacji | formalną semantykę |
| Cykl życia zadań A2A | contract-net + request-when | asynchroniczny cykl życia, przejścia stanów | formalne gwarancje kompletności |
| Zdarzenia strumieniowe A2A | subscribe/notify | asynchroniczne push | subskrypcja typowanego predykatu |
| Współdzielony kontekst CA-MCP | blackboard (Hayes-Roth 1985) | wielozapisywalna pamięć współdzielona | model spójności logicznej |
| NLIP | treść w języku naturalnym | natywność LLM | schemat |

Czytając tabelę od góry do dołu, wzór jest następujący: zachowaj strukturalny prymityw, porzuć formalizm, pozwól LLM-om zamazywać dwuznaczność.

## Build It

`code/main.py` implementuje czysto-stdlibowy translator FIPA-ACL. Koduje i dekoduje kanoniczną kopertę ACL i pokazuje, jak każdy kształt wiadomości MCP/A2A sprowadza się do tych samych siedmiu pól. Demo:

- Koduje pięć wiadomości w stylu MCP i A2A jako FIPA-ACL.
- Dekoduje FIPA-ACL z powrotem do nowoczesnego odpowiednika.
- Przeprowadza zabawową negocjację Contract Net między jednym menedżerem a trzema oferentami za pomocą `cfp`, `propose`, `accept-proposal`, `reject-proposal`.

Uruchom:

```
python3 code/main.py
```

Wynik to ślad obok siebie pokazujący każdą nowoczesną wiadomość zarówno w jej formie JSON z 2026 roku, jak i w formie FIPA-ACL, a następnie podróż w obie strony oferty contract-net. Te same prymitywy protokołu przetrwają podróż w obie strony; różni się tylko składnia.

## Use It

`outputs/skill-fipa-mapper.md` to umiejętność, która czyta dowolną specyfikację protokołu agentowego i tworzy mapowanie na FIPA-ACL. Użyj jej przed przyjęciem nowego protokołu, aby odpowiedzieć: „Czy to jest naprawdę nowe, czy to `inform` ze składnią JSON?"

## Ship It

Nie przywracaj FIPA-ACL. Przywróć jego listę kontrolną:

- Jaki jest prymityw intencji (performatywa) każdej wiadomości?
- Czy istnieje identyfikator korelacji dla żądanie-odpowiedź i anulowania?
- Czy istnieje jawny język treści (JSON-RPC, czysty tekst, strukturalny typowany artefakt)?
- Czy protokoły interakcji są pierwszorzędne, czy implementujesz contract-net od zera?
- Co się dzieje, gdy dwóch agentów nie zgadza się co do znaczenia treści (dryf semantyczny)?

Udokumentuj te pięć pytań dla każdego nowego protokołu, zanim wdrożysz go do produkcji.

## Exercises

1. Uruchom `code/main.py`. Zaobserwuj kodowanie w obie strony. Zidentyfikuj, która performatywa FIPA odpowiada `tools/call`, `resources/read` i tworzeniu zadania A2A.
2. Rozszerz demo contract-net o performatywę `cancel`, która pozwala menedżerowi wycofać zadanie w trakcie licytacji. Jaki przypadek awarii rozwiązuje `cancel`, którego same ponowienia nie rozwiązują?
3. Przeczytaj FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) sekcje 4.1–4.3. Wybierz jedną performatywę nieomówioną w tej lekcji i opisz jej nowoczesny odpowiednik JSON-RPC.
4. Przeczytaj Liu i in., arXiv:2505.02279. Dla każdego z MCP, A2A, ACP, ANP, wypisz rodziny performatyw FIPA, które zachowują i które pomijają.
5. Zaprojektuj minimalny JSON-Schema dla pola `content` performatywy `request` we własnym systemie. Co daje ci ten schemat, czego czysty język naturalny nie daje, i ile to kosztuje?

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Akt mowy | „Wypowiedź, która coś robi" | Austin/Searle: wypowiedzi jako działania. Teoretyczny rodzic ACL. |
| FIPA | „Ta stara rzecz XML" | IEEE Foundation for Intelligent Physical Agents. Standaryzował ACL w 2000. |
| ACL | „Agent Communication Language" | Format koperty FIPA: performatywa + treść + metadane. |
| Performative | „Czasownik" | Klasa intencji wiadomości: `inform`, `request`, `propose`, `cfp` itd. |
| KQML | „Poprzednik FIPA" | Knowledge Query and Manipulation Language (1993). Prostszy, węższy. |
| Ontologia | „Współdzielone słownictwo" | Formalna definicja koncepcji, o których mówi język treści. |
| SL0 / SL1 | „Języki treści FIPA" | Semantic Language poziomy 0 i 1 — rodzina formalnych języków treści. |
| Contract Net | „Rynek zadań" | Menedżer wydaje cfp; oferenci proponują; menedżer akceptuje. Kanoniczny protokół interakcji. |
| Protokół interakcji | „Wzór wiadomości" | Sekwencja performatyw ze znaną poprawnością: request-when, subscribe-notify itd. |

## Further Reading

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — kanoniczny przegląd z 2025 łączący nowoczesne specyfikacje z dziedzictwem FIPA
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — ratyfikowany format koperty z 2000
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — pełny katalog performatyw
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — nowoczesny odpowiednik `request`/`query-ref` dla użycia narzędzi
- [A2A specification](https://a2a-protocol.org/latest/specification/) — nowoczesny odpowiednik contract-net i subscribe-notify dla agent-równorzędny