# Strukturalne Wyjście — JSON Schema, Pydantic, Zod, Ograniczone Dekodowanie

> "Poproś model ładnie o zwrócenie JSON" zawodzi 5 do 15 procent przypadków, nawet na granicznych modelach. Strukturalne wyjścia zamykają tę lukę za pomocą ograniczonego dekodowania: model jest dosłownie uniemożliwiony wyemitowanie tokena, który naruszyłby schemat. Tryb ścisły OpenAI, narzędzia z typowaniem schematu w Anthropic, `responseSchema` Gemini, `output_type` w Pydantic AI i `.parse` w Zod to pięć form tej samej idei. Ta lekcja buduje walidator schematu i kontrakt trybu ścisłego, których uczący się użyją w każdym produkcyjnym potoku ekstrakcji.

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## Learning Objectives

- Napisz JSON Schema 2020-12 dla celu ekstrakcji, używając odpowiednich ograniczeń (enum, min/max, required, pattern).
- Wyjaśnij, dlaczego tryb ścisły i ograniczone dekodowanie dają inne gwarancje niż "waliduj po generacji".
- Odróżnij trzy tryby awarii: błąd parsowania, naruszenie schematu, odmowa modelu.
- Dostarcz potok ekstrakcji z typowaną naprawą i typowaną obsługą odmowy.

## The Problem

Agent czytający e-mail z zamówieniem zakupu musi przekształcić wolny tekst w `{customer, line_items, total_usd}`. Trzy podejścia.

**Podejście pierwsze: prompt na JSON.** "Odpowiedz w JSON z polami customer, line_items, total_usd." Działa 85 do 95 procent przypadków na granicznych modelach. Zawodzi na sześć sposobów: brakujący nawias klamrowy, końcowy przecinek, złe typy, zhallucynowane pola, obcięcie na limicie tokenów, wyciekająca proza jak "Oto Twój JSON:".

**Podejście drugie: waliduj po generacji.** Generuj swobodnie, parsuj, waliduj względem schematu, ponawiaj przy błędzie. Niezawodne, ale kosztowne — płacisz za każde ponowienie, a błędy obcięcia kosztują jedną dodatkową turę na wystąpienie.

**Podejście trzecie: ograniczone dekodowanie.** Dostawca egzekwuje schemat w czasie dekodowania. Nieprawidłowe tokeny są maskowane z rozkładu próbkowania. Wynik jest gwarantowany do sparsowania i gwarantowany do walidacji. Awaria sprowadza się do jednego trybu: odmowa (model decyduje, że wejście nie pasuje do schematu).

Każdy graniczny dostawca z 2026 roku oferuje jakąś formę podejścia trzeciego.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}` plus `refusal` w odpowiedzi, jeśli model odmawia.
- **Anthropic.** Egzekwowanie schematu na wejściach `tool_use`; `stop_reason: "refusal"` nie istnieje, ale `end_turn` bez wywołania narzędzia jest sygnałem.
- **Gemini.** `responseSchema` na poziomie żądania; w 2026 Gemini dostarcza ograniczenia gramatyki na poziomie tokenów dla wybranych typów.
- **Pydantic AI.** `output_type=InvoiceModel` emituje strukturalny `RunResult` typowany do `InvoiceModel`.
- **Zod (TypeScript).** Parser czasu wykonania, który waliduje dane wyjściowe dostawcy względem schematu Zod; współpracuje z `beta.chat.completions.parse` OpenAI.

Wspólny wątek: zadeklaruj schemat raz, egzekwuj go od końca do końca.

## The Concept

### JSON Schema 2020-12 — lingua franca

Każdy dostawca akceptuje JSON Schema 2020-12. Konstrukty, których używasz najczęściej:

- `type`: jeden z `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
- `properties`: mapa nazwy pola do podschematu.
- `required`: lista nazw pól, które muszą wystąpić.
- `enum`: zamknięty zbiór dozwolonych wartości.
- `minimum` / `maximum` (liczby), `minLength` / `maxLength` / `pattern` (ciągi).
- `items`: podschemat stosowany do każdego elementu tablicy.
- `additionalProperties`: `false` zabrania dodatkowych pól (domyślnie różni się w zależności od trybu).

Tryb ścisły OpenAI dodaje trzy wymagania: każde pole musi być wymienione w `required`, `additionalProperties: false` wszędzie i brak nierozwiązanych `$ref`. Jeśli je złamiesz, API zwraca 400 w czasie żądania.

### Pydantic, wiązanie Pythona

Pydantic v2 generuje JSON Schema z modeli w kształcie dataclass za pomocą `model_json_schema()`. Pydantic AI opakowuje to, więc piszesz:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

a framework agenta tłumaczy schemat na tryb ścisły OpenAI, `input_schema` Anthropic lub `responseSchema` Gemini na krawędzi. Wynik modelu wraca jako typowana instancja `Invoice`. Błędy walidacji podnoszą `ValidationError` z typowanymi ścieżkami błędów.

### Zod, wiązanie TypeScript

Zod (`z.object({customer: z.string(), ...})`) jest odpowiednikiem w TS. Node SDK OpenAI udostępnia `zodResponseFormat(Invoice)`, który tłumaczy na ładunek JSON Schema API.

### Odmowy

Tryb ścisły nie może zmusić modelu do odpowiedzi. Jeśli wejście nie pasuje do schematu ("e-mail był wierszem, a nie fakturą"), model emituje pole `refusal` zawierające powód. Twój kod musi obsługiwać to jako pierwszoklasowy wynik, a nie błąd. Odmowa jest również użyteczna jako sygnał bezpieczeństwa: model poproszony o wyodrębnienie numeru karty kredytowej z chronionej treści zwraca odmowę z dołączonym powodem bezpieczeństwa.

### Ograniczone dekodowanie w otwartej przestrzeni

Implementacje open-weights używają trzech technik.

1. **Dekodowanie oparte na gramatyce** (`outlines`, `guidance`, `lm-format-enforcer`): zbuduj deterministyczny automat skończony ze schematu; na każdym kroku maskuj logity tokenów, które naruszyłyby FSM.
2. **Maskowanie logitów za pomocą parsera JSON**: uruchom strumieniowy parser JSON w synchronizacji z modelem; na każdym kroku oblicz zbiór prawidłowych następnych tokenów.
3. **Spekulacyjne dekodowanie z weryfikatorem**: tani model proponujący tokeny, weryfikator egzekwuje schemat.

Komercyjni dostawcy wybierają jedną z tych opcji za kulisami. Stan sztuki w 2026 jest szybszy niż zwykła generacja dla krótkich strukturalnych wyjść i mniej więcej tak samo szybki dla długich.

### Trzy tryby awarii

1. **Błąd parsowania.** Wynik nie jest prawidłowym JSON. Nie może wystąpić w trybie ścisłym. Może nadal wystąpić u dostawców bez trybu ścisłego.
2. **Naruszenie schematu.** Wynik parsuje się, ale narusza schemat. Nie może wystąpić w trybie ścisłym. Częste poza nim.
3. **Odmowa.** Model odmawia. Musi być obsługiwana jako typowany wynik.

### Strategia ponawiania

Gdy jesteś poza trybem ścisłym (użycie narzędzi Anthropic, nierygorystyczne OpenAI, starsze Gemini), wzorzec odzyskiwania to:

```
generuj -> parsuj -> waliduj -> jeśli błąd, wstrzyknij błąd i ponów, max 3x
```

Jedno ponowienie zwykle wystarcza. Trzy ponowienia łapią flaki słabego modelu. Powyżej trzech to oznaka złego schematu: model nie może go spełnić dla niektórych wejść, a prompt lub schemat wymaga naprawy.

### Wsparcie dla małych modeli

Ograniczone dekodowanie działa na małych modelach. Otwarty model o 3B parametrach z egzekwowaniem gramatyki przewyższa model o 70B parametrach z surowym promptowaniem w zadaniach strukturalnych. To główny powód, dla którego strukturalne wyjścia mają znaczenie w produkcji: oddziela niezawodność od rozmiaru modelu.

## Use It

`code/main.py` zawiera minimalny walidator JSON Schema 2020-12 w stdlib (typy, required, enum, min/max, pattern, items, additionalProperties). Opakowuje schemat `Invoice` i przepuszcza fałszywe wyjście LLM przez walidator, demonstrując ścieżki błędu parsowania, naruszenia schematu i odmowy. Zamień fałszywe wyjście na odpowiedź dowolnego prawdziwego dostawcy w produkcji.

Na co zwrócić uwagę:

- Walidator zwraca typowaną listę `[ValidationError]` ze ścieżką i komunikatem. To jest kształt, który chcesz przekazać do prompta ponawiania.
- Gałąź odmowy NIE ponawia. Loguje i zwraca typowaną odmowę. Faza 14 · 09 używa odmów jako sygnału bezpieczeństwa.
- Sprawdzenie `additionalProperties: false` uruchamia się na adversarialnym wejściu testowym, pokazując, dlaczego tryb ścisły zamyka drzwi na zhallucynowane pola.

## Ship It

Ta lekcja produkuje `outputs/skill-structured-output-designer.md`. Mając cel ekstrakcji z wolnego tekstu (faktury, zgłoszenia pomocy, CV itp.), umiejętność produkuje JSON Schema 2020-12 zgodny z trybem ścisłym i model Pydantic, który go odzwierciedla, z wstawionymi typowanymi szkieletami obsługi odmowy i ponawiania.

## Exercises

1. Uruchom `code/main.py`. Dodaj czwarty przypadek testowy, którego `total_usd` jest liczbą ujemną. Potwierdź, że walidator odrzuca go z ograniczeniem ścieżki `minimum`.

2. Rozszerz walidator, aby obsługiwał `oneOf` z dyskryminatorem. Typowy przypadek: `line_item` jest albo produktem albo usługą, oznaczony przez `kind`. Tryb ścisły ma subtelne reguły tutaj; sprawdź przewodnik strukturalnych wyjść OpenAI.

3. Napisz ten sam schemat Invoice jako Pydantic BaseModel i porównaj dane wyjściowe `model_json_schema()` z ręcznie zrobionym schematem. Zidentyfikuj jedno pole, które Pydantic ustawia domyślnie, a które ręczna wersja pomija.

4. Zmierz wskaźniki odmów. Skonstruuj dziesięć wejść, które nie powinny być ekstrahowalne (tekst piosenki, dowód matematyczny, pusty e-mail) i przepuść je przez prawdziwego dostawcę w trybie ścisłym. Policz odmowy vs zhallucynowane wyniki. To jest Twoja prawda podstawowa dla ponowień świadomych odmowy.

5. Przeczytaj przewodnik strukturalnych wyjść OpenAI od deski do deski. Zidentyfikuj jedną konstrukcję, którą wyraźnie zabrania w trybie ścisłym, a którą zwykły JSON Schema pozwala. Następnie zaprojektuj schemat, który używa zabronionej konstrukcji nieistotnie i przepisz go, aby był zgodny z trybem ścisłym.

## Key Terms

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "Specyfikacja schematu" | Dialekt schematu IETF-draft, którym mówi każdy nowoczesny dostawca |
| Tryb ścisły (Strict mode) | "Gwarantowany schemat" | Flaga OpenAI, która egzekwuje schemat przez ograniczone dekodowanie |
| Ograniczone dekodowanie (Constrained decoding) | "Maskowanie logitów" | Egzekwowanie w czasie dekodowania, które maskuje nieprawidłowe następne tokeny |
| Odmowa (Refusal) | "Model odmawia" | Typowany wynik, gdy wejście nie pasuje do schematu |
| Błąd parsowania (Parse error) | "Nieprawidłowy JSON" | Wynik nie sparsował się jako JSON; niemożliwe w trybie ścisłym |
| Naruszenie schematu (Schema violation) | "Zły kształt" | Sparsował się, ale naruszył typy / required / enum / zakres |
| `additionalProperties: false` | "Żadnych dodatków" | Zabrania nieznanych pól; wymagane w trybie ścisłym OpenAI |
| Pydantic BaseModel | "Typowane wyjście" | Klasa Pythona, która emituje i waliduje JSON Schema |
| Schema Zod | "Typ wyjścia TypeScript" | Schemat czasu wykonania TS do walidacji wyjścia dostawcy |
| Egzekwowanie gramatyki (Grammar enforcement) | "Ograniczone dekodowanie open-weights" | Maskowanie logitów oparte na FSM, jak w outlines / guidance |

## Further Reading

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — tryb ścisły, odmowy i wymagania schematu
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — post premierowy z sierpnia 2024 wyjaśniający gwarancję dekodowania
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — typowane wiązania output_type, które serializują do każdego dostawcy
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — kanoniczna specyfikacja
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — notatki o wdrożeniach korporacyjnych i zastrzeżenia dotyczące trybu ścisłego
