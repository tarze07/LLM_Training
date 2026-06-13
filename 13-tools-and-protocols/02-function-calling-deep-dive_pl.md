# Function Calling — Szczegółowe Porównanie OpenAI, Anthropic, Gemini

> Trzej graniczni dostawcy zbiegli się do tego samego cyklu wywołania narzędzia w 2024 roku, a potem rozeszli się we wszystkim innym. OpenAI używa `tools` i `tool_calls`. Anthropic używa bloków `tool_use` i `tool_result`. Gemini używa `functionDeclarations` i korelacji za pomocą unikalnych identyfikatorów. Ta lekcja porównuje trzech dostawców obok siebie, aby kod działający u jednego dostawcy nie pękł przy przenoszeniu.

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 minutes

## Learning Objectives

- Wymień trzy różnice kształtu między ładunkami function-calling w OpenAI, Anthropic i Gemini (deklaracja, wywołanie, wynik).
- Przetłumacz jedną deklarację narzędzia na wszystkie trzy formaty dostawców i przewidź, gdzie będą się różnić ograniczenia trybu ścisłego.
- Użyj `tool_choice` u każdego dostawcy, aby wymusić, zabronić lub pozwolić na automatyczny wybór wywołań narzędzi.
- Poznaj ograniczenia twarde na dostawcę (liczba narzędzi, głębokość schematu, długość argumentów) i sygnatury błędów, które każdy emituje po przekroczeniu limitów.

## The Problem

Kształt żądania function-calling różni się w zależności od dostawcy. Trzy konkretne przykłady z produkcyjnych stosów z 2026 roku:

**OpenAI Chat Completions / Responses API.** Przekazujesz `tools: [{type: "function", function: {name, description, parameters, strict}}]`. Odpowiedź modelu zawiera `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`, gdzie `arguments` to ciąg JSON, który musisz sparsować. Tryb ścisły (`strict: true`) egzekwuje zgodność schematu przez ograniczone dekodowanie.

**Anthropic Messages API.** Przekazujesz `tools: [{name, description, input_schema}]`. Odpowiedź wraca jako `content: [{type: "text"}, {type: "tool_use", id, name, input}]`. `input` jest już sparsowany (obiekt, nie ciąg). Odpowiadasz nową wiadomością `user` zawierającą blok `{type: "tool_result", tool_use_id, content}`.

**Google Gemini API.** Przekazujesz `tools: [{functionDeclarations: [{name, description, parameters}]}]` (zagnieżdżone pod `functionDeclarations`). Odpowiedź przychodzi jako `candidates[0].content.parts: [{functionCall: {name, args, id}}]`, gdzie `id` jest unikalne w Gemini 3 i nowszych do korelacji równoległych wywołań. Odpowiadasz za pomocą `{functionResponse: {name, id, response}}`.

Ten sam cykl. Różne nazwy pól, różne zagnieżdżenia, różne konwencje ciąg-vs-obiekt, różne mechanizmy korelacji. Zespół, który napisze agenta pogodowego na OpenAI, płaci dwa dni portowania do Anthropic i kolejny dzień do Gemini tylko za infrastrukturę.

Ta lekcja buduje translator, który ujednolica trzy formaty w jedną kanoniczną deklarację narzędzia i kieruje na krawędzi. Faza 13 · 17 uogólnia ten sam wzorzec w bramkę LLM.

## The Concept

### Wspólna struktura

Każdy dostawca potrzebuje pięciu rzeczy:

1. **Lista narzędzi.** Nazwa, opis i schemat wejściowy na narzędzie.
2. **Wybór narzędzia.** Wymuś konkretne narzędzie, zabroń narzędzi lub pozwól modelowi zdecydować.
3. **Emisja wywołania.** Strukturalne wyjście nazywające narzędzie i argumenty.
4. **Identyfikator wywołania.** Koreluj odpowiedź z właściwym wywołaniem (ważne dla równoległości).
5. **Wstrzyknięcie wyniku.** Wiadomość lub blok, który łączy wynik z wywołaniem.

### Różnice kształtów, pole po polu

| Aspekt | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Otoczka deklaracji | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Pole schematu | `parameters` | `input_schema` | `parameters` |
| Kontener odpowiedzi | `tool_calls[]` na wiadomości asystenta | `content[]` typu `tool_use` | `parts[]` typu `functionCall` |
| Typ argumentów | JSON w postaci ciągu | sparsowany obiekt | sparsowany obiekt |
| Format Id | `call_...` (generuje OpenAI) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| Blok wyniku | rola `tool`, `tool_call_id` | `user` z `tool_result`, `tool_use_id` | `functionResponse` z pasującym `id` |
| Wymuś narzędzie | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Zabroń narzędzi | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Ścisły schemat | `strict: true` | schemat-jest-schematem (zawsze egzekwowany) | `responseSchema` na poziomie żądania |

### Limity, które faktycznie osiągniesz

- **OpenAI.** 128 narzędzi na żądanie. Głębokość schematu 5. Ciąg argumentów <= 8192 bajtów. Tryb ścisły wymaga braku `$ref`, braku `oneOf`/`anyOf`/`allOf` z nakładaniem się, każde pole wymienione w `required`.
- **Anthropic.** 64 narzędzia na żądanie. Głębokość schematu praktycznie nieograniczona, ale praktyczny limit 10. Brak flagi trybu ścisłego; schemat jest kontraktem i model ma tendencję do przestrzegania.
- **Gemini.** 64 funkcje na żądanie. Typy schematu są podzbiorem OpenAPI 3.0 (niewielkie odejście od JSON Schema 2020-12). Równoległe wywołania z unikalnym id od Gemini 3.

### Zachowanie `tool_choice`

Trzy tryby, które wszyscy obsługują, nazwane inaczej.

- **Auto.** Model wybiera narzędzie lub tekst. Domyślny.
- **Required / Any.** Model musi wywołać co najmniej jedno narzędzie.
- **None.** Model nie może wywoływać narzędzi.

Plus jeden tryb unikalny dla każdego dostawcy:

- **OpenAI.** Wymuś konkretne narzędzie po nazwie.
- **Anthropic.** Wymuś konkretne narzędzie po nazwie; flaga `disable_parallel_tool_use` oddziela pojedyncze od wielu.
- **Gemini.** `mode: "VALIDATED"` kieruje każdą odpowiedź przez walidator schematu niezależnie od intencji modelu.

### Równoległe wywołania

`parallel_tool_calls: true` OpenAI (domyślnie) emituje wiele wywołań w jednej wiadomości asystenta. Uruchamiasz je wszystkie i odpowiadasz zbiorczą wiadomością z rolą tool zawierającą jeden wpis na `tool_call_id`. Anthropic historycznie robił pojedyncze wywołanie; `disable_parallel_tool_use: false` (domyślnie od Claude 3.5) umożliwia wiele. Gemini 2 pozwalał na równoległe wywołania, ale nie dawał stabilnych identyfikatorów; Gemini 3 dodaje UUID, aby nieuporządkowane odpowiedzi korelowały czysto.

### Strumieniowanie

Wszyscy trzej obsługują strumieniowane wywołania narzędzi. Format na kablu różni się:

- **OpenAI.** Chunks delta `tool_calls[i].function.arguments` przychodzą przyrostowo. Akumulujesz, aż do `finish_reason: "tool_calls"`.
- **Anthropic.** Zdarzenia block-start / block-delta / block-stop. `input_json_delta` chunks niosą częściowe argumenty.
- **Gemini.** `streamFunctionCallArguments` (nowe w Gemini 3) emituje chunks z `functionCallId`, więc wiele równoległych wywołań może się przeplatać.

Faza 13 · 03 zagłębia się w równoległe + strumieniowe składanie. Ta lekcja skupia się na deklaracji i kształtach pojedynczego wywołania.

### Błędy i naprawa

Błędy nieprawidłowych argumentów też wyglądają inaczej.

- **OpenAI (nierygorystyczne).** Model zwraca `arguments: "{bad json}"`, Twój JSON parse pada, wstrzykujesz komunikat błędu i wywołujesz ponownie.
- **OpenAI (tryb ścisły).** Walidacja odbywa się podczas dekodowania; nieprawidłowy JSON jest niemożliwy, ale może pojawić się `refusal`.
- **Anthropic.** `input` może zawierać nieoczekiwane pola; schemat jest doradczy. Waliduj po stronie serwera.
- **Gemini.** Dziwactwo OpenAPI 3.0: `enum` na polach obiektów jest cicho ignorowane; waliduj sam.

### Wzorzec translatora

Kanoniczna deklaracja narzędzia w Twoim kodzie wygląda tak (wybierasz kształt):

```python
Tool(
    name="get_weather",
    description="Użyj gdy ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

Trzy małe funkcje tłumaczą to na trzy kształty dostawców. Harness w `code/main.py` robi dokładnie to, a następnie wykonuje round-trip fałszywego wywołania narzędzia przez kształt odpowiedzi każdego dostawcy. Bez sieci — ta lekcja uczy kształtów, nie HTTP.

Zespoły produkcyjne opakowują ten translator w `AbstractToolset` (Pydantic AI), `UniversalToolNode` (LangGraph) lub `BaseTool` (LlamaIndex). Faza 13 · 17 dostarcza bramkę, która udostępnia API w kształcie OpenAI przed dowolnym z trzech.

## Use It

`code/main.py` definiuje jedną kanoniczną dataclass `Tool` i trzy translatory, które emitują JSON deklaracji OpenAI, Anthropic i Gemini. Następnie parsuje ręcznie wykonaną odpowiedź dostawcy każdego kształtu do tego samego kanonicznego obiektu wywołania, demonstrując, że semantyka jest identyczna pod skórą. Uruchom go i porównaj trzy deklaracje obok siebie.

Na co zwrócić uwagę:

- Trzy bloki deklaracji różnią się tylko otoczką i nazwami pól.
- Trzy bloki odpowiedzi różnią się tym, gdzie mieszka wywołanie (najwyższego poziomu `tool_calls`, blok `content[]`, wpis `parts[]`).
- Jedna funkcja `canonical_call()` wyodrębnia `{id, name, args}` ze wszystkich trzech kształtów odpowiedzi.

## Ship It

Ta lekcja produkuje `outputs/skill-provider-portability-audit.md`. Mając integrację function-calling z jednym dostawcą, umiejętność produkuje audyt przenośności: na jakich limitach dostawcy polega, które pola wymagają zmiany nazwy i co się psuje przy przenoszeniu do każdego innego dostawcy.

## Exercises

1. Uruchom `code/main.py` i zweryfikuj, że trzy deklaracje JSON dostawców serializują ten sam bazowy obiekt `Tool`. Zmodyfikuj kanoniczne narzędzie, aby dodać parametr enum i potwierdź, że tylko translator Gemini musi obsługiwać dziwactwo OpenAPI.

2. Dodaj parser `ListToolsResponse` dla każdego dostawcy, który wyodrębnia listę narzędzi zwróconą przez model po wywołaniu `list_tools` lub discovery. OpenAI nie ma tego natywnie; zanotuj tę asymetrię.

3. Zaimplementuj konwersję `tool_choice`: odwzoruj kanoniczny `ToolChoice(mode="force", tool_name="x")` na wszystkie trzy kształty dostawców. Następnie odwzoruj `mode="any"` i `mode="none"`. Sprawdź tabelę różnic w lekcji.

4. Wybierz jednego z trzech dostawców i przeczytaj jego przewodnik function-calling od deski do deski. Znajdź jedno pole w jego specyfikacji schematu, którego nie obsługują pozostali dwaj. Kandydaci: OpenAI `strict`, Anthropic `disable_parallel_tool_use`, Gemini `function_calling_config.allowed_function_names`.

5. Napisz wektor testowy: wywołanie narzędzia, którego argumenty naruszają zadeklarowany schemat. Przepuść je przez walidator każdego dostawcy (stdlib z Lekcji 01 wystarczy jako proxy) i zarejestruj, które błędy się uruchamiają. Udokumentuj, którego dostawcy użyłbyś w produkcji dla ścisłości.

## Key Terms

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Function calling | "Użycie narzędzia" | API na poziomie dostawcy do strukturalnej emisji wywołania narzędzia |
| Deklaracja narzędzia (Tool declaration) | "Spec narzędzia" | Nazwa + opis + ładunek wejściowy JSON Schema |
| `tool_choice` | "Wymuś / zabroń" | Tryby auto / required / none / konkretna-nazwa |
| Tryb ścisły (Strict mode) | "Egzekwowanie schematu" | Flaga OpenAI, która ogranicza dekodowanie do zgodności ze schematem |
| Blok `tool_use` | "Kształt wywołania Anthropic" | Wbudowany blok treści z id, name, input |
| Część `functionCall` | "Kształt wywołania Gemini" | Wpis `parts[]` zawierający name, args i id |
| Argumenty-jako-ciąg | "JSON w ciągu" | OpenAI zwraca args jako ciąg JSON, a nie obiekt |
| Równoległe wywołania narzędzi (Parallel tool calls) | "Fan-out w jednej turze" | Wiele wywołań narzędzi w jednej wiadomości asystenta |
| Odmowa (Refusal) | "Model odmawia" | Blok odmowy dostępny tylko w trybie ścisłym zamiast wywołania |
| Podzbiór OpenAPI 3.0 | "Dziwactwo schematu Gemini" | Gemini używa dialektu podobnego do JSON-Schema z drobnymi różnicami |

## Further Reading

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — kanoniczna referencja obejmująca tryb ścisły i równoległe wywołania
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — semantyka bloków `tool_use` i `tool_result`
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — równoległe wywołania, unikalne identyfikatory i podzbiór OpenAPI
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — korporacyjna powierzchnia Gemini
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — szczegóły egzekwowania schematu w trybie ścisłym
