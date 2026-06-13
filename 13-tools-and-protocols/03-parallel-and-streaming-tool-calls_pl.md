# Równoległe Wywołania Narzędzi i Strumieniowanie z Narzędziami

> Trzy niezależne zapytania pogodowe zserializowane to trzy rundy. Uruchom je równolegle, a całkowity czas spada do czasu najwolniejszego pojedynczego wywołania. Każdy graniczny dostawca emituje teraz wiele wywołań narzędzi w jednej turze. Zysk jest realny; infrastruktura jest subtelna. Ta lekcja omawia obie połowy: równoległe rozgałęzianie i składanie strumieniowanych argumentów, z naciskiem na pułapkę korelacji identyfikatorów.

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## Learning Objectives

- Wyjaśnij, dlaczego `parallel_tool_calls: true` istnieje i kiedy go wyłączyć.
- Koreluj strumieniowane fragmenty argumentów z właściwym identyfikatorem wywołania narzędzia podczas równoległego rozgałęziania.
- Składaj częściowe ciągi `arguments` w kompletny JSON bez parsowania na wczesnym etapie.
- Uruchom benchmark pogodowy dla trzech miast, który demonstruje opóźnienie sekwencyjne vs równoległe.

## The Problem

Bez równoległych wywołań, agent odpowiadający "jaka jest pogoda w Bengaluru, Tokio i Zurychu" robi to:

```
user -> LLM
LLM -> wywołaj get_weather(Bengaluru)
host -> uruchom wykonawcę, odpowiedz wynikiem
LLM -> wywołaj get_weather(Tokio)
host -> uruchom wykonawcę, odpowiedz wynikiem
LLM -> wywołaj get_weather(Zurych)
host -> uruchom wykonawcę, odpowiedz wynikiem
LLM -> końcowa odpowiedź tekstowa
```

Trzy rundy LLM, z których każda płaci również opóźnienie wykonawcy. Mniej więcej 4x idealny czas ścienny.

Z równoległymi wywołaniami:

```
user -> LLM
LLM -> wywołaj get_weather(Bengaluru); wywołaj get_weather(Tokio); wywołaj get_weather(Zurych)
host -> uruchom wszystkie trzy wykonawców współbieżnie, odpowiedz trzema wynikami
LLM -> końcowa odpowiedź tekstowa
```

Jedna runda LLM. Czas wykonawcy to maksimum z trzech, a nie suma. Produkcyjne benchmarki na OpenAI, Anthropic i Gemini pokazują 60 do 70 procent redukcji czasu ściennego na obciążeniach rozgałęzionych.

Ceną jest złożoność korelacji. Gdy trzy wywołania kończą się poza kolejnością, Twoje wyniki muszą nieść pasujący `tool_call_id`, aby model mógł je dopasować. Gdy wyniki strumieniują, musisz złożyć częściowe fragmenty argumentów w kompletny JSON przed wykonaniem. Gemini 3 dodał unikalne identyfikatory częściowo po to, by rozwiązać rzeczywisty problem, gdzie dwa równoległe wywołania do tego samego narzędzia były nierozróżnialne.

## The Concept

### Włączanie równoległości

- **OpenAI.** `parallel_tool_calls: true` domyślnie włączone. Ustaw `false`, aby wymusić sekwencyjność.
- **Anthropic.** Równoległość przez `disable_parallel_tool_use: false` (domyślnie w Claude 3.5 i nowszych). Ustaw `true` dla sekwencyjności.
- **Gemini.** Zawsze zdolny do równoległości; `tool_config.function_calling_config.mode = "AUTO"` pozwala modelowi zdecydować.

Wyłącz równoległość, gdy narzędzia mają zależności kolejnościowe (`create_file` potem `write_file`), gdy wynik jednego wywołania informuje wejście innego, lub gdy ogranicznik szybkości nie radzi sobie z rozgałęzieniem.

### Korelacja identyfikatorów

Każde wywołanie wyemitowane przez model ma `id`. Każdy wynik zwrócony przez hosta musi zawierać to samo id. Bez tego wyniki są niejednoznaczne.

- **OpenAI.** `tool_call_id` na każdej wiadomości z rolą narzędzia.
- **Anthropic.** `tool_use_id` na każdym bloku `tool_result`.
- **Gemini.** `id` na każdym `functionResponse` (Gemini 3 i nowsze; Gemini 2 dopasowywał po nazwie, co psuło się dla równoległych wywołań o tej samej nazwie).

### Uruchamianie wywołań współbieżnie

Host uruchamia wykonawcę każdego wywołania na własnym wątku, korutynie lub zdalnym pracowniku. Najprostszy harness używa puli wątków; produkcja używa asyncio z `asyncio.gather` lub strukturalnej współbieżności. Kolejność zakończenia jest nieprzewidywalna — identyfikator jest kluczem.

Jeden częsty błąd: odpowiadaj wynikami w kolejności listy wywołań zamiast w kolejności zakończenia. To zwykle działa, ponieważ model dba tylko o `tool_call_id`, ale jeśli wynik zostanie pominięty lub zduplikowany, wysyłanie poza kolejnością utrudnia debugowanie. Lepiej odpowiadać w kolejności zakończenia z jawnymi identyfikatorami.

### Strumieniowanie wywołań narzędzi

Gdy model strumieniuje, `arguments` przychodzą w kawałkach. Trzy osobne strumienie kawałków dla trzech równoległych wywołań przeplatają się na kablu. Potrzebujesz jednego akumulatora na identyfikator.

Kształt według dostawcy:

- **OpenAI.** Każdy chunk to `choices[0].delta.tool_calls[i].function.arguments` (częściowy ciąg). Chunk niesie `index` (pozycję na liście wywołań). Akumulujesz na indeks, czytasz `id`, gdy pojawia się po raz pierwszy, i parsujesz JSON, gdy `finish_reason = "tool_calls"`.
- **Anthropic.** Zdarzenia strumienia to `message_start`, potem jeden `content_block_start` na blok z typem `tool_use` (zawierający id, name, pusty input). Zdarzenia `content_block_delta` niosą fragmenty `input_json_delta`. `content_block_stop` zamyka każdy blok.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 i nowsze) emituje fragmenty z `functionCallId`, więc wywołania przeplatają się czysto. Przed Gemini 3 strumieniowanie zwracało jedno kompletne wywołanie na raz.

### Częściowy JSON i pułapka wczesnego parsowania

Nie możesz parsować `arguments`, dopóki nie są kompletne. Częściowy JSON, taki jak `{"city": "Beng` nie jest prawidłowy i zgłosi błąd. Właściwą bramką jest sygnał zakończenia wywołania od dostawcy: `finish_reason = "tool_calls"` OpenAI, `content_block_stop` Anthropic lub zdarzenie końca strumienia Gemini. Dopiero wtedy wywołaj `json.loads`. Bardziej niezawodne podejście używa przyrostowego parsera JSON, który emituje zdarzenia w miarę uzupełniania struktury; przewodnik strumieniowania OpenAI zaleca to do UX pokazującego żywy wskaźnik "myśli". Liczenie nawiasów klamrowych jest zawodne jako test kompletności (nawiasy wewnątrz cytowanych ciągów lub escapowanych treści powodują fałszywie pozytywne wyniki) i powinno być używane tylko jako nieformalna heurystyka debugowania.

### Zakończenie poza kolejnością

```
call_A: szybkie API, zwraca pierwsze
call_B: wolne API, zwraca drugie
call_C: średnie API, zwraca trzecie
```

Odpowiedź hosta musi nadal podawać identyfikatory:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

Kolejność w odpowiedzi nie ma znaczenia dla poprawności w OpenAI lub Anthropic. Gemini akceptuje dowolną kolejność, o ile identyfikatory są zgodne.

### Benchmark: sekwencyjny vs równoległy

Harness w `code/main.py` symuluje trzech wykonawców z opóźnieniami 400, 600 i 800 ms. Sekwencyjne uruchomienie zajmuje łącznie 1800 ms. Równoległe uruchomienie zajmuje max(400, 600, 800) = 800 ms. Różnica jest stała, a nie proporcjonalna, więc oszczędności rosną wraz z liczbą narzędzi.

Zastrzeżenie ze świata rzeczywistego: równoległe wywołania obciążają downstreamowe API. 10-krotne rozgałęzienie do usługi z ograniczeniem szybkości nie powiedzie się. Faza 13 · 17 obejmuje backpressure na poziomie bramki; semantyka ponawiania jest planowana w przyszłej fazie.

### Czas ścienny strumieniowanego rozgałęzienia

Jeśli sam model strumieniuje, możesz rozpocząć wykonywanie, gdy tylko argumenty jednego wywołania będą kompletne, zamiast czekać na sfinalizowanie wszystkich wywołań. To optymalizacja, którą dokumentuje OpenAI, ale nie wszystkie SDK udostępniają. Harness w tej lekcji to robi: gdy tylko symulowany strumień dostarczy kompletny obiekt argumentów, host uruchamia to wywołanie.

## Use It

`code/main.py` ma dwie połowy. Pierwsza uruchamia trzy symulowane wywołania pogodowe sekwencyjnie i równolegle przy użyciu `concurrent.futures.ThreadPoolExecutor` i wypisuje czas ścienny. Druga połowa odtwarza fałszywą odpowiedź strumieniową — fragmenty `arguments` dla trzech równoległych wywołań przeplatanych w jednym strumieniu — i składa je na identyfikator za pomocą `StreamAccumulator`. Bez LLM, bez sieci, tylko logika składania.

Na co zwrócić uwagę:

- Timer sekwencyjny osiąga 1.8 sekundy. Timer równoległy osiąga 0.8 sekundy na tych samych fałszywych opóźnieniach.
- Akumulator obsługuje fragmenty przychodzące poza kolejnością, buforując na identyfikator i parsując tylko wtedy, gdy JSON każdego wywołania jest kompletny.
- Wykonawca uruchamia się, gdy tylko argumenty identyfikatora zostaną sfinalizowane, a nie po zakończeniu wszystkich strumieni.

## Ship It

Ta lekcja produkuje `outputs/skill-parallel-call-safety-check.md`. Mając rejestr narzędzi, umiejętność audytuje, które narzędzia są bezpieczne do zrównoleglenia, które mają zależności kolejnościowe, a które przeciążyłyby downstreamowe ograniczniki szybkości — zwracając poprawiony rejestr z flagami `parallel_safe` na narzędzie.

## Exercises

1. Uruchom `code/main.py` i zmień symulowane opóźnienia. Potwierdź, że stosunek równoległy-do-sekwencyjnego wynosi w przybliżeniu `max/sum` (rzeczywiste uruchomienia odbiegają nieco od ideału z powodu planowania wątków, serializacji i narzutu harnessu). Przy jakim rozkładzie opóźnień równoległość przestaje mieć znaczenie?

2. Rozszerz akumulator, aby obsługiwał przypadek "wywołanie zostało anulowane w trakcie strumienia" przez usunięcie jego bufora i wyemitowanie zdarzenia `cancelled`. Który dostawca jawnie to dokumentuje? Sprawdź semantykę `content_block_stop` w Anthropic i zachowanie `finish_reason: "length"` w OpenAI.

3. Zastąp pulę wątków `asyncio.gather`. Porównaj oba. Powinieneś zobaczyć niewielkie korzyści na async z powodu niższego kosztu przełączania kontekstu, ale tylko jeśli wykonawcy wykonują prawdziwe I/O.

4. Wybierz dwa narzędzia, które NIE powinny być zrównoleglone (np. `create_file` a następnie `write_file`). Dodaj graf `ordering_dependency` do rejestru i zablokuj równoległe rozgałęzianie na tym grafie. To minimalny mechanizm dla planowania świadomego zależności, który przyszła faza inżynierii agentów sformalizuje.

5. Przeczytaj sekcję o równoległym function-calling w OpenAI i dokumenty `disable_parallel_tool_use` w Anthropic. Zidentyfikuj jeden typ narzędzia ze świata rzeczywistego, dla którego Anthropic zaleca wyłączenie równoległości. (Podpowiedź: konsekwencjalne mutacje na tym samym zasobie.)

## Key Terms

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|----------------|------------------------|
| Równoległe wywołania narzędzi (Parallel tool calls) | "Rozgałęzienie w jednej turze" | Model emituje wiele wywołań narzędzi w jednej wiadomości asystenta |
| `parallel_tool_calls` | "Flaga OpenAI" | Włącz lub wyłącz emisję wielu wywołań |
| `disable_parallel_tool_use` | "Odwrotność w Anthropic" | Flaga rezygnacji; domyślnie równoległość włączona |
| Identyfikator wywołania narzędzia (Tool call id) | "Uchwyt korelacji" | Identyfikator na wywołanie, który wiadomość wynikowa musi odbić |
| Akumulator (Accumulator) | "Bufor strumienia" | Bufor ciągu na identyfikator dla częściowych fragmentów `arguments` |
| Zakończenie poza kolejnością (Out-of-order completion) | "Najszybszy pierwszy" | Równoległe wywołania kończą się w nieprzewidywalnej kolejności; identyfikatory są spoiwem |
| Graf zależności (Dependency graph) | "Ograniczenia kolejnościowe" | Narzędzia, których wyniki są wejściami innych narzędzi; nie można zrównoleglić |
| Pułapka wczesnego parsowania (Parse-early trap) | "JSON.parse wybuchł" | Próba parsowania niekompletnego ciągu `arguments` |
| `streamFunctionCallArguments` | "Funkcja Gemini 3" | Strumieniowane fragmenty argumentów z unikalnym id na wywołanie |
| Odpowiedź w kolejności zakończenia (Completion-order reply) | "Nie czekaj na wszystkie" | Odpowiadaj wynikami, gdy przychodzą, kluczowane po id |

## Further Reading

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — domyślne zachowanie i flaga rezygnacji
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` i grupowanie wyników
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — skorelowane równoległe wywołania z Gemini 3
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — składanie fragmentów argumentów dla strumieni OpenAI
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` z `input_json_delta`
