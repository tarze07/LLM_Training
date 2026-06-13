# Dlaczego Wieloagentowość?

> Jeden agent uderza w ścianę. Mądrzejszym posunięciem nie jest większy agent — to więcej agentów.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Learning Objectives

- Zidentyfikuj limity pojedynczego agenta (przepełnienie kontekstu, mieszana wiedza specjalistyczna, sekwencyjne wąskie gardło) i wyjaśnij, kiedy podział na wielu agentów jest właściwym posunięciem
- Porównaj wzorce orkiestracji (potok, równoległe rozgałęzienie, nadzorca, hierarchia) i wybierz odpowiedni dla danej struktury zadania
- Zaprojektuj system wieloagentowy z jasno określonymi rolami, współdzielonym stanem i kontraktem komunikacyjnym
- Przeanalizuj kompromisy złożoności wieloagentowej (opóźnienie, koszt, trudność debugowania) względem prostoty pojedynczego agenta

## The Problem

Zbudowałeś pojedynczego agenta w Fazie 14. Działa. Potrafi czytać pliki, uruchamiać polecenia, wywoływać API i analizować wyniki. Potem kierujesz go na prawdziwą bazę kodu: 200 plików, trzy języki, testy zależne od infrastruktury i wymóg badania zewnętrznych API przed pisaniem kodu.

Agent się dusi. Nie dlatego, że LLM jest głupi, ale dlatego, że zadanie przekracza to, co jedna pętla agenta może obsłużyć. Okno kontekstu wypełnia się zawartością plików. Agent zapomina, co przeczytał 40 wywołań narzędzi temu. Próbuje być jednocześnie badaczem, programistą i recenzentem, a wszystkie trzy role wykonuje słabo.

To jest limit pojedynczego agenta. Osiągasz go za każdym razem, gdy zadanie wymaga:

- **Więcej kontekstu, niż mieści się w jednym oknie** — czytanie 50 plików przekracza 200k tokenów
- **Różnej wiedzy specjalistycznej na różnych etapach** — badania wymagają innych promptów niż generowanie kodu
- **Pracy, która może być wykonywana równolegle** — po co czytać trzy pliki sekwencyjnie, skoro można je czytać jednocześnie?

## The Concept

### Limit Pojedynczego Agenta

Pojedynczy agent to jedna pętla, jedno okno kontekstu, jeden prompt systemowy. Wyobraź to sobie:

```
┌─────────────────────────────────────────┐
│            POJEDYNCZY AGENT             │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Okno kontekstu            │  │
│  │                                   │  │
│  │  notatki badawcze                  │  │
│  │  + pliki kodu                     │  │
│  │  + wyniki testów                  │  │
│  │  + opinie z przeglądu            │  │
│  │  + dokumentacja API               │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ PEŁNE ███  │  │
│  │  ██████████████████████         ████ │
│  └───────────────────────────────────┘  │
│                                         │
│  Jeden prompt systemowy próbuje         │
│  objąć badania + kodowanie +            │
│  recenzję + testowanie                  │
│                                         │
│  Rezultat: przeciętny we wszystkim      │
└─────────────────────────────────────────┘
```

Trzy rzeczy się psują:

1. **Nasycenie kontekstu** — wyniki narzędzi kumulują się. Do 30. tury agent zużył 150k tokenów zawartości plików, wyników poleceń i wcześniejszego rozumowania. Kluczowe szczegóły z 5. tury zostają utracone.

2. **Zamieszanie ról** — prompt systemowy mówiący „jesteś badaczem, programistą, recenzentem i testerem" produkuje agenta, który w połowie bada, w połowie koduje i nigdy nie kończy recenzji.

3. **Sekwencyjne wąskie gardło** — agent czyta plik A, potem plik B, potem plik C. Trzy serialne wywołania LLM. Trzy serialne wykonania narzędzi. Żadnej równoległości.

### Rozwiązanie Wieloagentowe

Podziel pracę. Daj każdemu agentowi jedną pracę, jedno okno kontekstu i jeden prompt systemowy dostrojony do tej pracy:

```
┌──────────────────────────────────────────────────────────┐
│                    ORKIESTRATOR                          │
│                                                          │
│  „Zbuduj REST API do zarządzania użytkownikami"          │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │ BADACZ   │ │ PROGRAM. │ │ RECENZENT│ │ TESTER   │  │
│   │          │ │          │ │          │ │          │  │
│   │ Czyta    │ │ Pisze    │ │ Sprawdza │ │ Uruchamia│  │
│   │ dok.,    │ │ kod      │ │ jakość   │ │ testy,   │  │
│   │ znajduje │ │ na podst.│ │ kodu,    │ │ raportuje│  │
│   │ wzorce   │ │ badań    │ │ znajduje │ │ wyniki   │  │
│   │          │ │ + spec   │ │ błędy    │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Scal wyniki                          │
└──────────────────────────────────────────────────────────┘
```

Każdy agent ma:
- Skoncentrowany prompt systemowy („Jesteś recenzentem kodu. Twoim jedynym zadaniem jest znajdowanie błędów.")
- Własne okno kontekstu (niezanieczyszczone pracą innych agentów)
- Jasny kontrakt wejścia/wyjścia (otrzymuje notatki badawcze, zwraca kod)

### Prawdziwe Systemy, Które To Robią

**Subagenci Claude Code** — gdy Claude Code tworzy subagenta za pomocą `Task`, tworzy agenta potomnego z ograniczonym zadaniem. Rodzic zachowuje czysty kontekst. Dziecko wykonuje skoncentrowaną pracę i zwraca podsumowanie.

**Devin** — używa agenta planisty, agenta programisty i agenta przeglądarki. Planista dzieli pracę na kroki. Programista pisze kod. Przeglądarka bada dokumentację. Każdy ma oddzielny kontekst.

**Wieloagentowe zespoły programistyczne (SWE-bench)** — najlepsze systemy na SWE-bench używają badacza czytającego bazę kodu, planisty projektującego poprawkę i programisty ją implementującego. Systemy jednoagentowe osiągają niższe wyniki.

**ChatGPT Deep Research** — tworzy wiele agentów wyszukujących równolegle, każdy eksploruje inny kąt, a następnie syntezuje wyniki.

### Spektrum

Wieloagentowość nie jest binarna. To spektrum:

```
PROSTE ──────────────────────────────────────────── ZŁOŻONE

 Pojedynczy   Sub-          Potok         Zespół       Rój
 Agent       agenci

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │współ- │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ dziel.│
             └───┘          └───┘───┘  │  msg   │    │ stan  │
                                        │  bus   │    └───────┘
 1 pętla     Rodzic +      Etap po     │       │    N równych,
 1 kontekst  zadania       etapie      └───────┘    emerg.
             potomne                   Wyraźne      zachowanie
                                        role
```

**Pojedynczy agent** — jedna pętla, jeden prompt. Dobry do prostych zadań.

**Subagenci** — rodzic tworzy dzieci do skoncentrowanych podzadań. Rodzic utrzymuje plan. Dzieci raportują. To właśnie robi Claude Code.

**Potok** — agenci uruchamiani sekwencyjnie. Wynik agenta A staje się wejściem agenta B. Dobry do etapowych przepływów pracy: badania -> kod -> recenzja -> test.

**Zespół** — agenci działają równolegle ze współdzieloną magistralą komunikatów. Każdy ma rolę. Orkiestrator koordynuje. Dobry, gdy różne umiejętności są potrzebne jednocześnie.

**Rój** — wielu identycznych lub prawie identycznych agentów ze współdzielonym stanem. Bez stałego orkiestratora. Agenci pobierają pracę z kolejki. Dobry do zadań o wysokiej przepustowości równoległej.

### Cztery Wzorce Wieloagentowe

#### Wzorzec 1: Potok

```
Wejście ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Wyjście
           (badania)    (kod)        (recenzja)
```

Każdy agent przekształca dane i przekazuje dalej. Łatwy w analizie. Awaria jednego etapu blokuje resztę.

#### Wzorzec 2: Rozgałęzienie / Scalenie

```
                 ┌──▶ Agent A ──┐
                 │              │
 Wejście ──▶ Podział ├──▶ Agent B ──├──▶ Scal ──▶ Wyjście
                 │              │
                 └──▶ Agent C ──┘
```

Podziel pracę na równoległych agentów, a następnie scal wyniki. Dobry do zadań, które rozkładają się na niezależne podzadania.

#### Wzorzec 3: Orkiestrator-Pracownik

```
                     ┌──────────┐
                     │  Ork.    │
                     └──┬───┬───┘
                   task │   │ task
                  ┌─────┘   └─────┐
                  ▼               ▼
            ┌──────────┐   ┌──────────┐
            │ Pracownik│   │ Pracownik│
            │    A     │   │    B     │
            └──────────┘   └──────────┘
```

Inteligentny orkiestrator decyduje, co robić, deleguje do pracowników i syntezuje wyniki. Orkiestrator sam jest agentem z narzędziami do tworzenia pracowników.

#### Wzorzec 4: Rój Równorzędny

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Współ-   │◄────┘
                │  dzielony │
           ┌───▶│  Stan     │◄────┐
           │    │  /Kolejka │     │
      msg  │    └───────────┘     │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

Brak centralnego orkiestratora. Agenci komunikują się bezpośrednio między sobą. Decyzje wyłaniają się z interakcji. Trudniejsze do debugowania, ale skaluje się do wielu agentów.

### Kiedy NIE stosować Wieloagentowości

Wieloagentowość dodaje złożoności. Każda wiadomość między agentami to potencjalny punkt awarii. Debugowanie przechodzi z „przeczytaj jedną rozmowę" do „śledź wiadomości w pięciu agentach."

**Zostań przy pojedynczym agencie gdy:**
- Zadanie mieści się w jednym oknie kontekstu (poniżej ~100k tokenów danych roboczych)
- Nie potrzebujesz różnych promptów systemowych dla różnych etapów
- Wykonanie sekwencyjne jest wystarczająco szybkie
- Zadanie jest na tyle proste, że dzielenie go dodaje więcej narzutu niż wartości

**Koszt złożoności:**
- Każda granica agenta to krok kompresji stratnej: pełny kontekst agenta A jest podsumowywany w wiadomości dla agenta B
- Logika koordynacji (kto co robi, kiedy, w jakiej kolejności) sama w sobie jest źródłem błędów
- Opóźnienie wzrasta: N agentów oznacza minimum N serialnych wywołań LLM, więcej jeśli muszą ze sobą rozmawiać
- Koszt mnoży się: każdy agent spala tokeny niezależnie

Zasada kciuka: jeśli zadanie wymaga mniej niż 20 wywołań narzędzi i mieści się w 100k tokenów, zostań przy pojedynczym agencie.

```figure
swarm-messages
```

## Build It

### Krok 1: Przeciążony Pojedynczy Agent

Oto pojedynczy agent próbujący robić wszystko. Ma jeden ogromny prompt systemowy i jedno okno kontekstu mieszczące badania, kod i recenzje:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Problemy z tym podejściem:
- Okno kontekstu rośnie z każdym etapem. W kroku recenzji zawiera notatki badawcze ORAZ kod ORAZ wcześniejsze rozumowanie.
- Prompt systemowy jest ogólny. Nie może być dostrojony do każdego etapu.
- Nic nie działa równolegle.

### Krok 2: Agenci Specjaliści

Teraz podziel to. Każdy agent dostaje jedną pracę:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

Każdy specjalista ma skoncentrowany prompt. Każdy dostaje czyste okno kontekstu z tylko potrzebnymi danymi wejściowymi.

### Krok 3: Koordynacja przez Wiadomości

Połącz specjalistów z jawnym przekazywaniem wiadomości:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Każdy agent otrzymuje tylko wiadomości zaadresowane do niego. Żadnego zanieczyszczenia kontekstu. 50k tokenów czytania dokumentacji przez badacza nigdy nie trafia do kontekstu recenzenta.

### Krok 4: Porównanie

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Pojedynczy Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokeny: ${single.tokensUsed}`);
  console.log(`Wywołania narzędzi: ${single.toolCalls}`);

  console.log("\n=== Wieloagentowy ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokeny: ${multi.tokensUsed}`);
  console.log(`Wywołania narzędzi: ${multi.toolCalls}`);
}
```

Wersja wieloagentowa zużywa więcej tokenów ogólnie (trzech agentów, trzy oddzielne wywołania LLM), ale kontekst każdego agenta pozostaje czysty. Jakość każdego etapu poprawia się, ponieważ prompt systemowy jest wyspecjalizowany.

## Use It

Ta lekcja tworzy wielokrotnego użytku prompt do podejmowania decyzji o przejściu na wielu agentów. Zobacz `outputs/prompt-multi-agent-decision.md`.

## Exercises

1. Dodaj czwartego specjalistę: agenta „tester", który otrzymuje kod od programisty i opinie z recenzji od recenzenta, a następnie pisze testy
2. Zmodyfikuj potok tak, aby recenzent mógł wysłać informację zwrotną z powrotem do programisty w celu pętli poprawek (maks. 2 rundy)
3. Przekształć sekwencyjny potok w rozgałęzienie: uruchom badacza i agenta „analityka wymagań" równolegle, a następnie scal ich wyniki przed przekazaniem do programisty

## Key Terms

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|----------------------|
| Rój | „Ul umysłowy agentów AI" | Zbiór równorzędnych agentów ze współdzielonym stanem i bez stałego lidera. Zachowanie wyłania się z lokalnych interakcji. |
| Orkiestrator | „Agent szef" | Agent, którego narzędzia obejmują tworzenie i zarządzanie innymi agentami. Planuje i deleguje, ale może nie wykonywać właściwej pracy. |
| Koordynator | „Policjant drogowy" | Komponent niebędący agentem (często tylko kod, a nie LLM), który kieruje wiadomości między agentami na podstawie reguł. |
| Konsensus | „Agenci się zgadzają" | Protokół, w którym wielu agentów musi osiągnąć porozumienie przed kontynuowaniem. Używany, gdy sprzeczne wyniki wymagają rozstrzygnięcia. |
| Zachowanie emergentne | „Agenci sami to rozpracowali" | Wzorce na poziomie systemu wynikające z interakcji agentów, ale nie zostały jawnie zaprogramowane. Mogą być użyteczne lub szkodliwe. |
| Rozgałęzienie / scalenie | „Map-reduce dla agentów" | Dzielenie zadania na równoległych agentów (rozgałęzienie), a następnie łączenie ich wyników (scalenie). |
| Przekazywanie wiadomości | „Agenci rozmawiają ze sobą" | Mechanizm komunikacji między agentami: ustrukturyzowane dane wysyłane od jednego agenta do drugiego, zastępujące współdzielone okna kontekstu. |

## Further Reading

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) — przegląd wzorców wieloagentowych
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) — framework rozmów wieloagentowych Microsoftu
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) — jak Claude Code deleguje za pomocą Task
- [CrewAI documentation](https://docs.crewai.com/) — framework wieloagentowy oparty na rolach