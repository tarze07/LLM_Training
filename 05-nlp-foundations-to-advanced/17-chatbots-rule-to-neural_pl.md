# Chatboty — Od Regułowych przez Neuronowe do Agentów LLM

> ELIZA odpowiadała dopasowaniami wzorców. DialogFlow mapował intencje. GPT odpowiadał z wag. Claude uruchamia narzędzia i weryfikuje. Każda era naprawiała najgorszą porażkę poprzedniej.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## Problem

Użytkownik mówi "Chcę zmienić mój lot." System musi ustalić, czego chce, jakich informacji brakuje, jak je zdobyć i jak wykonać akcję. Następnie użytkownik mówi "chwila, a co jeśli zamiast tego anuluję?" i system musi zapamiętać kontekst, przełączyć zadania i zachować stan.

Rozmowa jest trudna dla systemu ML. Wejście jest otwarte. Wyjście musi być spójne przez wiele tur. System może potrzebować działać w świecie (zmienić lot, obciążyć kartę). Każdy zły krok jest widoczny dla użytkownika.

Architektury chatbotów przeszły przez cztery paradygmaty, każdy wprowadzony, ponieważ poprzedni zawodził zbyt widocznie. Ta lekcja omawia je po kolei. Krajobraz produkcyjny 2026 to hybryda dwóch ostatnich.

## Koncepcja

![Ewolucja chatbotów: regułowe → wyszukiwanie → neuronowe → agent](../assets/chatbot.svg)

**Regułowe (ELIZA, AIML, DialogFlow).** Ręcznie napisane wzorce dopasowują wejście użytkownika i generują odpowiedzi. Klasyfikatory intencji kierują do predefiniowanych przepływów. Maszyny stanów wypełniania slotów zbierają wymagane informacje. Działa znakomicie w wąskim zakresie, dla którego zostało zaprojektowane. Zawodzi natychmiast poza nim. Wciąż jest używane w domenach krytycznych dla bezpieczeństwa (uwierzytelnianie bankowe, rezerwacja lotów), gdzie halucynacje nie są tolerowane.

**Oparte na wyszukiwaniu.** System w stylu FAQ. Zakoduj każdą parę (wypowiedź, odpowiedź). W czasie wykonania zakoduj wiadomość użytkownika i pobierz najbliższą zapisaną odpowiedź. Pomyśl o klasycznej funkcji "podobne artykuły" Zendesk. Lepiej radzi sobie z parafrazami niż reguły. Brak generacji, więc brak halucynacji.

**Neuronowe (seq2seq).** Enkoder-dekoder trenowany na logach konwersacji. Generuje odpowiedzi od zera. Płynne, ale podatne na ogólne odpowiedzi ("I don't know") i dryf faktograficzny. Nigdy niezawodnie na temat. Powód, dla którego Google, Facebook i Microsoft miały rozczarowujące chatboty w latach 2016-2019.

**Agenci LLM.** Model językowy owinięty w pętlę, która planuje, wywołuje narzędzia i weryfikuje wyniki. Nie chatbot z długim promptem. Pętla agenta: planuj → wywołaj narzędzie → obserwuj wynik → zdecyduj następny krok. Ugruntowanie przez wyszukiwanie (RAG) zapobiega halucynacjom. Wywołania narzędzi pozwalają mu faktycznie robić rzeczy. To jest architektura 2026 roku.

Cztery paradygmaty nie są sekwencyjnymi zamiennikami. Produkcyjny chatbot w 2026 roku kieruje przez wszystkie cztery: regułowe do uwierzytelniania i destrukcyjnych akcji, wyszukiwanie do FAQ, generację neuronową do naturalnego formułowania, agenta LLM do niejednoznacznych zapytań otwartych.

## Zbuduj To

### Krok 1: regułowe dopasowywanie wzorców

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA w 20 liniach. Sztuczka odbicia ("I feel sad" → "Why do you feel sad") to kanoniczne demo psychoterapeuty z Weizenbaum 1966. Wciąż pouczające.

### Krok 2: oparte na wyszukiwaniu (FAQ)

Ten ilustracyjny fragment wymaga `pip install sentence-transformers` (który ściąga torch). Uruchamialny `code/main.py` dla tej lekcji używa zamiast tego podobieństwa Jaccarda ze stdlib, więc lekcja działa bez zewnętrznych zależności.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

Odmowa oparta na progu to kluczowa decyzja projektowa. Jeśli najlepsze dopasowanie nie jest wystarczająco blisko, zwróć `None` i pozwól systemowi eskalować.

### Krok 3: generacja neuronowa (baseline)

Użyj małego enkodera-dekodera dostrojonego do instrukcji (FLAN-T5) lub dostrojonego modelu konwersacyjnego. Samodzielnie bezużyteczne produkcyjnie w 2026 (sprzeczności, dryf tematyczny, faktograficzny nonsens), ale jest używane w systemach hybrydowych do naturalnego formułowania. Modele tylko z dekoderem w stylu DialoGPT potrzebują jawnych separatorów tur i obsługi EOS, aby produkować spójne odpowiedzi; potok text2text FLAN-T5 działa od razu dla przykładu dydaktycznego.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Krok 4: pętla agenta LLM

Kształt produkcyjny 2026:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

Trzy rzeczy do nazwania. Narzędzia to wywoływalne funkcje, które LLM może wywołać. Pętla kończy się, gdy LLM zwróci ostateczną odpowiedź zamiast wywołania narzędzia. Budżet kroków zapobiega nieskończonym pętlom przy niejednoznacznych zadaniach.

Prawdziwa produkcja dodaje: ugruntowanie przez wyszukiwanie (wstrzykuj odpowiednie dokumenty przed każdym wywołaniem LLM), zabezpieczenia (odmawiaj destrukcyjnych akcji bez potwierdzenia), obserwowalność (loguj każdy krok) i ewaluacje (automatyczne sprawdzenia, czy zachowanie agenta pozostaje zgodne ze specyfikacją).

### Krok 5: routing hybrydowy

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

Wzorzec: deterministyczne reguły dla wszystkiego, co destrukcyjne, wyszukiwanie dla gotowych FAQ, agenci LLM dla wszystkiego innego. To jest to, co jest wdrażane w systemach obsługi klienta w 2026 roku.

## Użyj Tego

Stos w 2026 roku:

| Zastosowanie | Architektura |
|---------|---------------|
| Rezerwacja, płatność, uwierzytelnianie | Regułowe maszyny stanów + wypełnianie slotów |
| FAQ obsługi klienta | Wyszukiwanie w przygotowanych odpowiedziach |
| Otwarty czat pomocy | Agent LLM z RAG + wywołania narzędzi |
| Narzędzia wewnętrzne / asystenci IDE | Agent LLM z wywołaniami narzędzi (szukaj, czytaj, pisz) |
| Chatboty towarzyszące / postaci | Dostrojony LLM z promptem osobowości, wyszukiwanie w wiedzy |

Zawsze używaj routingu hybrydowego w produkcji. Żadna pojedyncza architektura nie obsługuje dobrze każdego żądania. Sama warstwa routingu to zazwyczaj mały klasyfikator intencji.

## Tryby awarii, które wciąż występują

- **Pewne fabrykowanie.** Agent LLM twierdzi, że wykonał akcję, której nie wykonał. Łagodzenie: weryfikuj wyniki, loguj wywołania narzędzi, nigdy nie pozwalaj LLM twierdzić, że coś zrobił bez pomyślnego zwrotu narzędzia.
- **Prompt injection.** Użytkownik wstawia tekst, który nadpisuje prompt systemowy. Zajmuje LLM01 w OWASP Top 10 dla Aplikacji LLM 2025. Dwa rodzaje: bezpośrednia iniekcja (wklejona do czatu) i pośrednia iniekcja (ukryta w dokumentach, e-mailach lub wynikach narzędzi, które agent czyta).

  Wskaźniki ataków różnią się w zależności od scenariusza. Zmierzone wskaźniki sukcesu wahają się od ~0.5-8.5% w modelach granicznych w ogólnych benchmarkach użycia narzędzi i kodowania. Specyficzne konfiguracje wysokiego ryzyka (ataki adaptacyjne na agentów kodujących AI, podatna orkiestracja) osiągnęły ~84%. Produkcyjne CVE obejmują EchoLeak (CVE-2025-32711, CVSS 9.3) — zero-click wada eksfiltracji danych w Microsoft 365 Copilot wyzwalana przez e-mail kontrolowany przez atakującego.

  Łagodzenia: traktuj dane wejściowe użytkownika jako niezaufane w całej pętli; sanityzuj przed wywołaniami narzędzi; izoluj wyniki narzędzi od głównego promptu; używaj wzorca Plan-Verify-Execute (PVE), gdzie agent najpierw planuje, a następnie weryfikuje każdą akcję względem tego planu przed wykonaniem (to zapobiega wstrzykiwaniu nowych niezaplanowanych akcji przez wyniki narzędzi); wymagaj potwierdzenia użytkownika dla destrukcyjnych akcji; stosuj zasadę najmniejszego uprzywilejowania do zakresów narzędzi.

  Żadna ilość inżynierii promptów nie eliminuje w pełni tego ryzyka. Wymagane są zewnętrzne warstwy obrony w czasie wykonania (LLM Guard, walidacja na białej liście, semantyczne wykrywanie anomalii).
- **Rozszerzanie zakresu.** Agent schodzi z zadania, ponieważ wywołanie narzędzia zwróciło stycznie powiązane informacje. Łagodzenie: zawężaj kontrakty narzędzi; utrzymuj skupiony prompt systemowy; dodaj ewaluacje wskaźnika schodzenia z zadania.
- **Nieskończone pętle.** Agent ciągle wywołuje to samo narzędzie. Łagodzenie: budżet kroków, deduplikacja wywołań narzędzi, sędzia LLM na "czy robimy postępy."
- **Wyczerpanie okna kontekstu.** Długie rozmowy wypychają najwcześniejsze tury poza kontekst. Łagodzenie: podsumuj starsze tury, pobierz odpowiednie przeszłe tury przez podobieństwo lub użyj modelu z długim kontekstem.

## Wdróż To

Zapisz jako `outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj powyższą regułową odpowiedź z 10 wzorcami dla bota zamawiającego w kawiarni. Przetestuj przypadki brzegowe: podwójne zamówienia, modyfikacje, anulowanie, niejasny zamiar.
2. **Średnie.** Zbuduj hybrydowe FAQ + fallback LLM. 50 gotowych wpisów FAQ dla produktu SaaS, fallback LLM z wyszukiwaniem w witrynie dokumentacji. Zmierz wskaźnik odmowy i dokładność na 100 prawdziwych pytaniach wsparcia.
3. **Trudne.** Zaimplementuj powyższą pętlę agenta z trzema narzędziami (szukaj, czytaj-dane-użytkownika, wyślij-email). Uruchom ewaluację z 50 scenariuszami testowymi, w tym próbami prompt injection. Podaj wskaźnik schodzenia z zadania, wskaźnik nieudanych zadań i wszelkie udane iniekcje.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Intencja | Czego chce użytkownik | Etykieta kategoryczna (book_flight, reset_password). Kierowana do handlera. |
| Slot | Kawałek informacji | Parametr, którego potrzebuje bot (data, miejsce docelowe). Wypełnianie slotów to sekwencja pytań. |
| RAG | Wyszukiwanie plus generacja | Pobierz odpowiednie dokumenty, a następnie ugruntuj odpowiedź LLM. |
| Wywołanie narzędzia | Wywołanie funkcji | LLM emituje strukturalne wywołanie z nazwą + argumentami. Środowisko wykonawcze wykonuje, zwraca wynik. |
| Pętla agenta | Planuj, działaj, weryfikuj | Kontroler, który uruchamia wywołania LLM przeplatane wywołaniami narzędzi, aż zadanie zostanie ukończone. |
| Prompt injection | Użytkownik atakuje prompt | Złośliwe wejście, które próbuje nadpisać prompt systemowy. |

## Dalsza Lektura

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — oryginalny artykuł o regułowym chatbotcie.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — późny artykuł Google o neuronowym chatbotcie, tuż przed przejęciem przez agentów LLM.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — artykuł, który nazwał wzorzec pętli agenta.
- [Przewodnik Anthropic o budowaniu efektywnych agentów](https://www.anthropic.com/research/building-effective-agents) — wytyczne produkcyjne z 2024, które wciąż obowiązują w 2026.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — artykuł o prompt injection.
- [OWASP Top 10 dla Aplikacji LLM 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — ranking, który uczynił prompt injection najważniejszym zagrożeniem bezpieczeństwa.
- [AWS — Zabezpieczanie Amazon Bedrock Agents przed pośrednimi iniekcjami promptów](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — praktyczne zabezpieczenia warstwy orkiestracji, w tym Plan-Verify-Execute i przepływy potwierdzenia użytkownika.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — kanoniczne CVE zero-click eksfiltracji danych z pośredniej iniekcji promptu. Przypadek referencyjny, dlaczego agenci z dostępem do zapisu potrzebują zabezpieczeń w czasie wykonania.