# LLM Evaluation — RAGAS, DeepEval, G-Eval (Ewaluacja LLM — RAGAS, DeepEval, G-Eval)

> Exact-match i F1 nie wyłapują równoważności semantycznej. Recenzja ludzka nie skaluje się. LLM-as-judge to produkcyjna odpowiedź — z wystarczającą kalibracją, by ufać liczbie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## Problem

Twój system RAG odpowiada: "June 29th, 2007."
Złota referencja to: "June 29, 2007."
Exact Match daje 0. F1 daje ~75%. Człowiek dałby 100%.

Teraz pomnóż przez 10 000 przypadków testowych. Pomnóż jeszcze raz przez każdą zmianę retrievera, chunkingu, promptu lub modelu. Potrzebujesz ewaluatora, który rozumie znaczenie, działa tanio na skalę, nie kłamie o regresjach i ujawnia właściwe tryby awarii.

2026 ma trzy frameworki, które rozwiązują ten problem.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. Cztery metryki RAG (wiarygodność, trafność odpowiedzi, precyzja kontekstu, recall kontekstu) z backendami NLI + LLM. Poparte badaniami, lekkie.
- **DeepEval.** Pytest dla LLM. G-Eval, kompletność zadania, halucynacja, bias. Natywne dla CI/CD.
- **G-Eval.** Metoda (i metryka DeepEval): LLM-as-judge z chain-of-thought, niestandardowymi kryteriami, oceną 0-1.

Wszystkie trzy opierają się na LLM-as-judge. Ta lekcja buduje intuicję dla tej metody i warstwy zaufania wokół niej.

## Koncepcja

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge.** Zastąp statyczną metrykę LLM, który ocenia wyniki według rubryki. Dla `(zapytanie, kontekst, odpowiedź)` promptuj sędziego LLM: "Oceń 0-1 pod kątem wiarygodności." Zwróć wynik.

Dlaczego działa: LLM przybliżają ludzki osąd przy ułamku kosztów. GPT-4o-mini za ~$0.003 na oceniany przypadek umożliwia przebiegi ewaluacji regresyjnej na 1000 próbkach za mniej niż $5.

Dlaczego zawodzi po cichu:

1. **Bias sędziego.** Sędziowie wolą dłuższe odpowiedzi, odpowiedzi z własnej rodziny modeli, odpowiedzi pasujące do stylu promptu.
2. **Błędy parsowania JSON.** Nieprawidłowy JSON → wynik NaN → ciche wykluczenie z agregacji. Użytkownicy RAGAS znają ten ból. Zabezpiecz try/except + jawny tryb awarii.
3. **Dryf wraz z wersjami modelu.** Aktualizacja sędziego zmienia każdą metrykę. Zamroź model sędziego + wersję.

**Cztery metryki RAG.**

| Metryka | Pytanie | Backend |
|--------|----------|---------|
| Wiarygodność | Czy każde twierdzenie w odpowiedzi pochodzi z pobranego kontekstu? | NLI-based entailment |
| Trafność odpowiedzi | Czy odpowiedź odnosi się do pytania? | Generuj hipotetyczne pytania z odpowiedzi; porównaj z rzeczywistym pytaniem |
| Precyzja kontekstu | Jaka część pobranych fragmentów była istotna? | LLM-judge |
| Recall kontekstu | Czy wyszukiwanie zwróciło wszystko, co potrzebne? | LLM-judge względem złotej odpowiedzi |

**G-Eval.** Zdefiniuj niestandardowe kryterium: "Czy odpowiedź zacytowała właściwe źródło?" Framework automatycznie rozwija to w kroki ewaluacji chain-of-thought, a następnie ocenia 0-1. Dobre dla domenowych wymiarów jakości, których RAGAS nie obejmuje.

**Kalibracja.** Nigdy nie ufaj surowej ocenie sędziego, dopóki nie masz korelacji z ludzkimi etykietami. Przeprowadź 100 ręcznie oznaczonych przykładów. Wykreśl sędziego vs człowieka. Oblicz Spearman rho. Jeśli rho < 0.7, twoja rubryka sędziego wymaga pracy.

## Zbuduj To

### Krok 1: wiarygodność z NLI (w stylu RAGAS)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Rozłóż odpowiedź na atomowe twierdzenia. Sprawdź NLI każde twierdzenie względem pobranego kontekstu. Wiarygodność = frakcja popartych.

### Krok 2: trafność odpowiedzi

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

Jeśli odpowiedź implikuje inne pytania niż zadane, trafność spada.

### Krok 3: niestandardowa metryka G-Eval

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Kroki ewaluacji to rubryka. Wyraźne kroki są bardziej stabilne niż niejawne promptowanie "oceń 0-1".

### Krok 4: bramka CI

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Dostarcz jako plik pytest. Uruchamiaj na każdym PR. Blokuj scalanie przy regresjach.

### Krok 5: zabawkowa ewaluacja od zera

Zobacz `code/main.py`. Przybliżenia wiarygodności (pokrycie twierdzeń odpowiedzi z kontekstem) i trafności (pokrycie tokenów odpowiedzi z tokenami pytania) tylko przy użyciu stdlib. Nieprodukcyjne. Pokazuje kształt.

## Pułapki

- **Brak kalibracji.** Sędzia z korelacją 0.3 z ludzkimi etykietami to szum. Wymagaj przebiegu kalibracyjnego przed wdrożeniem.
- **Samoocena.** Używanie tego samego LLM do generowania i oceniania zawyża wyniki o 10-20%. Użyj innej rodziny modeli dla sędziego.
- **Bias pozycyjny w ocenie parami.** Sędziowie preferują pierwszą opcję. Zawsze randomizuj kolejność i oceniaj obie strony.
- **Surowa agregacja ukrywa porażki.** Średni wynik 0.85 często ukrywa 5% katastrofalnych błędów. Zawsze sprawdzaj najniższy kwantyl.
- **Gnijący złoty zbiór.** Niewersjonowane zbiory ewaluacyjne, które dryfują w czasie, psują porównania podłużne. Oznacz zbiór tagiem przy każdej zmianie.
- **Koszt LLM.** Na dużą skalę, wywołania sędziego dominują koszty. Użyj najtańszego modelu spełniającego próg kalibracji. GPT-4o-mini, Claude Haiku, Mistral-small.

## Użyj Tego

Stos w 2026:

| Zastosowanie | Framework |
|---------|-----------|
| Monitorowanie jakości RAG | RAGAS (4 metryki) |
| Bramki regresyjne CI/CD | DeepEval + pytest |
| Niestandardowe kryteria domenowe | G-Eval w ramach DeepEval |
| Monitorowanie ruchu na żywo online | RAGAS z trybem bez referencji |
| Kontrole punktowe z udziałem człowieka | LangSmith lub Phoenix z interfejsem UI do adnotacji |
| Red-teaming / ewaluacja bezpieczeństwa | Promptfoo + DeepEval |

Typowy stos: RAGAS do monitorowania, DeepEval do CI, G-Eval dla nowych wymiarów. Uruchom wszystkie trzy; przydatnie się różnią.

## Dostarcz To

Zapisz jako `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Ćwiczenia

1. **Łatwe.** Użyj RAGAS na 10 przykładach RAG ze znanymi halucynacjami. Sprawdź, czy metryka wiarygodności wyłapuje każdą z nich.
2. **Średnie.** Ręcznie oznacz 50 odpowiedzi QA 0-1 pod kątem poprawności. Oceń za pomocą G-Eval. Zmierz Spearman rho między sędzią a człowiekiem.
3. **Trudne.** Zbuduj bramkę CI pytest z DeepEval. Celowo pogorsz retriever. Sprawdź, czy bramka zawodzi. Dodaj alertowanie na najniższy kwantyl poprzez sprawdzenie progu na najniższych 10%.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|------|-----------------|-----------------------|
| LLM-as-judge | Ocena za pomocą LLM | Promptuj model sędziego do oceny wyników 0-1 według rubryki. |
| RAGAS | Biblioteka metryk RAG | Framework ewaluacyjny open-source z 4 metrykami RAG bez referencji. |
| Wiarygodność | Czy odpowiedź jest ugruntowana? | Frakcja twierdzeń odpowiedzi wynikających z pobranego kontekstu. |
| Precyzja kontekstu | Czy pobrane fragmenty były istotne? | Frakcja top-K fragmentów, które faktycznie miały znaczenie. |
| Recall kontekstu | Czy wyszukiwanie znalazło wszystko? | Frakcja twierdzeń złotej odpowiedzi wspartych przez pobrane fragmenty. |
| G-Eval | Niestandardowy sędzia LLM | Rubryka + kroki ewaluacji chain-of-thought + ocena 0-1. |
| Kalibracja | Ufaj, ale weryfikuj | Korelacja Spearmana między oceną sędziego a oceną człowieka. |

## Dalsza Literatura

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — the RAGAS paper.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — the G-Eval paper.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — open production stack.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — biases, calibration, limits.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — unifying framework that integrates RAGAS, DeepEval, Phoenix.