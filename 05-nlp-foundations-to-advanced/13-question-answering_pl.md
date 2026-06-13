# Systemy Odpowiadania na Pytania

> Trzy systemy ukształtowały nowoczesne QA. Ekstrakcyjne znajdowało fragmenty. Wzbogacone o wyszukiwanie opierało je na dokumentach. Generatywne tworzyło odpowiedzi. Każdy nowoczesny asystent AI to mieszanka tych trzech.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 minutes

## Problem

Użytkownik wpisuje "Kiedy wypuszczono pierwszego iPhone'a?" i oczekuje "29 czerwca 2007 r." Nie "Historia Apple jest długa i zróżnicowana." Nie "2007" w izolacji, bez zdania. Bezpośrednia, ugruntowana, poprawna odpowiedź.

Trzy architektury dominowały w QA w ciągu ostatniej dekady.

- **Ekstrakcyjne QA.** Mając pytanie i fragment tekstu, który zawiera odpowiedź, znajdź indeksy początku i końca fragmentu odpowiedzi. SQuAD to kanoniczny benchmark.
- **Otwarte QA.** Fragment nie jest podany. Najpierw pobierz odpowiedni fragment, a następnie wyodrębnij lub wygeneruj odpowiedź. To podstawa każdego dzisiejszego potoku RAG.
- **Generatywne / Closed-book QA.** Duży model językowy odpowiada z pamięci parametrycznej. Brak wyszukiwania. Najszybsze w inferencji, najmniej niezawodne w faktach.

Trend w 2026 roku jest hybrydowy: pobierz najlepsze kilka fragmentów, a następnie zasil model generatywny, aby odpowiedział w oparciu o te fragmenty. To jest RAG, a lekcja 14 szczegółowo omawia połowę dotyczącą wyszukiwania. Ta lekcja buduje połowę QA.

## Koncepcja

![Architektury QA: ekstrakcyjna, wzbogacona o wyszukiwanie, generatywna](../assets/qa.svg)

**Ekstrakcyjne.** Zakoduj pytanie i fragment razem za pomocą transformera (rodzina BERT). Trenuj dwie głowice, które przewidują indeksy tokenów początku i końca odpowiedzi. Funkcja straty to cross-entropia dla poprawnych pozycji. Wynikiem jest fragment z tekstu. Nigdy nie halucynuje (z założenia), nigdy nie obsługuje pytań, na które fragment nie może odpowiedzieć (z założenia).

**Wzbogacone o wyszukiwanie (RAG).** Dwa etapy. Po pierwsze, wyszukiwarka znajduje górne `k` fragmentów z korpusu. Po drugie, czytnik (ekstrakcyjny lub generatywny) tworzy odpowiedź na podstawie tych fragmentów. Podział na wyszukiwarkę i czytnik pozwala na niezależne szkolenie i ocenę każdego z nich. Nowoczesny RAG często dodaje między nimi reranker.

**Generatywne.** LLM tylko z dekoderem (GPT, Claude, Llama) odpowiada z wyuczonych wag. Brak etapu wyszukiwania. Doskonałe w przypadku powszechnej wiedzy, katastrofalne w przypadku rzadkich lub najnowszych faktów. Wskaźnik halucynacji jest odwrotnie skorelowany z częstotliwością występowania faktu w danych treningowych.

## Zbuduj To

### Krok 1: ekstrakcyjne QA z wytrenowanym modelem

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` jest trenowany na SQuAD 2.0, który zawiera pytania, na które nie można odpowiedzieć. Domyślnie potok `question-answering` zwraca fragment z najwyższym wynikiem, nawet gdy wynik modelu dla "brak odpowiedzi" jest wyższy — nie zwraca automatycznie pustej odpowiedzi. Aby uzyskać jawne zachowanie "brak odpowiedzi", przekaż `handle_impossible_answer=True` do wywołania potoku: potok zwróci wtedy pustą odpowiedź tylko wtedy, gdy wynik dla braku odpowiedzi przekracza każdy wynik fragmentu. W każdym przypadku sprawdzaj pole `score`.

### Krok 2: potok wzbogacony o wyszukiwanie (szkic)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

Dwuetapowy potok. Gęsta wyszukiwarka (Sentence-BERT) znajduje odpowiednie fragmenty na podstawie podobieństwa semantycznego. Ekstrakcyjny czytnik (RoBERTa-SQuAD) wyciąga fragment odpowiedzi z połączonych górnych fragmentów. Działa na małych korpusach. Dla korpusu z milionem dokumentów użyj FAISS lub bazy wektorowej.

### Krok 3: generatywne z RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Wzorzec promptu ma znaczenie. Wyraźne poinstruowanie modelu, aby opierał się na kontekście i zwracał "I don't know", gdy kontekst jest niewystarczający, obniża wskaźnik halucynacji o 40-60% w porównaniu do naiwnego promptowania. Bardziej rozbudowane wzorce dodają cytowania, wyniki ufności i strukturalną ekstrakcję.

### Krok 4: ewaluacja odzwierciedlająca rzeczywisty świat

SQuAD używa **Exact Match (EM)** i **F1 na poziomie tokenów**. EM to ścisłe dopasowanie po normalizacji (małe litery, usunięcie interpunkcji, usunięcie rodzajników) — albo przewidywanie pasuje dokładnie, albo dostaje 0. F1 jest obliczane na podstawie nakładania się tokenów między przewidywaniem a referencją i daje częściowe punkty. Oba nie doceniają parafraz: "June 29, 2007" vs "June 29th, 2007" zazwyczaj dostaje 0 EM (liczebnik porządkowy psuje normalizację), ale wciąż zdobywa znaczące F1 z nakładających się tokenów.

Dla produkcyjnego QA:

- **Dokładność odpowiedzi** (oceniana przez LLM lub człowieka, ponieważ metryki nie oddają równoważności semantycznej).
- **Dokładność cytowania.** Czy cytowany fragment faktycznie wspiera odpowiedź? Banalne do automatycznego sprawdzenia przez dopasowanie ciągów między wygenerowanymi cytowaniami a pobranymi fragmentami.
- **Kalibracja odmowy.** Gdy odpowiedzi nie ma w pobranych fragmentach, czy system poprawnie mówi "I don't know"? Mierz wskaźnik fałszywej pewności.
- **Recall wyszukiwania.** Przed oceną czytnika zmierz, czy wyszukiwarka umieszcza właściwy fragment w górnym `k`. Czytnik nie może naprawić brakującego fragmentu.

### RAGAS: produkcyjny framework ewaluacji w 2026

`RAGAS` jest stworzony specjalnie dla systemów RAG i jest domyślnym wyborem w 2026 roku. Ocenia cztery wymiary bez potrzeby posiadania złotych referencji:

- **Wierność.** Czy każde twierdzenie w odpowiedzi pochodzi z pobranego kontekstu? Mierzone przez NLI-based entailment. Twoja podstawowa metryka halucynacji.
- **Trafność odpowiedzi.** Czy odpowiedź odnosi się do pytania? Mierzone przez generowanie hipotetycznych pytań z odpowiedzi i porównywanie z rzeczywistym pytaniem.
- **Precyzja kontekstu.** Jaka część pobranych fragmentów była rzeczywiście istotna? Niska precyzja = szum w prompcie.
- **Recall kontekstu.** Czy pobrany zestaw zawierał wszystkie potrzebne informacje? Niski recall = czytnik nie może odnieść sukcesu.

Ocena bez referencji pozwala na ewaluację na żywym ruchu produkcyjnym bez przygotowanych złotych odpowiedzi. Dodaj LLM-as-judge na wierzchu dla pytań otwartych, gdzie metryki dokładnego dopasowania są bezużyteczne.

`pip install ragas`. Podłącz wyszukiwarkę + czytnik. Otrzymaj cztery skalary na zapytanie. Alertuj na regresje.

## Użyj Tego

Stos w 2026 roku.

| Zastosowanie | Zalecane |
|---------|-------------|
| Dany fragment, znajdź fragment odpowiedzi | `deepset/roberta-base-squad2` |
| Dla stałego korpusu, closed-book nieakceptowalne | RAG: gęsta wyszukiwarka + czytnik LLM |
| Czas rzeczywisty w magazynie dokumentów | RAG z hybrydową (BM25 + gęsta) wyszukiwarką + reranker (lekcja 14) |
| Konwersacyjne QA (pytania uzupełniające) | LLM z historią konwersacji + RAG na każdej turze |
| Wysoce faktograficzne, regulowane domeny | Ekstrakcyjne na autorytatywnym korpusie; nigdy samo generatywne |

Ekstrakcyjne QA jest niemodne w 2026 roku, ponieważ RAG z LLM obsługuje więcej przypadków. Wciąż jest używane w kontekstach, gdzie wymagane jest dosłowne cytowanie: badania prawne, zgodność z przepisami, narzędzia audytowe.

## Wdróż To

Zapisz jako `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## Ćwiczenia

1. **Łatwe.** Skonfiguruj powyższy ekstrakcyjny potok SQuAD na 10 fragmentach z Wikipedii. Ręcznie przygotuj 10 pytań. Zmierz, jak często odpowiedź jest poprawna. Powinieneś zobaczyć 7-9 poprawnych, jeśli fragmenty i pytania są czyste.
2. **Średnie.** Dodaj klasyfikator odmowy. Gdy górny wynik wyszukiwania jest poniżej progu (np. 0.3 cosinus), zwróć "I don't know" zamiast wywoływać czytnik. Dostrój próg na zestawie wstrzymanym.
3. **Trudne.** Zbuduj potok RAG na wybranym korpusie 10 000 dokumentów. Zaimplementuj hybrydowe wyszukiwanie (BM25 + gęste) z fuzją RRF (patrz lekcja 14). Zmierz dokładność odpowiedzi z krokiem hybrydowym i bez niego. Udokumentuj, które typy pytań odnoszą największe korzyści.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Ekstrakcyjne QA | Znajdź fragment odpowiedzi | Przewidź indeksy początku i końca odpowiedzi w danym fragmencie. |
| Otwarte QA | QA na korpusie | Brak danego fragmentu; trzeba najpierw wyszukać, potem odpowiedzieć. |
| RAG | Wyszukaj, potem wygeneruj | Generacja wzbogacona o wyszukiwanie. Potok wyszukiwarka + czytnik. |
| SQuAD | Kanoniczny benchmark | Stanford Question Answering Dataset. Metryki EM + F1. |
| Halucynacja | Zmyślona odpowiedź | Wynik czytnika niepoparty pobranym kontekstem. |
| Kalibracja odmowy | Wiedzieć, kiedy milczeć | System poprawnie mówi "I don't know", gdy nie jest w stanie odpowiedzieć. |

## Dalsza Lektura

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — artykuł benchmarkowy.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, kanoniczna gęsta wyszukiwarka dla QA.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — artykuł, który nazwał RAG.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — kompleksowy przegląd RAG.