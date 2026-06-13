# Capstone 08 — Produkcyjny Chatbot RAG dla Regulowanej Wertykali

> Harvey, Glean, Mendable i LlamaCloud wszystkie działają w tym samym produkcyjnym kształcie w 2026 roku. Indeksacja z docling lub Unstructured i ColPali dla materiałów wizualnych. Wyszukiwanie hybrydowe. Ponowne sortowanie z bge-reranker-v2-gemma. Synteza z Claude Sonnet 4.7 przy użyciu prompt caching ze współczynnikiem trafień 60-80%. Zabezpieczenie z Llama Guard 4 i NeMo Guardrails. Monitorowanie z Langfuse i Phoenix. Ocena z RAGAS na 200-pytaniowym złotym zestawie. Zbuduj taki system w regulowanej domenie (prawnej, klinicznej, ubezpieczeniowej), a capstone polega na przejściu złotego zestawu, red team i kokpitu dryfu.

**Type:** Capstone
**Languages:** Python (pipeline + API), TypeScript (chat UI)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18
**Time:** 30 hours

## Problem

RAG w regulowanych domenach (umowy prawne, protokoły badań klinicznych, polisy ubezpieczeniowe) jest najczęściej wdrażanym produkcyjnym kształtem 2026 roku, ponieważ ROI jest oczywiste, a stawki są konkretne. Harvey (Allen & Overy) zbudował go dla prawa. Mendable dostarcza wersję dla dokumentacji deweloperskiej. Glean obejmuje wyszukiwanie korporacyjne. Wzór to: indeksacja wysokiej wierności, hybrydowe wyszukiwanie z ponownym sortowaniem, synteza z egzekwowaniem cytowań i prompt caching, zabezpieczenie wieloma warstwami bezpieczeństwa i ciągłe monitorowanie dryfu.

Trudne części to nie model. To zgodność jurysdykcyjna (HIPAA, GDPR, SOC2), audytowalność na poziomie cytowań, kontrola kosztów (prompt caching daje 60-90% zniżki przy wysokim współczynniku trafień), wykrywanie halucynacji przez RAGAS faithfulness i wykrywanie dryfu, gdy dokumenty źródłowe są aktualizowane bez nadrabiania przez indeks. Ten capstone wymaga dostarczenia tego wszystkiego na 200-pytaniowym złotym zestawie z zestawem red team obok.

## Koncepcja

Potok ma dwie strony. **Indeksacja**: docling lub Unstructured parsuje ustrukturyzowane dokumenty; ColPali obsługuje bogate wizualnie; fragmenty otrzymują podsumowania, tagi i etykiety dostępu oparte na rolach. Wektory trafiają do pgvector + pgvectorscale (poniżej 50M wektorów) lub Qdrant Cloud; rzadki BM25 działa obok. **Rozmowa**: LangGraph obsługuje pamięć i wieloobrotowość; każde zapytanie uruchamia hybrydowe wyszukiwanie, sortuje z bge-reranker-v2-gemma-2b, syntezuje z Claude Sonnet 4.7 (prompt-cached), przepuszcza wyjście przez Llama Guard 4 i NeMo Guardrails i emituje odpowiedź z cytowaniami.

Stos ewaluacyjny ma cztery warstwy. **Złoty zestaw** (200 oznaczonych Pytanie/Odpowiedź z cytowaniami) dla poprawności. **Red team** (jailbreaks, próby ekstrakcji PII, pytania poza domeną) dla bezpieczeństwa. **RAGAS** dla wierności / trafności odpowiedzi / precyzji kontekstu automatycznie na każdą turę. **Kokpit dryfu** (Arize Phoenix) monitorujący jakość wyszukiwania i wynik halucynacji co tydzień.

Prompt caching jest dźwignią kosztową. Claude 4.5+ i GPT-5+ obsługują cache'owanie promptów systemowych + pobranego kontekstu. Przy współczynniku trafień 60-80%, koszt na zapytanie spada 3-5x. Potok musi być zaprojektowany dla stabilnych prefiksów (prompt systemowy + posegregowany kontekst jako pierwsze), aby osiągnąć wysokie współczynniki trafień cache.

## Architektura

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## Stack

- Indeksacja: Unstructured.io lub docling dla ustrukturyzowanych dokumentów; ColPali dla bogatych wizualnie PDFów
- Baza wektorowa: pgvector + pgvectorscale poniżej 50M wektorów; Qdrant Cloud w innym przypadku
- Rzadka: Tantivy BM25 z wagami pól
- Orkiestracja: LlamaIndex Workflows (indeksacja) + LangGraph (rozmowa)
- Ponowne sortowanie: bge-reranker-v2-gemma-2b samodzielnie hostowany lub Voyage rerank-2 hostowany
- LLM: Claude Sonnet 4.7 z prompt caching; zapasowy Llama 3.3 70B samodzielnie hostowany
- Ewaluacja: RAGAS 0.2 online, DeepEval dla zestawów halucynacji i jailbreak
- Obserwowalność: Langfuse samodzielnie hostowany z kolejką adnotacji; Arize Phoenix dla dryfu
- Zabezpieczenia: Llama Guard 4 klasyfikator wejścia/wyjścia, NeMo Guardrails v0.12 policy, Presidio PII scrub
- Zgodność: etykiety dostępu oparte na rolach na fragmentach; tagi jurysdykcyjne dla GDPR/HIPAA

```figure
canary-rollout
```

## Build It

1. **Indeksacja.** Sparsuj swój korpus (1000-10000 dokumentów dla poważnej implementacji) z Unstructured lub docling. Dla skanowanych / bogatych wizualnie stron, kieruj przez ColPali. Wyprodukuj fragmenty z podsumowaniami, etykietami ról, tagami jurysdykcyjnymi.

2. **Indeks.** Gęste osadzenia (Voyage-3 lub Nomic-embed-v2) do pgvector + pgvectorscale. Indeks boczny BM25 przez Tantivy. Filtry roli i jurysdykcji jako ładunek.

3. **Hybrydowe wyszukiwanie.** Najpierw filtruj według roli+jurysdykcji; następnie równolegle gęste + BM25; połącz z reciprocal rank fusion; top-20 do re-rankra; top-5 do syntezy.

4. **Synteza z prompt caching.** Prompt systemowy + statyczne polityki w nagłówku cache; posegregowany kontekst jako rozszerzenie cache; pytanie użytkownika jako niescache'owany sufiks. Celuj w 60-80% współczynnik trafień cache w stanie ustalonym.

5. **Zabezpieczenia.** Llama Guard 4 na wejściu; NeMo Guardrails blokują pytania poza domeną lub tematy zabronione przez politykę; Presidio czyści przypadkowe PII na wyjściu; filtr po-processingu egzekwujący cytowania.

6. **Złoty zestaw.** 200 par Pytanie/Odpowiedź oznaczonych przez eksperta domenowego z (odpowiedź, cytowania). Punktuj agenta za dokładne dopasowanie cytowań, poprawność odpowiedzi, wierność (RAGAS).

7. **Red team.** 50 adversarialnych promptów: jailbreaks (PAIR, TAP), próby eksfiltracji PII, pytania poza domeną, wycieki międzyjurysdykcyjne. Punktuj pass/fail i wagą.

8. **Kokpit dryfu.** Arize Phoenix śledzi jakość wyszukiwania (nDCG, wierność cytowań) co tydzień. Alarmuj przy 5% spadku.

9. **Raport kosztów.** Langfuse: współczynnik trafień prompt caching, tokeny na zapytanie, podział $/zapytanie według etapu.

## Use It

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Ship It

`outputs/skill-production-rag.md` opisuje rezultat. Chatbot w regulowanej domenie wdrożony z etykietami zgodności, przepuszczony przez rubrykę, monitorowany za pomocą live drift monitoring.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | Wyniki online na złotym zestawie (200 P/Q) |
| 20 | Poprawność cytowań | Frakcja odpowiedzi z weryfikowalnymi kotwicami źródłowymi |
| 20 | Pokrycie zabezpieczeń | Wskaźnik przepustowości Llama Guard 4 + wyniki zestawu jailbreak |
| 20 | Inżynieria kosztów/opóźnień | Współczynnik trafień prompt-cache, p95 opóźnienia, $/zapytanie |
| 15 | Kokpit monitorowania dryfu | Kokpit Phoenix na żywo z tygodniowym trendem jakości wyszukiwania |
| **100** | | |

## Ćwiczenia

1. Zbuduj drugi wycinek korpusu pod inną jurysdykcją (np. HIPAA obok GDPR). Zademonstruj filtrowanie rola+jurysdykcja zapobiegające międzyjurysdykcyjnemu wyciekowi na 20-pytaniowym teście.

2. Zmierz współczynnik trafień prompt-cache w ciągu tygodnia ruchu produkcyjnego. Zidentyfikuj, które zapytania łamią prefiks cache. Restrukturyzuj.

3. Dodaj wieloobrotową pamięć z buforem podsumowania 10k tokenów. Zmierz, czy wierność spada wraz ze wzrostem rozmowy.

4. Zamień Claude Sonnet 4.7 na samodzielnie hostowany Llama 3.3 70B. Zmierz $/zapytanie i różnicę wierności.

5. Dodaj tryb "niepewny": jeśli wyniki posegregowanych fragmentów są poniżej progu, agent mówi "nie mam pewnych cytowań" zamiast odpowiadać. Zmierz redukcję fałszywej pewności.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Prompt caching | "Cache'owany system + kontekst" | Funkcja Claude/OpenAI: cache'owane tokeny prefiksu rabatowane 60-90% przy trafieniu |
| RAGAS | "Ewaluator RAG" | Automatyczna punktacja wierności, trafności odpowiedzi, precyzji kontekstu |
| Golden set | "Oznaczona ewaluacja" | 200+ oznaczonych przez eksperta P/Q z cytowaniami; ground truth |
| Jurisdiction tag | "Etykieta zgodności" | Zakres GDPR/HIPAA/SOC2 dołączony do fragmentów; egzekwowany przez filtr wyszukiwania |
| Citation faithfulness | "Wskaźnik ugruntowanych odpowiedzi" | Frakcja twierdzeń popartych możliwymi do odzyskania zakresami źródłowymi |
| Drift | "Spadek jakości wyszukiwania" | Tygodniowa zmiana nDCG lub wyniku cytowań; próg alarmu 5% |
| Red team | "Ewaluacja adversarialna" | Jailbreak przed wydaniem, ekstrakcja PII, testy poza domeną |

## Dalsza lektura

- [Harvey AI](https://www.harvey.ai) — reference legal production stack
- [Glean enterprise search](https://www.glean.com) — reference RAG at enterprise scale
- [Mendable documentation](https://mendable.ai) — developer-docs RAG reference
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — managed ingestion
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — the cost-lever reference
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — the canonical RAG eval framework
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — reference drift observability
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 safety classifier
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — policy rail framework