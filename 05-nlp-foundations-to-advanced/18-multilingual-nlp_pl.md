# Wielojęzyczne NLP

> Jeden model, 100+ języków, zero danych treningowych dla większości z nich. Transfer międzyjęzykowy to praktyczny cud lat 2020.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 minutes

## Problem

Angielski ma miliardy oznaczonych przykładów. Urdu ma tysiące. Maithili ma prawie żadnych. Każdy praktyczny system NLP obsługujący globalną publiczność musi działać na długim ogonie języków, dla których nie istnieją dane treningowe specyficzne dla zadania.

Modele wielojęzyczne rozwiązują to, trenując jeden model na wielu językach jednocześnie. Wspólna reprezentacja pozwala modelowi przenosić umiejętności wyuczone w językach z dużą ilością zasobów do tych z małą ilością. Dostrój model na angielskiej analizie sentymentu, a on produkuje zaskakująco dobre przewidywania sentymentu dla Urdu od razu. To jest zerowy transfer międzyjęzykowy i zmieniło to, jak NLP trafia do świata.

Ta lekcja nazywa kompromisy, kanoniczne modele i jedną decyzję, która potyka zespoły nowe w pracy wielojęzycznej: wybór języka źródłowego do transferu.

## Koncepcja

![Transfer międzyjęzykowy przez wspólną wielojęzyczną przestrzeń osadzeń](../assets/multilingual.svg)

**Wspólne słownictwo.** Modele wielojęzyczne używają tokenizatora SentencePiece lub WordPiece wytrenowanego na tekście ze wszystkich języków docelowych. Słownictwo jest współdzielone: ta sama jednostka podsłowna reprezentuje ten sam morfem w pokrewnych językach. `anty-` w angielskim i polskim dostaje ten sam token.

**Wspólna reprezentacja.** Transformer wstępnie wytrenowany na maskowanym modelowaniu języka w wielu językach uczy się, że semantycznie podobne zdania w różnych językach produkują podobne stany ukryte. mBERT, XLM-R i NLLB wszystkie to wykazują. Osadzenia dla "cat" w angielskim grupują się blisko "chat" we francuskim i "gato" w hiszpańskim, podobnie jak osadzenia całych zdań.

**Transfer zerowy.** Dostrój model na oznaczonych danych w jednym języku (zazwyczaj angielskim). W inferencji uruchom go na dowolnym innym języku obsługiwanym przez model. Nie są potrzebne etykiety w języku docelowym. Wyniki są silne dla języków typologicznie spokrewnionych i słabsze dla odległych.

**Fine-tuning z kilkoma przykładami.** Dodaj 100-500 oznaczonych przykładów w języku docelowym. Dokładność skacze do 95-98% angielskiego baseline'u w zadaniach klasyfikacji. To jest najbardziej opłacalna dźwignia w wielojęzycznym NLP.

## Modele

| Model | Rok | Zasięg | Uwagi |
|-------|------|----------|-------|
| mBERT | 2018 | 104 języki | Trenowany na Wikipedii. Pierwszy praktyczny wielojęzyczny LM. Słaby na językach z małą ilością zasobów. |
| XLM-R | 2019 | 100 języków | Trenowany na CommonCrawl (znacznie większym niż Wikipedia). Ustanawia baseline międzyjęzykowy. Base 270M, Large 550M. |
| XLM-V | 2023 | 100 języków | XLM-R ze słownictwem 1M tokenów (vs 250k). Lepszy na językach z małą ilością zasobów. |
| mT5 | 2020 | 101 języków | Architektura T5 do wielojęzycznej generacji. |
| NLLB-200 | 2022 | 200 języków | Model tłumaczenia Meta; zawiera 55 języków z małą ilością zasobów. |
| BLOOM | 2022 | 46 języków + 13 programowania | Otwarty 176B LLM trenowany wielojęzycznie. |
| Aya-23 | 2024 | 23 języki | Wielojęzyczny LLM Cohere. Silny na arabskim, hindi, suahili. |

Wybierz według przypadku użycia. Klasyfikacja działa dobrze z XLM-R-base jako rozsądnym domyślnym wyborem. Zadania generacyjne wymagają mT5 lub NLLB w zależności od tłumaczenia vs otwartej generacji. Praca w stylu LLM łączy się z Aya-23 lub Claude przy użyciu jawnego wielojęzycznego promptowania.

## Decyzja o języku źródłowym (badania 2026)

Większość zespołów domyślnie wybiera angielski jako źródło fine-tuningu. Najnowsze badania (2026) pokazują, że to często błędne.

Podobieństwo językowe przewiduje jakość transferu lepiej niż surowy rozmiar korpusu. Dla celów słowiańskich, niemiecki lub rosyjski często biją angielski. Dla celów indyjskich, hindi często bije angielski. Metryka podobieństwa **qWALS** (2026, oparta na cechach World Atlas of Language Structures) kwantyfikuje to. **LANGRANK** (Lin et al., ACL 2019) to oddzielna, wcześniejsza metoda, która rankinguje potencjalne języki źródłowe z kombinacji podobieństwa językowego, rozmiaru korpusu i pokrewieństwa genetycznego.

Praktyczna zasada: jeśli twój język docelowy ma typologicznie bliskiego krewnego z dużą ilością zasobów, spróbuj najpierw fine-tuningu na nim, a następnie porównaj z fine-tuningiem na angielskim.

## Zbuduj To

### Krok 1: zerowa klasyfikacja międzyjęzykowa

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

Jeden model, trzy języki, to samo API. XLM-R trenowany na danych NLI dobrze przenosi się na klasyfikację przez sztuczkę entailment.

### Krok 2: wielojęzyczna przestrzeń osadzeń

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Tłumaczenia lądują blisko w przestrzeni osadzeń. Inne angielskie zdanie ląduje dalej. To sprawia, że działa międzyjęzykowe wyszukiwanie, grupowanie i podobieństwo.

### Krok 3: strategia fine-tuningu z kilkoma przykładami

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

Dla 100-500 przykładów w języku docelowym, `num_train_epochs=5` i `learning_rate=2e-5` to bezpieczne wartości domyślne. Wyższe wskaźniki uczenia powodują załamanie wielojęzycznego dopasowania i otrzymujesz model tylko angielski.

## Ewaluacja, która faktycznie działa

- **Dokładność na język na wstrzymanych zestawach.** Nie zagregowana. Agregacja ukrywa długi ogon.
- **Benchmark względem jednojęzycznego baseline'u.** Dla języków z wystarczającą ilością danych, jednojęzyczny model trenowany od zera czasami bije wielojęzyczny. Przetestuj.
- **Testy na poziomie encji.** Nazwy własne w języku docelowym. Modele wielojęzyczne często mają słabą tokenizację dla skryptów dalekich od łacińskiego.
- **Spójność międzyjęzykowa.** To samo znaczenie w dwóch językach powinno dawać tę samą predykcję. Zmierz lukę.

## Użyj Tego

Stos w 2026 roku:

| Zadanie | Zalecane |
|-----|-------------|
| Klasyfikacja, 100 języków | XLM-R-base (~270M) dostrojony |
| Zerowa klasyfikacja tekstu | `joeddav/xlm-roberta-large-xnli` |
| Wielojęzyczne osadzenia zdań | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Tłumaczenie, 200 języków | `facebook/nllb-200-distilled-600M` (patrz lekcja 11) |
| Generatywne wielojęzyczne | Claude, GPT-4, Aya-23, mT5-XXL |
| NLP dla języków z małą ilością zasobów | XLM-V lub domenowy fine-tuning na pokrewnym języku z dużą ilością zasobów |

Zawsze planuj budżet na fine-tuning w języku docelowym, jeśli wydajność ma znaczenie. Zerowy transfer to punkt wyjścia, a nie ostateczna odpowiedź.

### Podatek tokenizacji (co idzie źle dla języków z małą ilością zasobów)

Modele wielojęzyczne dzielą jeden tokenizator między wszystkie swoje języki. To słownictwo jest trenowane na korpusie zdominowanym przez angielski, francuski, hiszpański, chiński, niemiecki. Dla każdego języka poza dominującym zestawem, trzy podatki kumulują się po cichu:

- **Podatek płodności.** Tekst w języku z małą ilością zasobów tokenizuje się na znacznie więcej tokenów na słowo niż angielski. Zdanie w hindi może potrzebować 3-5x więcej tokenów niż równoważne zdanie angielskie. To 3-5x zjada twoje okno kontekstu, wydajność treningu i opóźnienie.
- **Podatek odzyskiwania wariantów.** Każda literówka, wariant diakrytyczny, niedopasowanie normalizacji Unicode lub wariacja przypadku staje się zimną, niepowiązaną sekwencją w przestrzeni osadzeń. Model nie może nauczyć się korespondencji ortograficznych, które native speaker uważa za oczywiste.
- **Podatek przepełnienia pojemności.** Podatki 1 i 2 zużywają pozycje kontekstu, głębokość warstw i wymiary osadzeń. To, co pozostaje do rzeczywistego rozumowania, jest systematycznie mniejsze niż to, co język z dużą ilością zasobów dostaje z tego samego modelu.

Praktyczny symptom: twój model trenuje normalnie na hindi, krzywa straty wygląda dobrze, perplexity ewaluacyjne wygląda rozsądnie, a wyniki produkcyjne są subtelnie błędne. Morfologia załamuje się w środku zdania. Rzadkie fleksje pozostają nieodzyskiwalne. **Nie możesz przeskalować danych, aby naprawić zepsuty tokenizator.**

Łagodzenia: wybierz tokenizator z dobrym pokryciem dla twojego języka docelowego (słownictwo 1M tokenów XLM-V to bezpośrednia naprawa); zweryfikuj płodność tokenizacji na wstrzymanym tekście docelowym przed treningiem; użyj fallbacku bajtowego (SentencePiece `byte_fallback=True`, BPE na poziomie bajtów w stylu GPT-2) dla naprawdę długiego ogona skryptów, aby nic nie było nigdy OOV.

## Wdróż To

Zapisz jako `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Ćwiczenia

1. **Łatwe.** Uruchom potok zerowej klasyfikacji na 10 zdaniach na język w angielskim, francuskim, hindi i arabskim. Podaj dokładność dla każdego. Powinieneś zobaczyć silny francuski, przyzwoity hindi, zmienny arabski.
2. **Średnie.** Użyj `paraphrase-multilingual-MiniLM-L12-v2` do zbudowania międzyjęzycznej wyszukiwarki na małym mieszanym korpusie językowym. Zapytanie po angielsku, pobierz dokumenty w dowolnym języku. Zmierz recall@5.
3. **Trudne.** Porównaj fine-tuning ze źródłem angielskim i źródłem hindi dla zadania klasyfikacji w hindi. Użyj 500 przykładów w języku docelowym do fine-tuningu z kilkoma przykładami w obu reżimach. Podaj, które źródło daje lepszą dokładność hindi i o ile. To jest teza LANGRANK w miniaturze.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Model wielojęzyczny | Jeden model, wiele języków | Współdzielone słownictwo i parametry między językami. |
| Transfer międzyjęzykowy | Trenuj na jednym języku, uruchom na innym | Dostrój na źródle, oceń na celu bez etykiet w języku docelowym. |
| Zerowy | Brak etykiet w języku docelowym | Transfer bez fine-tuningu na języku docelowym. |
| Kilka przykładów | Małe etykiety docelowe | 100-500 przykładów w języku docelowym używanych do fine-tuningu. |
| mBERT | Pierwszy wielojęzyczny LM | 104-językowy BERT wstępnie wytrenowany na Wikipedii. |
| XLM-R | Standardowy baseline międzyjęzykowy | 100-językowy RoBERTa wstępnie wytrenowany na CommonCrawl. |
| NLLB | MT Meta na 200 języków | No Language Left Behind. Obejmuje 55 języków z małą ilością zasobów. |

## Dalsza Lektura

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — artykuł XLM-R.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — artykuł analityczny, który zapoczątkował linię badawczą transferu międzyjęzykowego.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — artykuł NLLB-200.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya, wielojęzyczny LLM Cohere.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — artykuł o języku źródłowym qWALS / LANGRANK.