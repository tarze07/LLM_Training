# Modelowanie Tematów — LDA i BERTopic

> LDA: dokumenty są mieszankami tematów, tematy są rozkładami na słowach. BERTopic: dokumenty grupują się w przestrzeni osadzeń, klastry są tematami. Ten sam cel, różne dekompozycje.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 minutes

## Problem

Masz 10 000 zgłoszeń obsługi klienta, 50 000 artykułów prasowych lub 200 000 tweetów. Musisz wiedzieć, o czym jest zbiór, bez czytania go. Nie masz oznaczonych kategorii. Nie wiesz nawet, ile kategorii istnieje.

Modelowanie tematów odpowiada na to bez nadzoru. Daj mu korpus, a otrzymasz mały zestaw spójnych tematów i, dla każdego dokumentu, rozkład na te tematy.

Dwie rodziny algorytmów dominują. LDA (2003) traktuje każdy dokument jako mieszankę ukrytych tematów, a każdy temat jako rozkład na słowach. Inferencja jest bayesowska. Wciąż jest używane w produkcji tam, gdzie potrzebne są przypisania tematów o mieszanym członkostwie i wyjaśnialne rozkłady prawdopodobieństwa na poziomie słów.

BERTopic (2020) koduje dokumenty za pomocą BERT, redukuje wymiarowość za pomocą UMAP, grupuje za pomocą HDBSCAN i wyodrębnia słowa tematów za pomocą TF-IDF opartego na klasach. Wygrywa na krótkich tekstach, mediach społecznościowych i wszystkim, gdzie podobieństwo semantyczne ma większe znaczenie niż nakładanie się słów. Jeden dokument dostaje jeden temat, co jest ograniczeniem w przypadku długich treści.

Ta lekcja buduje intuicję dla obu i nazywa, który wybrać dla danego korpusu.

## Koncepcja

![Model mieszanki LDA vs grupowanie BERTopic](../assets/topic-modeling.svg)

**Historia generatywna LDA.** Każdy temat to rozkład na słowach. Każdy dokument to mieszanka tematów. Aby wygenerować słowo w dokumencie, pobierz temat z mieszanki dokumentu, a następnie pobierz słowo z rozkładu tego tematu. Inferencja odwraca to: mając zaobserwowane słowa, wywnioskuj rozkład tematów na dokument i rozkład słów na temat. Collapsed Gibbs sampling lub variational Bayes wykonuje matematykę.

Kluczowe wyniki LDA:

- `doc_topic`: macierz `(n_docs, n_topics)`, każdy wiersz sumuje się do 1 (mieszanka tematów dokumentu).
- `topic_word`: macierz `(n_topics, vocab_size)`, każdy wiersz sumuje się do 1 (rozkład słów tematu).

**Potok BERTopic.**

1. Zakoduj każdy dokument za pomocą sentence transformera (np. `all-MiniLM-L6-v2`). Wektory 384-wymiarowe.
2. Zredukuj wymiarowość za pomocą UMAP do ~5 wymiarów. Osadzenia BERT są zbyt wysokowymiarowe do grupowania.
3. Grupuj za pomocą HDBSCAN. Oparte na gęstości, tworzy klastry o zmiennej wielkości i etykietę "outlier".
4. Dla każdego klastra oblicz TF-IDF oparte na klasach na dokumentach klastra, aby wyodrębnić najważniejsze słowa.

Wynikiem jest jeden temat na dokument (plus etykieta outlira -1). Opcjonalnie, miękkie członkostwo za pomocą wektora prawdopodobieństwa HDBSCAN.

## Zbuduj To

### Krok 1: LDA przez scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

Zauważ: usunięte stopwords, min_df i max_df filtrują rzadkie i wszechobecne terminy, CountVectorizer (nie TfidfVectorizer), ponieważ LDA oczekuje surowych zliczeń.

### Krok 2: BERTopic (produkcja)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

Filtr `Topic != -1` odrzuca wiadro outlierów BERTopic (dokumenty, których HDBSCAN nie mógł zgrupować). `min_topic_size` kontroluje minimalny rozmiar klastra HDBSCAN; domyślna wartość biblioteki BERTopic to 10. Ten przykład ustawia ją na 15 jawnie dla skali lekcji. Dla korpusów powyżej 10 000 dokumentów zwiększ do 50 lub 100.

### Krok 3: ewaluacja

Obie metody zwracają słowa tematów. Pytanie brzmi, czy te słowa są spójne.

- **Spójność tematów (c_v).** Łączy NPMI (znormalizowaną wzajemną informację punktową) par najważniejszych słów w kontekstach przesuwnego okna, agreguje wyniki w wektory tematów i porównuje te wektory za pomocą podobieństwa cosinusowego. Wyższe jest lepsze. Użyj `gensim.models.CoherenceModel` z `coherence="c_v"`.
- **Różnorodność tematów.** Frakcja unikalnych słów we wszystkich najważniejszych słowach tematów. Wyższe jest lepsze (tematy nie nakładają się).
- **Inspekcja jakościowa.** Przeczytaj najważniejsze słowa każdego tematu. Czy nazywają one prawdziwą rzecz? Ludzki osąd jest wciąż ostatnią linią obrony.

## Kiedy wybrać które

| Sytuacja | Wybierz |
|-----------|------|
| Krótki tekst (tweety, recenzje, nagłówki) | BERTopic |
| Długie dokumenty z mieszankami tematów | LDA |
| Brak GPU / ograniczone zasoby obliczeniowe | LDA lub NMF |
| Potrzeba rozkładów wielotematycznych na poziomie dokumentu | LDA |
| Integracja LLM do etykietowania tematów | BERTopic (bezpośrednie wsparcie) |
| Wdrożenie na urządzeniach brzegowych z ograniczonymi zasobami | LDA |
| Maksymalna spójność semantyczna | BERTopic |

Największym praktycznym czynnikiem jest długość dokumentu. Osadzenia BERT obcinają; zliczenia LDA działają na dowolnej długości. Dla dokumentów dłuższych niż kontekst modelu osadzania, albo chunkuj + agreguj, albo użyj LDA.

## Użyj Tego

Stos w 2026 roku:

- **BERTopic.** Domyślny wybór dla krótkich tekstów i wszystkiego, gdzie semantyka ma znaczenie.
- **`gensim.models.LdaModel`.** Klasyczne LDA dla produkcji, dojrzałe, sprawdzone w boju.
- **`sklearn.decomposition.LatentDirichletAllocation`.** Łatwe LDA do eksperymentów.
- **NMF.** Nieujemna faktoryzacja macierzy. Szybka alternatywa dla LDA, porównywalna jakość na krótkich tekstach.
- **Top2Vec.** Podobny projekt do BERTopic. Mniejsza społeczność, ale dobra na niektórych benchmarkach.
- **FASTopic.** Nowszy, szybszy niż BERTopic na bardzo dużych korpusach.
- **Etykietowanie oparte na LLM.** Uruchom dowolne grupowanie, a następnie zasil model, aby nazwał każdy klaster.

## Wdróż To

Zapisz jako `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## Ćwiczenia

1. **Łatwe.** Dopasuj LDA z 5 tematami na zbiorze danych 20 Newsgroups. Wydrukuj 10 najważniejszych słów na temat. Oznacz każdy temat ręcznie. Czy algorytm znalazł prawdziwe kategorie?
2. **Średnie.** Dopasuj BERTopic na tym samym podzbiorze 20 Newsgroups. Porównaj liczbę znalezionych tematów, najważniejsze słowa i jakościową spójność z LDA. Który czyściej wydobywa prawdziwe kategorie?
3. **Trudne.** Oblicz spójność c_v dla LDA i BERTopic na swoim korpusie. Uruchom każdy z 5, 10, 20, 50 tematami. Narysuj wykres spójności vs liczba tematów. Podaj, która metoda jest bardziej stabilna w różnych liczbach tematów.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Temat | Rzecz, o której jest korpus | Rozkład prawdopodobieństwa na słowach (LDA) lub klaster podobnych dokumentów (BERTopic). |
| Mieszane członkostwo | Dokument to wiele tematów | LDA przypisuje każdemu dokumentowi rozkład na wszystkie tematy. |
| UMAP | Redukcja wymiarowości | Uczenie rozmaitości zachowujące lokalną strukturę; używane w BERTopic. |
| HDBSCAN | Grupowanie gęstościowe | Znajduje klastry o zmiennej wielkości; tworzy etykietę "szum" (-1) dla outlierów. |
| Spójność c_v | Metryka jakości tematów | Średnia wzajemna informacja punktowa najważniejszych słów tematów w przesuwnych oknach. |

## Dalsza Lektura

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — artykuł LDA.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — artykuł BERTopic.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — artykuł, który wprowadził c_v i pokrewne.
- [Dokumentacja BERTopic](https://maartengr.github.io/BERTopic/) — referencja produkcyjna. Doskonałe przykłady.