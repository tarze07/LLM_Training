# Tłumaczenie Maszynowe

> Tłumaczenie to zadanie, które płaciło za badania NLP przez trzydzieści lat i wciąż płaci.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## Problem

Model czyta zdanie w jednym języku i produkuje zdanie w innym. Długość się zmienia. Szyk wyrazów się zmienia. Niektóre słowa źródłowe mapują się na wiele słów docelowych i odwrotnie. Idiomy opierają się mapowaniu jeden-do-jednego. "I miss you" po francusku to "tu me manques" — dosłownie "brakuje mnie tobie". Żadne dopasowanie na poziomie słów tego nie przetrwa.

Tłumaczenie maszynowe to zadanie, które zmusiło NLP do wynalezienia enkoder-dekoderów, attention, transformerów i ostatecznie całego paradygmatu LLM. Każdy krok naprzód nastąpił, ponieważ jakość tłumaczenia była mierzalna, a przepaść między człowiekiem a maszyną była uparta.

Ta lekcja pomija lekcję historii i uczy działającego potoku roku 2026: wstępnie wytrenowany wielojęzyczny enkoder-dekoder (NLLB-200 lub mBART), tokenizacja podsłowowa, beam search, ewaluacja BLEU i chrF oraz garść trybów awarii, które wciąż trafiają do produkcji niewykryte.

## Koncepcja

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

Nowoczesne MT to enkoder-dekoder oparty na transformerze, wytrenowany na równoległym tekście. Enkoder czyta źródło w tokenizacji swojego języka. Dekoder generuje cel, jeden podtoken na raz, używając wyjścia enkodera przez cross-attention (lekcja 10). Dekodowanie używa beam search, aby uniknąć pułapki zachłannego dekodowania. Wyjście jest detokenizowane, usuwane z wielkich liter i punktowane względem referencji.

Trzy wybory operacyjne napędzają jakość MT w świecie rzeczywistym.

- **Tokenizator.** SentencePiece BPE wytrenowany na wielojęzycznym korpusie. Współdzielony słownik między językami jest tym, co umożliwia pary zero-shot w NLLB.
- **Rozmiar modelu.** NLLB-200 distilled 600M mieści się na laptopie. NLLB-200 3.3B jest opublikowanym domyślnym dla produkcji. 54.5B to sufit badawczy.
- **Dekodowanie.** Szerokość wiązki 4-5 dla ogólnych treści. Kara za długość, aby uniknąć zbyt krótkiego wyjścia. Ograniczone dekodowanie, gdy potrzebujesz spójności terminologii.

```figure
seq2seq-alignment
```

## Zbuduj to

### Krok 1: wywołanie wstępnie wytrenowanego MT

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

Trzy rzeczy mają znaczenie. `src_lang` mówi tokenizatorowi, którego skryptu i segmentacji użyć. `forced_bos_token_id` mówi dekoderowi, który język generować. Oba to sztuczki specyficzne dla NLLB; mBART i M2M-100 używają własnych konwencji i nie są wymienne.

### Krok 2: BLEU i chrF

BLEU mierzy nakładanie się n-gramów między wyjściem a referencją. Cztery rozmiary n-gramów referencyjnych (1-4), średnia geometryczna precyzji, kara za zbyt krótki output. Wynik w [0, 100]. Powszechnie używane. Frustrujące w interpretacji: 30 BLEU to "używalne"; 40 to "dobre"; 50 to "wyjątkowe"; różnice poniżej 1 BLEU to szum.

chrF mierzy F-score na poziomie znaku. Bardziej czuły dla języków bogatych morfologicznie, gdzie BLEU niedolicza dopasowań. Często raportowany obok BLEU.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Zawsze używaj `sacrebleu`. Normalizuje tokenizację, więc wyniki są porównywalne między artykułami. Implementowanie własnego BLEU to sposób na wprowadzające w błąd benchmarki.

### Trójpoziomowa hierarchia ewaluacji (2026)

Nowoczesna ewaluacja MT używa trzech komplementarnych rodzin metryk. Dostarczaj z co najmniej dwiema.

- **Heurystyczne** (BLEU, chrF). Szybkie, oparte na referencji, interpretowalne, niewrażliwe na parafrazę. Używaj do porównań z legacy i wykrywania regresji.
- **Uczone** (COMET, BLEURT, BERTScore). Modele neuronowe trenowane na ludzkich ocenach; porównują podobieństwo semantyczne tłumaczenia do źródła i referencji. COMET ma najwyższą korelację z badaniami MT od 2023 i jest domyślnym dla produkcji w 2026, gdy jakość ma znaczenie.
- **LLM-jako-sędzia** (bez referencji). Zachęć duży model do oceny tłumaczeń pod kątem płynności, adekwatności, tonu, odpowiedniości kulturowej. GPT-4-jako-sędzia osiąga zgodność z człowiekiem ~80% czasu, gdy rubryka jest dobrze zaprojektowana. Używaj do otwartych treści, dla których nie istnieje referencja.

Praktyczny stos na 2026: `sacrebleu` dla BLEU i chrF, `unbabel-comet` dla COMET i zachęcony LLM dla końcowego sygnału skierowanego do człowieka. Kalibruj każdą metrykę na 50-100 ręcznie oznakowanych przykładach, zanim jej zaufasz na danych produkcyjnych.

Metryki bez referencji (COMET-QE, BLEURT-QE, LLM-jako-sędzia) pozwalają oceniać tłumaczenia bez referencji, co ma znaczenie dla par językowych o długim ogonie, dla których referencyjne tłumaczenia nie istnieją.

### Krok 3: co psuje się w produkcji

Działający potok powyżej przetłumaczy płynnie 80% czasu i po cichu zawiedzie w pozostałych 20%. Nazwane tryby awarii:

- **Halucynacja.** Model wymyśla treści, których nie było w źródle. Częste przy nieznanym słownictwie dziedzinowym. Objaw: output jest płynny, ale twierdzi fakty, których źródło nie podało. Łagodzenie: ograniczone dekodowanie na terminach dziedzinowych, przegląd ludzki w regulowanych treściach, monitorowanie outputu znacznie dłuższego niż input.
- **Generowanie poza docelowym językiem.** Model tłumaczy na zły język. NLLB jest zaskakująco podatny na to w rzadkich parach językowych. Łagodzenie: zweryfikuj `forced_bos_token_id` i zawsze dekoduj z modelem identyfikacji języka na wyjściu.
- **Dryf terminologii.** "Sign up" staje się "s'inscrire" w dokumencie 1 i "créer un compte" w dokumencie 2. Dla tekstu UI i stringów skierowanych do użytkownika spójność ma znaczenie bardziej niż surowa jakość. Łagodzenie: dekodowanie ograniczone glosariuszem lub słownik po edycji.
- **Niedopasowanie formalności.** Francuskie "tu" vs "vous", poziomy grzeczności w japońskim. Model wybiera formę, która była częstsza w treningu. Dla treści skierowanych do klienta jest to zwykle błędne. Łagodzenie: prefiks promptu z tokenem formalności, jeśli model go obsługuje, lub dostrój mały model na korpusie tylko formalnym.
- **Eksplozja długości przy krótkim wejściu.** Bardzo krótkie zdania wejściowe często produkują zbyt długie tłumaczenia, ponieważ kara za długość spada z klifu poniżej ~5 tokenów źródłowych. Łagodzenie: twardy maksymalny limit długości proporcjonalny do długości źródła.

### Krok 4: dostrajanie dla domeny

Wstępnie wytrenowane modele są generalistami. Tłumaczenie prawne, medyczne lub dialogów gier zyskuje mierzalnie dzięki dostrojeniu na dziedzinowych danych równoległych. Przepis nie jest egzotyczny:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

Kilka tysięcy wysokiej jakości równoległych przykładów bije kilkaset tysięcy zaszumionych ze skrobania sieci. Jakość danych treningowych to pojedyncza największa dźwignia produkcyjna.

## Użyj tego

Stos produkcyjny MT na 2026:

| Zastosowanie | Zalecany punkt startowy |
|---------|---------------------------|
| Dowolny-na-dowolny, 200 języków | `facebook/nllb-200-distilled-600M` (laptop) lub `nllb-200-3.3B` (produkcja) |
| Angielskocentryczne, wysoka jakość, 50 języków | `facebook/mbart-large-50-many-to-many-mmt` |
| Krótkie przebiegi, tania inferencja, angielski-francuski/niemiecki/hiszpański | Modele Helsinki-NLP / Marian |
| Strona przeglądarki krytyczna dla opóźnień | ONNX-skwantowany Marian (~50 MB) |
| Maksymalna jakość, gotowość do płacenia | GPT-4 / Claude / Gemini z promptami tłumaczeniowymi |

LLM przewyższają teraz wyspecjalizowane modele MT w kilku parach językowych od 2026, szczególnie w idiomatycznych treściach i długim kontekście. Kompromisem jest koszt na token i opóźnienie. Wybierz LLM, gdy długość kontekstu, spójność stylistyczna lub adaptacja domeny przez promptowanie ma znaczenie bardziej niż przepustowość.

## Dostarcz to

Zapisz jako `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Ćwiczenia

1. **Łatwe.** Przetłumacz 5-zdaniowy angielski akapit na francuski i z powrotem na angielski używając `nllb-200-distilled-600M`. Zmierz, jak blisko tłumaczenie w obie strony jest do oryginału. Powinieneś zobaczyć zachowanie semantyki z dryfem w doborze słów.
2. **Średnie.** Zaimplementuj sprawdzanie identyfikacji języka na wyjściach tłumaczenia używając `fasttext lid.176` lub `langdetect`. Zintegruj z wywołaniem MT, aby generacje poza docelowym językiem zostały wyłapane przed zwróceniem.
3. **Trudne.** Dostrój `nllb-200-distilled-600M` na wybranym przez siebie dziedzinowym korpusie 5,000 par. Zmierz BLEU na wstrzymanym zestawie przed i po dostrojeniu. Raportuj, które rodzaje zdań się poprawiły, a które cofnęły.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| BLEU | Wynik tłumaczenia | Precyzja n-gramów z karą za zwięzłość. [0, 100]. |
| chrF | F-score znakowy | F-score na poziomie znaku. Bardziej czuły dla języków bogatych morfologicznie. |
| NMT | Neuronowe MT | Enkoder-dekoder transformer trenowany na równoległym tekście. Domyślny od 2017+. |
| NLLB | No Language Left Behind | Rodzina modeli MT Meta na 200 języków. |
| Ograniczone dekodowanie | Kontrolowany output | Wymuś pojawienie się / niepojawienie się konkretnych tokenów lub n-gramów w wyjściu. |
| Halucynacja | Wymyślone treści | Output modelu, który nie jest poparty przez źródło. |

## Dalsze czytanie

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — artykuł NLLB.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — dlaczego `sacrebleu` jest jedynym poprawnym sposobem raportowania BLEU.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — artykuł o chrF.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — praktyczny przewodnik po dostrajaniu.