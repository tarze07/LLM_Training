# T5, BART — Modele Enkoder-Dekoder

> Enkodery rozumieją. Dekodery generują. Złóż je z powrotem, a otrzymasz model stworzony do zadań wejście → wyjście: tłumacz, streszczaj, przepisuj, transkrybuj.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## Problem

GPT tylko z dekoderem i BERT tylko z enkoderem każdy okrawa architekturę z 2017 roku dla innego celu. Ale wiele zadań jest naturalnie wejściowo-wyjściowych:

- Tłumaczenie: angielski → francuski.
- Streszczanie: artykuł 5000 tokenów → streszczenie 200 tokenów.
- Rozpoznawanie mowy: tokeny audio → tokeny tekstu.
- Ekstrakcja strukturalna: proza → JSON.

Dla nich enkoder-dekoder stanowi najczystsze dopasowanie. Enkoder produkuje gęstą reprezentację źródła. Dekoder generuje wyjście, stosując cross-attention do tej reprezentacji na każdym kroku. Trenowanie to przesunięcie o jeden po stronie wyjścia. Ta sama strata co GPT, tylko uwarunkowana wyjściem enkodera.

Dwa artykuły zdefiniowały nowoczesny podręcznik postępowania:

1. **T5** (Raffel i in. 2019). "Text-to-Text Transfer Transformer." Każde zadanie NLP przeformułowane jako tekst na wejściu, tekst na wyjściu. Pojedyncza architektura, pojedynczy słownik, pojedyncza strata. Pretrenowany na przewidywaniu maskowanych fragmentów (span prediction) — zniekształć fragmenty wejścia, dekoduj je na wyjściu.
2. **BART** (Lewis i in. 2019). "Bidirectional and Auto-Regressive Transformer." Autoenkoder usuwający szum: zniekształcaj wejście na wiele sposobów (miksuj, maskuj, usuwaj, obracaj), poproś dekoder o rekonstrukcję oryginału.

W 2026 roku format enkoder-dekoder żyje dalej tam, gdzie struktura wejścia ma znaczenie:

- Whisper (mowa → tekst).
- Stos tłumaczeniowy Google'a.
- Niektóre modele uzupełniania/naprawiania kodu, które mają odrębne struktury kontekstu i edycji.
- Flan-T5 i warianty do strukturalnych zadań rozumowania.

Tylko dekoder zdobył reflektor, ale enkoder-dekoder nigdy nie zniknął.

## Koncepcja

![Enkoder-dekoder z cross-attention](../assets/encoder-decoder.svg)

### Pętla w przód

```
tokeny źródła ─▶ enkoder ─▶ (N_src, d_model)  ──┐
                                                   │
tokeny celu ─▶ blok dekodera                       │
               ├─▶ maskowany self-attention         │
               ├─▶ cross-attention ◀───────────────┘
               └─▶ FFN
              ↓
            logity następnego tokenu
```

Kluczowo: enkoder uruchamia się raz na wejście. Dekoder działa autoregresyjnie, ale stosuje cross-attention do *tego samego* wyjścia enkodera na każdym kroku. Buforowanie wyjścia enkodera to darmowe przyspieszenie dla długich wejść.

### Pretrenowanie T5 — zniekształcanie fragmentów (span corruption)

Wybierz losowe fragmenty wejścia (średnia długość 3 tokeny, 15% całości). Zastąp każdy fragment unikalnym znacznikiem wartowniczym: `<extra_id_0>`, `<extra_id_1>`, itd. Dekoder wyprowadza tylko zniekształcone fragmenty z ich prefiksem wartowniczym:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

Tańszy sygnał niż przewidywanie całej sekwencji. Konkurencyjne względem MLM (BERT) i prefix-LM (UniLM) w ablacjach artykułu T5.

### Pretrenowanie BART — usuwanie szumu wieloma metodami

BART wypróbowuje pięć funkcji zakłócających:

1. Maskowanie tokenów.
2. Usuwanie tokenów.
3. Wypełnianie tekstu (zamaskuj fragment, dekoder wstawia odpowiednią długość).
4. Permutacja zdań.
5. Rotacja dokumentu.

Połączenie wypełniania tekstu + permutacji zdań dało najlepsze wyniki w zadaniach docelowych. Dekoder zawsze rekonstruuje oryginał. Wyjście BARTa to pełna sekwencja, nie tylko zniekształcone fragmenty — więc pretrenowanie wymaga więcej obliczeń niż T5.

### Wnioskowanie

To samo autoregresyjne generowanie co w GPT. Stosuje się próbkowanie zachłanne / wiązkowe / top-p. Wyszukiwanie wiązkowe (szerokość 4–5) jest standardem dla tłumaczenia i streszczania, ponieważ rozkład wyjścia jest węższy niż w czacie.

### Kiedy wybrać który wariant w 2026 roku

| Zadanie | Enkoder-dekoder? | Dlaczego |
|---------|------------------|----------|
| Tłumaczenie | Tak, zwykle | Wyraźna sekwencja źródłowa; stały rozkład wyjścia; wyszukiwanie wiązkowe działa |
| Mowa na tekst | Tak (Whisper) | Modalność wejścia różni się od wyjścia; enkoder kształtuje cechy audio |
| Czat / rozumowanie | Nie, tylko dekoder | Brak trwałego "wejścia" — rozmowa jest sekwencją |
| Uzupełnianie kodu | Zwykle nie | Tylko dekoder z długim kontekstem wygrywa; modele kodowe jak Qwen 2.5 Coder są tylko z dekoderem |
| Streszczanie | Oba działają | BART, PEGASUS biły wcześniejsze bazowe modele tylko z dekoderem; nowoczesne LLMy tylko z dekoderem dorównują im |
| Ekstrakcja strukturalna | Oba | T5 jest czysty, ponieważ "tekst → tekst" absorbuje dowolny format wyjścia |

Trend od ~2022: tylko dekoder przejmuje zadania, które kiedyś należały do enkoder-dekoder, ponieważ (a) instrukcyjnie dostrojone modele tylko z dekoderem uogólniają się do wszystkiego poprzez podpowiadanie, (b) jedna architektura skaluje się łatwiej niż dwie, (c) RLHF zakłada dekoder. Enkoder-dekoder utrzymuje się tam, gdzie modalność wejścia się różni (mowa, obrazy) lub gdzie jakość wyszukiwania wiązkowego ma znaczenie.

## Zbuduj to

Zobacz `code/main.py`. Implementujemy zniekształcanie fragmentów w stylu T5 dla zabawkowego korpusu — najbardziej użyteczny pojedynczy element tej lekcji, ponieważ pojawia się w każdym przepisie na pretrenowanie enkoder-dekoder od tamtego czasu.

### Krok 1: zniekształcanie fragmentów

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Wybierz fragmenty sumujące się do ~mask_rate tokenów. Zwróć (zniekształcone_wejście, cel)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Format celu to konwencja T5: `<sent0> fragment0 <sent1> fragment1 ...`. Zniekształcone wejście przeplata niezmienione tokeny z tokenami wartowniczymi w miejscach fragmentów.

### Krok 2: zweryfikuj rundę tam i z powrotem

Mając zniekształcone wejście i cel, zrekonstruuj oryginalne zdanie. Jeśli twoje zniekształcanie jest odwracalne, przejście w przód jest dobrze zdefiniowane. To sprawdzenie poprawności — prawdziwe trenowanie nigdy tego nie robi, ale test jest tani i łapie błędy off-by-one w księgowaniu fragmentów.

### Krok 3: zakłócanie w stylu BART

Pięć funkcji: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. Połącz dwie z nich i pokaż wynik.

## Użyj tego

Referencja HuggingFace:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

Sztuczka T5: nazwa zadania trafia do tekstu wejściowego. Ten sam model obsługuje dziesiątki zadań, ponieważ każde zadanie to tekst na wejściu, tekst na wyjściu. W 2026 roku ten wzorzec został uogólniony przez instrukcyjnie dostrojone modele tylko z dekoderem, ale T5 skodyfikował go pierwszy.

## Dostarcz to

Zobacz `outputs/skill-seq2seq-picker.md`. Umiejętność (skill) wybiera między enkoder-dekoder a tylko dekoder dla nowego zadania, biorąc pod uwagę strukturę wejścia-wyjścia, opóźnienie i cele jakościowe.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`, zastosuj zniekształcanie fragmentów do 30-tokenowego zdania, zweryfikuj, że połączenie nie-wartowniczych tokenów źródła z zdekodowanymi fragmentami celu odtwarza oryginał.
2. **Średnie.** Zaimplementuj szum `text_infill` BART: zastąp losowe fragmenty pojedynczym tokenem `<mask>`, a dekoder musi wywnioskować odpowiednią długość fragmentu plus jego zawartość. Pokaż jeden przykład.
3. **Trudne.** Dostrój `flan-t5-small` na małym korpusie angielski → pig-Latin (200 par). Zmierz BLEU na wstrzymanym zbiorze 50 par. Porównaj z dostrajaniem `Llama-3.2-1B` na tych samych danych przy tych samych zasobach obliczeniowych.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co naprawdę oznacza |
|--------|-----------------|---------------------|
| Enkoder-dekoder | "Transformer sekwencja-do-sekwencji (seq2seq)" | Dwa stosy: dwukierunkowy enkoder dla wejścia, przyczynowy dekoder z cross-attention dla wyjścia. |
| Cross-attention | "Gdzie źródło rozmawia z celem" | Q dekodera × K/V enkodera. Jedyne miejsce, gdzie informacje z enkodera wchodzą do dekodera. |
| Zniekształcanie fragmentów | "Sztuczka pretrenowania T5" | Zastąp losowe fragmenty tokenami wartowniczymi; dekoder wyprowadza fragmenty. |
| Cel usuwania szumów | "Gra BART" | Zastosuj funkcję zakłócającą do wejścia, trenuj dekoder, aby zrekonstruował czystą sekwencję. |
| Token wartowniczy (sentinel) | "Symbol zastępczy `<extra_id_N>`" | Specjalne tokeny oznaczające zniekształcone fragmenty w źródle i oznaczające je ponownie w celu. |
| Flan | "Instrukcyjnie dostrojony T5" | T5 dostrojony na >1800 zadań; uczynił enkoder-dekoder konkurencyjnym w podążaniu za instrukcjami. |
| Wyszukiwanie wiązkowe (beam search) | "Strategia dekodowania" | Utrzymuj top-k częściowych sekwencji na każdym kroku; standard dla tłumaczenia/streszczania. |
| Teacher forcing | "Wejście w czasie trenowania" | Podczas trenowania podawaj dekoderowi prawdziwy poprzedni token wyjścia, a nie próbkowany. |

## Dalsze czytanie

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, kanoniczny enkoder-dekoder 2026.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — implementacja referencyjna.