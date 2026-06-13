# Modele Sekwencja-do-Sekwencji

> Dwa RNN udające tłumacza. Wąskie gardło, które napotkały, jest powodem, dla którego istnieje attention.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 minutes

## Problem

Klasyfikacja mapuje sekwencję o zmiennej długości na pojedynczą etykietę. Tłumaczenie mapuje sekwencję o zmiennej długości na inną sekwencję o zmiennej długości. Wejście i wyjście żyją w różnych słownikach, prawdopodobnie różnych językach, bez gwarancji równości długości.

Architektura seq2seq (Sutskever, Vinyals, Le, 2014) rozwiązała to celowo prostym przepisem. Dwa RNN. Jeden czyta zdanie źródłowe i produkuje wektor kontekstu o stałym rozmiarze. Drugi czyta ten wektor i generuje zdanie docelowe token po tokenie. Ten sam kod, który napisałeś dla lekcji 08, połączony inaczej.

Warto to studiować z dwóch powodów. Po pierwsze, wąskie gardło wektora kontekstu jest najbardziej pedagogicznie użyteczną porażką w NLP. Motywuje wszystko, w czym attention i transformery są dobre. Po drugie, przepis treningowy (teacher forcing, scheduled sampling, beam search przy inferencji) wciąż stosuje się do każdego nowoczesnego systemu generacji, włączając LLM.

## Koncepcja

**Enkoder.** RNN, który czyta zdanie źródłowe. Jego końcowy stan ukryty to **wektor kontekstu** — podsumowanie o stałym rozmiarze całego wejścia. Rzekomo nie tracąc niczego poza źródłem.

**Dekoder.** Kolejny RNN zainicjowany z wektora kontekstu. W każdym kroku przyjmuje poprzednio wygenerowany token jako wejście i produkuje rozkład na słowniku docelowym. Próbkuj lub argmax, aby wybrać następny token. Podaj go z powrotem. Powtarzaj, aż token `<EOS>` zostanie wyprodukowany lub zostanie osiągnięta maksymalna długość.

**Trenowanie:** Entropia krzyżowa w każdym kroku dekodera, sumowana po sekwencji. Standardowe wsteczne propagowanie w czasie przez obie sieci.

**Teacher forcing.** Podczas trenowania wejście dekodera w kroku `t` to *prawdziwy* token z pozycji `t-1`, a nie własna poprzednia predykcja dekodera. Stabilizuje to trenowanie; bez tego wczesne błędy kaskadują i model nigdy się nie uczy. Przy inferencji musisz używać własnych predykcji modelu, więc zawsze istnieje luka między trenowaniem a inferencją. Ta luka nazywa się **exposure bias**.

**Wąskie gardło.** Wszystko, czego enkoder nauczył się o źródle, musi zostać wciśnięte w ten jeden wektor kontekstu. Długie zdania tracą szczegóły. Rzadkie słowa są rozmywane. Zmiana kolejności (chat noir vs. black cat) musi być zapamiętana, a nie obliczona.

Attention (lekcja 10) naprawia to, pozwalając dekoderowi patrzeć na *każdy* stan ukryty enkodera, a nie tylko na ostatni. To cała koncepcja.

```figure
lstm-gates
```

## Zbuduj to

### Krok 1: enkoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` ma kształt `[batch, seq_len, hidden_dim]` — jeden stan ukryty na pozycję wejściową. `hidden` ma kształt `[1, batch, hidden_dim]` — ostatni krok. Lekcja 08 mówiła "pooluj po wyjściach dla klasyfikacji". Tutaj zachowujemy ostatni stan ukryty jako wektor kontekstu i ignorujemy wyjścia na krok.

### Krok 2: dekoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Dekoder jest wywoływany jeden krok na raz. Wejście: partia pojedynczych tokenów i bieżący stan ukryty. Wyjście: logity słownika dla następnego tokena i zaktualizowany stan ukryty.

### Krok 3: pętla treningowa z teacher forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Dwa pokrętła warte wymienienia. `ignore_index=0` pomija stratę na tokenach dopełniających. `teacher_forcing_ratio` to prawdopodobieństwo użycia prawdziwego tokena vs. predykcji modelu w każdym kroku. Zacznij od 1.0 (pełny teacher forcing) i wygaszaj do ~0.5 podczas trenowania, aby zamknąć lukę exposure bias.

### Krok 4: pętla inferencji (zachłanna)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Zachłanne dekodowanie wybiera token o najwyższym prawdopodobieństwie w każdym kroku. Może zboczyć: gdy już zaangażujesz się w token, nie możesz go cofnąć. **Beam search** utrzymuje przy życiu top-`k` częściowych sekwencji i wybiera tę z najwyższym wynikiem na końcu. Szerokość wiązki 3-5 jest standardowa.

### Krok 5: wąskie gardło, zademonstrowane

Wytrenuj model na zabawkowym zadaniu kopiowania: źródło `[a, b, c, d, e]`, cel `[a, b, c, d, e]`. Zwiększaj długość sekwencji. Obserwuj dokładność.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

Pojedynczy stan ukryty GRU nie może bezstratnie zapamiętać 40-tokenowego wejścia. Informacja jest tam w każdym kroku enkodera, ale dekoder widzi tylko ostatni stan. Attention naprawia to bezpośrednio.

## Użyj tego

PyTorch ma szablony seq2seq oparte na `nn.Transformer` i `nn.LSTM`. Biblioteka `transformers` Hugging Face dostarcza pełne modele enkoder-dekoder (BART, T5, mBART, NLLB) wytrenowane na miliardach tokenów.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Nowoczesne enkoder-dekodery porzuciły RNN na rzecz transformerów. Ogólny kształt (enkoder, dekoder, generuj token po tokenie) jest identyczny z artykułem seq2seq z 2014. Mechanizm wewnątrz każdego bloku jest inny.

### Kiedy wciąż sięgać po seq2seq oparty na RNN

Prawie nigdy, dla nowych projektów. Konkretne wyjątki:

- Tłumaczenie strumieniowe, gdzie konsumujesz wejście jeden token na raz z ograniczoną pamięcią.
- Generowanie tekstu na urządzeniu, gdzie koszt pamięci transformera jest zaporowy.
- Pedagogika. Zrozumienie wąskiego gardła enkoder-dekoder to najszybsza ścieżka do zrozumienia, dlaczego transformery wygrały.

### Exposure bias i jego łagodzenie

- **Scheduled sampling.** Wygaszaj współczynnik teacher forcing podczas trenowania, aby model uczył się odzyskiwać z własnych błędów.
- **Minimum risk training.** Trenuj na wyniku BLEU na poziomie zdania zamiast entropii krzyżowej na poziomie tokena. Bliższe temu, czego faktycznie chcesz.
- **Reinforcement learning fine-tuning.** Nagradzaj generator sekwencji metryką. Używane w nowoczesnym RLHF dla LLM.

Wszystkie trzy wciąż stosują się do generacji opartej na transformerach.

## Dostarcz to

Zapisz jako `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj zabawkowe zadanie kopiowania. Wytrenuj GRU seq2seq na parach wejście-wyjście, gdzie cel równa się źródło. Zmierz dokładność dla długości 5, 10, 20. Odtwórz wąskie gardło.
2. **Średnie.** Dodaj dekodowanie z beam search o szerokości wiązki 3. Zmierz BLEU na małym równoległym korpusie w porównaniu z zachłannym. Udokumentuj, gdzie beam search wygrywa (zwykle ostatnie tokeny) i gdzie nie ma różnicy.
3. **Trudne.** Dostrój `facebook/bart-base` na zbiorze danych 10k parafraz. Porównaj output z beam-4 dostrojonego modelu z modelem bazowym na wstrzymanych wejściach. Raportuj BLEU i wybierz 10 jakościowych przykładów.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Enkoder | Wejściowy RNN | Czyta źródło. Produkuje stany ukryte na krok i końcowy wektor kontekstu. |
| Dekoder | Wyjściowy RNN | Inicjowany z wektora kontekstu. Generuje tokeny docelowe jeden na raz. |
| Wektor kontekstu | Podsumowanie | Końcowy stan ukryty enkodera. Stały rozmiar. Wąskie gardło, które rozwiązuje attention. |
| Teacher forcing | Użyj prawdziwych tokenów | Podawaj prawdziwy poprzedni token podczas trenowania. Stabilizuje uczenie. |
| Exposure bias | Luka trening/test | Model trenowany na prawdziwych tokenach nigdy nie ćwiczył odzyskiwania z własnych błędów. |
| Beam search | Lepsze dekodowanie | Utrzymuj przy życiu top-k częściowych sekwencji w każdym kroku zamiast zachłannie wybierać. |

## Dalsze czytanie

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — oryginalny artykuł seq2seq. Cztery strony.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — wprowadził GRU i ramy enkoder-dekoder.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — artykuł o attention. Przeczytaj natychmiast po tej lekcji.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — możliwy do zbudowania kod seq2seq + attention.