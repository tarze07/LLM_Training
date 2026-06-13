# CNN i RNN dla Tekstu

> Sploty uczą się n-gramów. Rekurencje pamiętają. Oba są zastąpione przez attention. Oba wciąż mają znaczenie na ograniczonym sprzęcie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 minutes

## Problem

TF-IDF i Word2Vec produkowały płaskie wektory, które ignorowały kolejność słów. Klasyfikator zbudowany na nich nie mógł odróżnić `dog bites man` od `man bites dog`. Kolejność słów czasami niesie sygnał.

Dwie rodziny architektur wypełniły tę lukę, zanim pojawiły się transformery.

**Sieci splotowe dla tekstu (TextCNN).** Zastosuj 1D sploty na sekwencjach osadzeń słów. Filtr szerokości 3 to uczący się detektor trigramów: obejmuje trzy słowa i zwraca wynik. Ułóż różne szerokości (2, 3, 4, 5), aby wykrywać wzorce w wielu skalach. Max-pool do reprezentacji o stałym rozmiarze. Płaskie, równoległe, szybkie.

**Sieci rekurencyjne (RNN, LSTM, GRU).** Przetwarzają tokeny pojedynczo, utrzymując stan ukryty, który przenosi informacje do przodu. Sekwencyjne, pamiętające, elastyczne długości wejścia. Dominowały w modelowaniu sekwencji od 2014 do 2017, potem pojawiło się attention.

Ta lekcja buduje obie, a następnie nazywa porażkę, która zmotywowała attention.

## Koncepcja

**TextCNN** (Kim, 2014). Tokeny są osadzane. Splot 1D o szerokości `k` przesuwa filtr po kolejnych `k`-gramach osadzeń, tworząc mapę cech. Globalne max-poolowanie na tej mapie wybiera najsilniejszą aktywację. Połącz wyniki max-poolowania z kilku szerokości filtrów. Podaj do klasyfikatora.

Dlaczego to działa. Filtr to uczący się n-gram. Max-poolowanie jest niezmienne względem pozycji, więc "not good" uruchamia tę samą cechę na początku lub w środku recenzji. Trzy szerokości filtrów po 100 filtrów każda daje 300 uczących się detektorów n-gramów. Trenowanie jest równoległe; nie ma zależności sekwencyjnej.

**RNN.** W każdym kroku czasowym `t`, stan ukryty `h_t = f(W * x_t + U * h_{t-1} + b)`. Współdziel `W`, `U`, `b` w czasie. Stan ukryty w czasie `T` jest podsumowaniem całego prefiksu. Dla klasyfikacji, agreguj po `h_1 ... h_T` (max, średnia lub ostatni).

Zwykłe RNN cierpią na zanikający gradient. **LSTM** dodaje bramy, które decydują, co zapomnieć, co przechować i co zwrócić, stabilizując gradienty przez długie sekwencje. **GRU** upraszcza LSTM do dwóch bram; działa podobnie z mniejszą liczbą parametrów.

**Dwukierunkowe RNN** uruchamiają jeden RNN do przodu i drugi do tyłu, łącząc stany ukryte. Reprezentacja każdego tokena widzi zarówno lewy, jak i prawy kontekst. Niezbędne do zadań znakowania.

```figure
rnn-unroll
```

## Zbuduj to

### Krok 1: TextCNN w PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` przekształca `[batch, seq_len, embed_dim]` na `[batch, embed_dim, seq_len]`, ponieważ `nn.Conv1d` traktuje środkową oś jako kanały. Spoolowane wyjście ma stały rozmiar niezależnie od długości wejścia.

### Krok 2: klasyfikator LSTM

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pool po sekwencji, a nie poolowanie ostatniego stanu. Dla klasyfikacji max-poolowanie zwykle bije branie ostatniego stanu ukrytego, ponieważ informacje na końcu długiej sekwencji mają tendencję do dominowania ostatniego stanu.

### Krok 3: demonstracja zanikającego gradientu (intuicja)

Zwykły RNN bez bramkowania nie może nauczyć się zależności dalekiego zasięgu. Rozważ zadanie-zabawkę: przewidź, czy token `A` pojawił się gdziekolwiek w sekwencji. Jeśli `A` jest na pozycji 1, a sekwencja ma 100 tokenów, gradient z funkcji straty musi przepłynąć z powrotem przez 99 mnożeń wagi rekurencyjnej. Jeśli waga jest mniejsza niż 1, gradient zanika. Jeśli większa niż 1, eksploduje.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM naprawiają to za pomocą **stanu komórki**, który przepływa przez sieć tylko z addytywnymi interakcjami (brama zapominania skaluje go multiplikatywnie, ale gradienty wciąż płyną wzdłuż "autostrady"). GRU robią coś podobnego z mniejszą liczbą parametrów. Oba zapewniają stabilne trenowanie przez sekwencje 100+ kroków.

### Krok 4: dlaczego to wciąż nie wystarczało

Trzy problemy utrzymywały się nawet z LSTM.

1. **Wąskie gardło sekwencyjne.** Trenowanie RNN na sekwencji długości 1000 wymaga 1000 szeregowych kroków w przód/wstecz. Nie można równoleglić w czasie.
2. **Wektor kontekstu o stałym rozmiarze w konfiguracjach enkoder-dekoder.** Dekoder widzi tylko końcowy stan ukryty enkodera, skompresowany dla całego wejścia. Długie wejścia tracą szczegóły. Lekcja 09 omawia to bezpośrednio.
3. **Sufit dokładności dla odległych zależności.** LSTM przewyższają zwykłe RNN, ale wciąż mają trudności z propagowaniem konkretnych informacji przez 200+ kroków.

Attention rozwiązało wszystkie trzy. Transformery całkowicie porzuciły rekurencję. Lekcja 10 jest punktem zwrotnym.

## Użyj tego

`nn.LSTM`, `nn.GRU` i `nn.Conv1d` z PyTorch są gotowe do produkcji. Kod trenowania jest standardowy.

Hugging Face dostarcza wstępnie wytrenowane osadzenia, które podłączasz jako warstwę wejściową:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Lista kontrolna użycia, gdy pasuje do ograniczeń.

- **Inferencja brzegowa / na urządzeniu.** TextCNN z osadzeniami GloVe jest 10-100x mniejszy niż transformer. Jeśli twoim celem wdrożenia jest telefon, to jest stos.
- **Klasyfikacja strumieniowa / online.** RNN przetwarza jeden token na raz; transformery potrzebują pełnej sekwencji. Dla tekstu przychodzącego w czasie rzeczywistym LSTM wciąż wygrywają.
- **Małe modele dla modeli bazowych.** Szybka iteracja na nowym zadaniu. Wytrenuj TextCNN w 5 minut na CPU.
- **Znakowanie sekwencji z ograniczonymi danymi.** BiLSTM-CRF (lekcja 06) to wciąż architektura NER klasy produkcyjnej dla 1k-10k oznakowanych zdań.

Wszystko inne idzie do transformera.

## Dostarcz to

Zapisz jako `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Ćwiczenia

1. **Łatwe.** Wytrenuj TextCNN na 3-klasowym zbiorze danych-zabawce (sami wymyśl dane). Zweryfikuj, że szerokości filtrów (2, 3, 4) przewyższają pojedynczą szerokość (3) pod względem średniego F1.
2. **Średnie.** Zaimplementuj max-pool, mean-pool i last-state pool dla klasyfikatora LSTM. Porównaj na małym zbiorze danych; udokumentuj, które poolowanie wygrywa i postaw hipotezę dlaczego.
3. **Trudne.** Zbuduj tagger NER BiLSTM-CRF (połącz lekcję 06 i tę). Wytrenuj na CoNLL-2003. Porównaj z modelem bazowym CRF z lekcji 06 i z fine-tunem BERT. Raportuj czas trenowania, pamięć i F1.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| TextCNN | CNN dla tekstu | Stos splotów 1D na osadzeniach słów z globalnym max-pool. Kim (2014). |
| RNN | Sieć rekurencyjna | Stan ukryty aktualizowany w każdym kroku czasowym: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | RNN z bramowaniem | Dodaje bramy wejścia/zapominania/wyjścia + stan komórki. Trenuje stabilnie przez długie sekwencje. |
| GRU | Prostszy LSTM | Dwie bramy zamiast trzech. Podobna dokładność, mniej parametrów. |
| Dwukierunkowość | Oba kierunki | Forward + backward RNN połączone. Każdy token widzi obie strony swojego kontekstu. |
| Zanikający gradient | Sygnał treningowy zanika | Wielokrotne mnożenie przez wagi <1 w zwykłych RNN sprawia, że gradienty wczesnych kroków są efektywnie zerowe. |

## Dalsze czytanie

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — artykuł TextCNN. Osiem stron. Czytelny.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — artykuł LSTM. Zaskakująco przejrzysty.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — diagramy, które uczyniły LSTM dostępnymi dla każdego.