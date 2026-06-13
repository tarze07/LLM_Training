# Zbuduj Transformer od Podstaw — Projekt Końcowy

> Trzynaście lekcji. Jeden model. Żadnych skrótów.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## Problem

Przeczytałeś każdy artykuł. Zaimplementowałeś attention, podział na wiele głów, kodowanie pozycyjne, bloki enkodera i dekodera, straty BERT i GPT, MoE, cache KV. Teraz spraw, by działały razem na prawdziwym zadaniu.

Projekt końcowy: wytrenuj mały, wyłącznie dekoderowy transformer od początku do końca na zadaniu modelowania języka na poziomie znaków. Czyta Szekspira. Generuje nowego Szekspira. Jest wystarczająco mały, by trenować na laptopie w mniej niż 10 minut. Jest wystarczająco poprawny, by podmiana większego zbioru danych i dłuższego trenowania dała prawdziwy LM.

To jest "nanoGPT" tego kursu. Nie jest oryginalny — tutorial nanoGPT Karpathy'ego z 2023 to referencyjna implementacja, którą każdy student pisze przynajmniej raz. Podnosimy kształt i doposażamy go w to, co przerobiliśmy.

## Koncepcja

![Schemat blokowy transformera od podstaw](../assets/capstone.svg)

Architektura z adnotacjami:

```
tokeny wejściowe (B, N)
   │
   ▼
embedding tokenów + embedding pozycyjny  ◀── Lekcja 04 (opcja RoPE)
   │
   ▼
┌──── blok × L ────────────────────┐
│  RMSNorm                          │  ◀── Lekcja 05
│  MultiHeadAttention (przyczynowa) │  ◀── Lekcja 03 + 07 (maska przyczynowa)
│  residuum                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lekcja 05
│  residuum                         │
└────────────────────────────────── ┘
   │
   ▼
końcowy RMSNorm
   │
   ▼
lm_head (związany z embeddingiem tokenów)
   │
   ▼
logity (B, N, V)
   │
   ▼
przesunięta o jeden cross-entropia       ◀── Lekcja 07
```

### Co dostarczamy

- `GPTConfig` — jedno miejsce do konfiguracji wszystkich hiperparametrów.
- `MultiHeadAttention` — przyczynowa, wsadowa, z opcjonalną ścieżką w stylu Flash (PyTorch `scaled_dot_product_attention`).
- `SwiGLUFFN` — nowoczesna FFN.
- `Block` — pre-norm, attention + FFN owinięte residuum.
- `GPT` — embeddingi, ułożone bloki, głowa LM, generate().
- Pętla treningowa z AdamW, kosinusowym LR, przycinaniem gradientów.
- Tokenizer na poziomie znaków dla tekstu Szekspira.

### Czego nie dostarczamy

- RoPE — zaimplementowane koncepcyjnie w Lekcji 04. Tu używamy uczonych embeddingów pozycyjnych dla prostoty. Ćwiczenia proszą o zamianę na RoPE.
- Cache KV podczas generowania — każdy krok generowania przelicza attention na całym prefiksie. Wolniejsze, ale prostsze. Ćwiczenia proszą o dodanie cache KV.
- Flash Attention — PyTorch 2.0+ automatycznie wybiera, jeśli wejścia pasują; używamy `F.scaled_dot_product_attention`.
- MoE — pojedyncza FFN na blok. Widziałeś MoE w Lekcji 11.

### Docelowe metryki

Na laptopie Mac M2, 4-warstwowy, 4-głowowy, d_model=128 GPT trenowany przez 2000 kroków na `tinyshakespeare.txt`:

- Strata treningowa spada z ~4.2 (losowa) do ~1.5 w około 6 minut.
- Próbkowane wyjście wygląda jak Szekspir: archaiczne słowa, podziały wierszy, nazwy własne jak "ROMEO:" się pojawiają.
- Strata walidacyjna (pozostałe 10% tekstu) podąża blisko za stratą treningową; brak przetrenowania przy tym rozmiarze/budżecie.

## Zbuduj To

Ta lekcja używa PyTorch. Zainstaluj `torch` (wersja CPU jest w porządku). Zobacz `code/main.py`. Skrypt obsługuje:

- Pobieranie `tinyshakespeare.txt` jeśli brakuje (lub czytanie lokalnej kopii).
- Tokenizer znakowy na poziomie bajtów.
- Podział trening/walidacja 90/10.
- Pętla treningowa z bf16 autocast na obsługiwanym sprzęcie.
- Próbkowanie po zakończeniu trenowania.

### Krok 1: dane

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 unikalnych znaków. Malutkie słownictwo. Mieści się w 4-bajtowym vocab_size. Żadnego BPE, żadnego dramatu z tokenizerem.

### Krok 2: model

Zobacz `code/main.py`. Blok jest podręcznikowy z Lekcji 05 — pre-norm, RMSNorm, SwiGLU, przyczynowa MHA. Liczba parametrów dla 4/4/128: ~800K.

### Krok 3: pętla treningowa

Pobierz losową paczkę okien tokenów długości 256. Forward. Przesunięta o jeden cross-entropia. Backward. Krok AdamW. Loguj. Powtarzaj.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Krok 4: próbkowanie

Mając podpowiedź, wielokrotnie wykonuj forward, próbkuj z top-p logitów, dodawaj i kontynuuj. Zatrzymaj się po 500 tokenach.

### Krok 5: przeczytaj wyjście

Po 2000 krokach:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

To nie Szekspir. Ale ma kształt Szekspira. Wyraźne osiągnięcie dla ~800K parametrów i 6 minut na laptopie.

## Użyj Tego

Ten projekt końcowy jest architekturą referencyjną. Trzy rozszerzenia, by przekształcić go w coś prawdziwego:

1. **Wymień tokenizer.** Użyj BPE (np. `tiktoken.get_encoding("cl100k_base")`). Rozmiar słownictwa skacze z 65 do ~50,000. Pojemność modelu musi wzrosnąć, by to skompensować.
2. **Trenuj na większym korpusie.** Użyj `OpenWebText` lub `fineweb-edu` (HuggingFace). 10B tokenów na pojedynczym A100 zajmuje ~24 godziny dla GPT o 125M parametrów.
3. **Dodaj RoPE + cache KV + Flash Attention.** Ćwiczenia poniżej przeprowadzą cię przez każde.

To kończy się jako GPT o 125M parametrów, który generuje płynny angielski. Nie jest to model na granicy możliwości. Ale ta sama ścieżka kodu — tylko większa — jest tym, czego Karpathy, EleutherAI i Allen Institute używają do trenowania badawczych checkpointów w 2026.

## Dostarcz To

Zobacz `outputs/skill-transformer-review.md`. Umiejętność recenzuje implementację transformera od podstaw pod kątem poprawności względem wszystkich 13 poprzednich lekcji.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Sprawdź, czy końcowa strata walidacyjna twojego wytrenowanego modelu jest poniżej 2.0. Zmień `max_steps` z 2000 na 5000 — czy strata walidacyjna nadal się poprawia?
2. **Średnie.** Zastąp uczone embeddingi pozycyjne RoPE. Zastosuj rotację do Q i K wewnątrz `MultiHeadAttention`. Trenuj i sprawdź, czy strata walidacyjna jest co najmniej równie niska.
3. **Średnie.** Zaimplementuj cache KV w pętli próbkowania. Wygeneruj 500 tokenów z i bez cache. Czas ścienny powinien poprawić się 5–20× na laptopie.
4. **Trudne.** Dodaj drugą głowę do modelu, która przewiduje następny-plus-jeden token (MTP — Multi-Token Prediction z DeepSeek-V3). Trenuj łącznie. Czy to pomaga?
5. **Trudne.** Zastąp pojedynczą FFN na blok MoE z 4 ekspertami. Router + top-2 routing. Zobacz, jak zmienia się strata walidacyjna przy dopasowanych aktywnych parametrach.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co faktycznie oznacza |
|--------|-----------------|-----------------------|
| nanoGPT | "Repozytorium tutoriala Karpathy'ego" | Minimalny kod do trenowania wyłącznie dekoderowego transformera, ~300 linii; kanoniczna referencja. |
| tinyshakespeare | "Standardowy korpus zabawkowy" | ~1.1 MB tekstu; każdy tutorial LM na poziomie znaków od 2015 go używa. |
| Związane embeddingi | "Wspólna macierz wejścia/wyjścia" | Waga głowy LM = transpozycja macierzy embeddingów tokenów; oszczędza parametry, poprawia jakość. |
| bf16 autocast | "Sztuczka z precyzją trenowania" | Wykonuj forward/backward w bf16, przechowuj stan optymalizatora w fp32; standard od 2021. |
| Przycinanie gradientów | "Zatrzymuje skoki" | Ogranicz globalną normę gradientu do 1.0; zapobiega eksplozjom trenowania. |
| Kosinusowy harmonogram LR | "Domyślny od 2020" | LR rośnie liniowo (rozgrzewka), a następnie maleje po kosinusie do 10% wartości szczytowej. |
| MFU | "Wykorzystanie FLOPów modelu" | Osiągnięte FLOPy / teoretyczne maksimum; 40% gęsty, 30% MoE to dobry wynik w 2026. |
| Strata walidacyjna | "Strata na wstrzymanych danych" | Cross-entropia na danych, których model nigdy nie widział; detektor przetrenowania. |

## Dalsza lektura

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — klasyczna adnotowana implementacja.