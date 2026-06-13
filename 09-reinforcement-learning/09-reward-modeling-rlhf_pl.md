# Modelowanie Nagrody i RLHF

> Ludzie nie mogą napisać funkcji nagrody za "dobrą odpowiedź asystenta", ale mogą porównać dwie odpowiedzi i wybrać lepszą. Dopasuj model nagrody do tych porównań, a następnie ucz model językowy RL-em przeciwko niemu. Christiano 2017. InstructGPT 2022. Przepis, który zmienił GPT-3 w ChatGPT. W 2026 jest w większości zastępowany przez DPO — ale model mentalny pozostaje.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 minutes

## Problem

Wytrenowałeś model językowy na celu przewidywania następnego tokena. Pisze poprawną gramatycznie angielszczyznę. Ale też kłamie, rozwleka się i nie potrafi odmówić. Nie naprawisz tego większym pretreningiem — tekst internetowy jest problemem, nie lekarstwem.

Chcesz *skalarnej nagrody*, która mówi "odpowiedź A jest lepsza niż odpowiedź B dla instrukcji X." Ręczne napisanie takiej funkcji nagrody jest niemożliwe. "Pomocność" nie jest wyrażeniem w formie zamkniętej na tokenach. Ale ludzie mogą porównać dwa wyniki i zaznaczyć preferencję. To jest tanie do zebrania w skali.

RLHF (Christiano i in. 2017; Ouyang i in. 2022) przekształca preferencje w model nagrody, a następnie optymalizuje LM przez PPO przeciwko tej nagrodzie. W trzech krokach: SFT → RM → PPO. To przepis, który dostarczył ChatGPT, Claude, Gemini i każdy inny dopasowany LLM w latach 2023–2025.

W 2026 krok PPO jest w większości zastąpiony przez DPO (Faza 10 · 08), ponieważ jest tańszy i prawie tak dobry do dostrajania dopasowania. Ale element *modelu nagrody* wciąż leży u podstaw każdego samplera Best-of-N, każdego pipeline'u RL ze sprawdzalnych nagród i każdego modelu rozumowania używającego modelu nagrody procesu. Zrozum RLHF, a zrozumiesz cały stos dopasowania.

## Koncepcja

![Trójstopniowy RLHF: SFT, trening RM na parach preferencji, PPO z karą KL](../assets/rlhf.svg)

**Etap 1: Nadzorowane Dostrajanie (SFT).** Zacznij od wstępnie wytrenowanego modelu bazowego. Dostrój na ludzkich demonstracjach docelowego zachowania (odpowiedzi podążające za instrukcjami, pomocne odpowiedzi, itp.). Wynik: model `π_SFT`, który jest *ukierunkowany na dobre zachowanie*, ale wciąż ma nieograniczoną przestrzeń akcji.

**Etap 2: Trening modelu nagrody (RM).**

- Zbierz pary odpowiedzi `(y_+, y_-)` do promptów `x`, oznaczonych przez ludzi jako "y_+ jest preferowane nad y_-."
- Trenuj model nagrody `R_φ(x, y)`, aby przypisywał wyższe wyniki `y_+`.
- Strata: **parami logiczny Bradleya-Terry'ego**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ to sigmoida. Różnica w nagrodzie implikuje log-szanse preferencji. BT jest standardem od 1952 (Bradley-Terry) i dominującym wyborem w nowoczesnym RLHF.

- `R_φ` jest zwykle inicjowany z modelu SFT ze skalarną głową na wierzchu. Ten sam szkielet transformera; pojedyncza warstwa liniowa wypisuje nagrodę.

**Etap 3: PPO przeciwko RM z karą KL.**

- Zainicjuj trenowalną politykę `π_θ` z `π_SFT`. Utrzymuj zamrożoną *referencję* `π_ref = π_SFT`.
- Nagroda na końcu odpowiedzi `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  Kara KL zapobiega nieograniczonemu odbieganiu `π_θ` od `π_SFT` — jest *regularyzatorem*, nie twardym regionem zaufania. `β` typowo `0.01`-`0.05`.
- Uruchom PPO (Lekcja 08) z tą nagrodą. Przewagi są obliczane na trajektorii na poziomie tokenów, ale RM ocenia tylko pełną odpowiedź.

**Dlaczego KL?** Bez niego, PPO chętnie znajdzie strategie hakowania nagrody — RM był trenowany tylko na kompletacjach z rozkładu. Odpowiedź spoza rozkładu może uzyskać wyższy wynik niż jakakolwiek napisana przez człowieka. KL utrzymuje `π_θ` blisko rozmaitości, na której RM był trenowany. To najważniejsze pokrętło w RLHF.

**Status w 2026:**

- **DPO** (Rafailov 2023): algebra w formie zamkniętej zwija Etap 2+3 w jedną nadzorowaną stratę na danych preferencyjnych. Żaden RM, żadne PPO. Ta sama jakość na benchmarkach dopasowania za ułamek mocy obliczeniowej. Omówione w Fazie 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO z wartością bazową względną grupowo zamiast krytyka, nagroda z *weryfikatora* (kod działa / odpowiedź matematyczna zgadza się) zamiast RM trenowanego przez ludzi. Dominujące dla modeli rozumowania. Omówione w Fazie 9 · 12.
- **Modele nagrody procesu (PRM):** oceniają częściowe rozwiązania (każdy krok rozumowania), używane zarówno w wariantach RLHF, jak i GRPO dla rozumowania.
- **Konstytucyjna SI / RLAIF:** użyj dopasowanego LLM do generowania preferencji zamiast ludzi. Skaluje budżet preferencji.

## Zbuduj To

Ta lekcja używa małych syntetycznych "promptów" i "odpowiedzi" reprezentowanych jako ciągi znaków. RM jest liniowym skorerem nad reprezentacją worka tokenów. Żaden prawdziwy LLM — *kształt* pipeline'u ma znaczenie, nie skala. Zobacz `code/main.py`.

### Krok 1: syntetyczne dane preferencyjne

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

W prawdziwym RLHF jest to zastąpione przez ludzkich adnotatorów. Kształt — `(prompt, preferred_response, rejected_response)` — jest identyczny.

### Krok 2: Model nagrody Bradleya-Terry'ego

Wynik liniowy: `R(x, y) = w · bag(y)`. Trenuj, aby minimalizować parami stratę logistyczną BT:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

Po kilkuset aktualizacjach `w` przypisuje dodatnie wagi tokenom dobrych słów i ujemne złym.

### Krok 3: Polityka podobna do PPO na szczycie RM

Nasza zabawkowa polityka produkuje pojedynczy token ze słownika. Oceniamy token pod RM, obliczamy `log π_θ(token | prompt)`, dodajemy karę KL-do-referencji i stosujemy przycięty cel zastępczy PPO.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Krok 4: monitoruj KL

Śledź średnie `KL(π_θ || π_ref)` przy każdej aktualizacji. Jeśli przekroczy `~5-10`, polityka za bardzo odbiegła od `π_SFT` — `β` rośnie lub zaczyna się hakowanie nagrody. To najważniejsza diagnostyka w prawdziwym RLHF.

### Krok 5: produkcyjny przepis z TRL

Gdy zrozumiesz zabawkowy pipeline, oto ta sama pętla, jak pisze ją prawdziwy użytkownik biblioteki. Hugging Face [TRL](https://huggingface.co/docs/trl) to referencyjna implementacja — `RewardTrainer` dla Etapu 2 i `PPOTrainer` (z wbudowaną karą KL-do-referencji) dla Etapu 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Trzy rzeczy, które biblioteka robi za ciebie. `adap_kl_ctrl=True` implementuje adaptacyjny harmonogram β: jeśli obserwowane KL przekracza `target_kl`, β się podwaja; jeśli poniżej połowy, β się zmniejsza o połowę. Model referencyjny jest zamrożony z założenia — nie możesz przypadkiem współdzielić parametrów z `policy`. A głowa wartości znajduje się na tym samym szkielecie co polityka (`AutoModelForCausalLMWithValueHead` dołącza skalarną głowę MLP), dlatego TRL raportuje `policy/kl` i `value/loss` osobno.

## Pułapki

- **Nadmierna optymalizacja / hakowanie nagrody.** RM jest niedoskonały; `π_θ` znajduje adwersarialne kompletacje, które uzyskują wysoki wynik, ale są złe. Objawy: nagroda rośnie w nieskończoność, podczas gdy ocena ludzka osiąga plateau lub spada. Poprawka: zatrzymaj wcześnie, podnieś `β`, poszerz dane treningowe RM.
- **Hakowanie długości.** RM trenowane na pomocnych odpowiedziach często niejawnie nagradzają długość. Polityka uczy się rozwlekać odpowiedzi. Remedium: nagroda znormalizowana długością lub RLAIF ze świadomym długości RM.
- **Zbyt mały RM.** RM musi być co najmniej tak duży jak polityka. Mały RM nie może wiernie oceniać wyników polityki.
- **Dostrajanie KL.** Zbyt niskie β → dryf i hakowanie nagrody. Zbyt wysokie β → polityka prawie się nie zmienia. Standardową sztuczką jest *adaptacyjne* β, które celuje w stałe KL na krok.
- **Szum w danych preferencyjnych.** ~30% ludzkich etykiet jest zaszumionych lub niejednoznacznych. Kalibruj, trenując RM na danych filtrowanych ze względu na zgodność lub używając temperatury na BT.
- **Problemy poza polityką.** Dane PPO są lekko poza polityką po pierwszej epoce. Monitoruj frakcję przycięcia, jak w Lekcji 08.

## Użyj Tego

RLHF w 2026 jest warstwowy:

| Warstwa | Cel | Metoda |
|-------|--------|--------|
| Podążanie za instrukcjami, pomocność, nieszkodliwość | Dopasowanie | DPO (Faza 10 · 08) preferowane nad RLHF-PPO. |
| Poprawność rozumowania (matematyka, kod) | Możliwość | GRPO z weryfikatorem nagrody (Faza 9 · 12). |
| Długoterminowe zadania wielokrokowe | Agencyjne | PPO / GRPO z modelami nagrody procesu na krokach. |
| Bezpieczeństwo / zachowanie odmowy | Bezpieczeństwo | RLHF-PPO z oddzielnym RM bezpieczeństwa lub Konstytucyjna SI. |
| Best-of-N przy wnioskowaniu | Szybkie dopasowanie | Użyj RM w czasie dekodowania; żaden trening polityki nie jest potrzebny. |
| Dystylacja nagrody | Moc obliczeniowa wnioskowania | Trenuj małą "głowę nagrody" na zamrożonym LM. |

RLHF było *tą* metodą w latach 2022–2024. W 2026, produkcyjne pipeline'y dopasowania są DPO-pierwsze, PPO-tylko dla kroków intensywnie używających RM lub krytycznych dla bezpieczeństwa.

## Dostarcz To

Zapisz jako `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Ćwiczenia

1. **Łatwe.** Trenuj model nagrody Bradleya-Terry'ego w `code/main.py` na 500 syntetycznych parach preferencji. Zmierz dokładność parami na 100 wstrzymanych parach. Powinna przekroczyć 90%.
2. **Średnie.** Uruchom zabawkową pętlę PPO-RLHF z `β ∈ {0.0, 0.1, 1.0}`. Dla każdej, narysuj wynik RM vs KL-do-referencji w czasie aktualizacji. Które uruchomienia hakują nagrodę?
3. **Trudne.** Zaimplementuj DPO (strata preferencyjna w formie zamkniętej) na tych samych danych preferencyjnych i porównaj do pipeline'u RLHF-PPO pod względem użytej mocy obliczeniowej i końcowego osiągniętego wyniku RM.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|-----------------|-----------------------|
| RLHF | "Dopasowanie RL" | Trójstopniowy pipeline SFT + RM + PPO (Christiano 2017, Ouyang 2022). |
| Model Nagrody (RM) | "Sieć oceniająca" | Wyuczona funkcja skalarna dopasowana do preferencji parami przez Bradleya-Terry'ego. |
| Bradley-Terry | "Paramii strata logistyczna" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; standardowy cel RM. |
| Kara KL | "Pozostań blisko referencji" | `β · KL(π_θ \|\| π_ref)` w nagrodzie; regularyzator przeciw hakowaniu nagrody. |
| Hakowanie nagrody | "Prawo Goodharta" | Polityka wykorzystuje wady RM; objawy: nagroda w górę, ocena ludzka płaska. |
| RLAIF | "Preferencje oznaczone przez SI" | RLHF, gdzie etykiety pochodzą z innego LM zamiast ludzi. |
| PRM | "Model Nagrody Procesu" | Ocenia częściowe kroki rozumowania; używane w pipeline'ach rozumowania. |
| Konstytucyjna SI | "Metoda Anthropic" | Preferencje generowane przez SI kierowane przez jawne reguły. |

## Dalsza Lektura

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — praca, która zapoczątkowała RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — przepis stojący za ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — wcześniejszy RLHF do streszczania.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; domyślne po RLHF w 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF i pętla samokrytyki.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — praca HH.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — produkcyjny `RewardTrainer` i `PPOTrainer`. Przeczytaj kod źródłowy trainera, aby poznać szczegóły adaptacyjnego KL i głowy wartości.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) autorstwa Lambert, Castricato, von Werra, Havrilla — kanoniczny przewodnik po trójstopniowym pipeline z diagramami.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — biblioteka; `examples/` zawiera kompleksowe skrypty RLHF dla Llama, Mistral i Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — widok hipotezy nagrody; niezbędny warunek wstępny do myślenia o hakowaniu nagrody.