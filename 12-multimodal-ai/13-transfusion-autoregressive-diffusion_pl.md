# Transfusion: Autogresywny Tekst + Dyfuzyjny Obraz w Jednym Transformerze

> Chameleon i Emu3 postawili wszystko na dyskretne tokeny. Działają, ale wąskie gardło kwantyzacji jest widoczne — jakość obrazu osiąga plateau poniżej modeli dyfuzji w przestrzeni ciągłej. Transfusion (Meta, Zhou et al., sierpień 2024) stawia przeciwny zakład: utrzymaj obrazy ciągłe, porzuć VQ-VAE całkowicie i trenuj jeden transformer z dwoma kosztami. Tokeny tekstowe otrzymują przewidywanie następnego tokena. Łatki obrazkowe otrzymują koszt dopasowania przepływu / dyfuzji. Oba cele optymalizują te same wagi. Architektura leżąca u podstaw Stable Diffusion 3 (MMDiT) jest bliskim kuzynem. Ta lekcja czyta tezę Transfusion, buduje zabawkowy trener z dwoma kosztami i śledzi maskę atencji, która pozwala jednemu transformerowi robić obie prace.

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## Learning Objectives

- Podłącz transformer, który uruchamia dwa koszty (NTP na tokenach tekstowych, MSE dyfuzji na łatkach obrazkowych) na jednym szkielecie.
- Wyjaśnij, dlaczego dwukierunkowa atencja na łatkach obrazkowych plus przyczynowa atencja na tokenach tekstowych jest właściwym wyborem maski.
- Porównaj styl Transfusion (obrazy ciągłe, koszt dyfuzji) ze stylem Chameleona (obrazy dyskretne, NTP) pod względem obliczeń, jakości i złożoności kodu.
- Wymień wkład MMDiT: wagi specyficzne dla modalności w każdym bloku, wspólna atencja na strumieniu residuum.

## The Problem

Debata między dyskretnymi a ciągłymi reprezentacjami obrazów jest starsza niż LLM. Reprezentacje ciągłe (surowe piksele, latenty VAE) zachowują szczegóły. Dyskretne tokeny (indeksy VQ) pasują do rodzimego słownika transformera, ale tracą szczegóły na etapie kwantyzacji.

Chameleon / Emu3 poszli w dyskretność: jeden koszt, jedna architektura, ale wierność obrazu ograniczona jakością tokenizatora.

Modele dyfuzyjne poszły w ciągłość: wyjątkowa jakość obrazu, ale oddzielny model od LLM-a, złożona inżynieria harmonogramu szumu i brak czystej integracji z generacją tekstu.

Transfusion pyta: czy możemy mieć oba? Utrzymaj obrazy ciągłe, wciąż trenuj jeden model, użyj dwóch kosztów zszytych w jeden gradient.

## The Concept

### Architektura dwóch kosztów

Pojedynczy transformer typu decoder-only przetwarza sekwencję zawierającą:

- Tokeny tekstowe (dyskretne, ze słownika BPE).
- Łatki obrazkowe (ciągłe, bloki pikseli 16x16 rzutowane do ukrytego wymiaru przez liniowe osadzenie — tak jak wejście enkodera ViT).
- Znaczniki `<image>` i `</image>` oznaczające, gdzie żyją ciągłe łatki.

Przejście w przód uruchamiane raz. Koszt wybiera jedną z dwóch głów na token:

- Dla tokenów tekstowych: standardowa entropia krzyżowa na głowie logitów słownictwa.
- Dla łaek obrazkowych: koszt dyfuzji na ciągłych łatkach — przewidź szum, który został dodany do każdej łatki.

Gradient przepływa przez współdzielone ciało transformera. Oba koszty ulepszają współdzielone wagi jednocześnie.

### Maska atencji: przyczynowy tekst + dwukierunkowy obraz

Tokeny tekstowe muszą być przyczynowe — nie możesz pozwolić tokenowi tekstowemu na uwagę do przyszłego tekstu, bo teacher forcing się załamuje. Łatki obrazkowe reprezentują jednak jeden kadr; powinny zwracać uwagę na siebie nawzajem dwukierunkowo w obrębie tego samego bloku obrazu.

Maska:

```
M[i, j] = 1 jeśli:
  (i jest tekstem i j jest tekstem i j <= i)   # przyczynowe dla tekstu
  LUB (i jest obrazem i j jest obrazem i ten_sam_blok_obrazu(i, j))   # dwukierunkowe w obrębie obrazu
  LUB (i jest tekstem i j jest obrazem i j < i_koniec_obrazu)   # tekst skupia się na poprzednich obrazach
  LUB (i jest obrazem i j jest tekstem i j < i_początek_obrazu)   # obraz skupia się na poprzedzającym tekście
```

Zaimplementowane jako blokowo-trójkątna maska podczas treningu i inferencji.

### Koszt dyfuzji wewnątrz transformera

Koszt dyfuzji jest standardowy: dodaj szum do łatki obrazkowej, poproś model o przewidzenie szumu (lub czystej łatki, równoważnie). Wersja Transfusion używa dopasowania przepływu — przewidź pole prędkości od zaszumionego do czystego.

Podczas treningu:
1. Dla każdej łatki obrazkowej x0, wylosuj losowy krok czasowy t.
2. Wylosuj szum ε, oblicz xt = (1-t) * x0 + t * ε (interpolacja liniowa dla dopasowania przepływu).
3. Transformer przewiduje v_theta(xt, t); koszt = MSE(v_theta(xt, t), ε - x0).
4. Wsteczna propagacja wraz z kosztami NTP tekstu z tej samej sekwencji.

Na inferencji, generacja to:
- Tokeny tekstowe: standardowe autogresywne samplowanie.
- Łatki obrazkowe: pętla samplowania dyfuzji (typowe 10-30 kroków) warunkowana na poprzedzających tokenach tekstowych.

### MMDiT: wariant Stable Diffusion 3

Stable Diffusion 3 (Esser et al., marzec 2024) dostarczył MMDiT (Multimodal Diffusion Transformer) mniej więcej w tym samym czasie co Transfusion. Architektury są rodzeństwem.

Kluczowe różnice MMDiT:

- Wagi specyficzne dla modalności na blok. Każdy blok transformera ma osobne wagi Q, K, V i MLP dla tokenów tekstowych vs łaek obrazkowych. Atencja jest wspólna (między-modalna); wszystko inne jest specyficzne dla modalności.
- Trening ze sprecyzowanym przepływem. Specyficzny wariant dopasowania przepływu ze znanym samplowaniem i prostszą matematyką niż DDPM.
- Skala. MMDiT jest szkieletem dla SD3 (warianty 2B i 8B parametrów). Artykuł Transfusion skaluje się do 7B.

Oba zbiegają się do tego samego podstawowego pomysłu: jeden transformer uruchamia NTP na tekście i dyfuzję na ciągłych reprezentacjach obrazów.

### Dlaczego to bije styl Chameleona

Luka jakościowa między ciągłą-dyfuzją a dyskretnym-NTP w generacji obrazów jest mierzalna. Artykuł Transfusion raportuje:

- Przy 7B parametrach, bije model w stylu Chameleona tego samego rozmiaru na FID o 3-5 punktów.
- Nie wymaga treningu tokenizatora — enkoder obrazu jest prostszy (liniowa projekcja do ukrytego wymiaru, taka sama jak warstwa wejściowa ViT).
- Inferencja może zrównoleglić odszumianie łaek obrazkowych, w przeciwieństwie do autogresywnych tokenów obrazkowych.

Wada: Transfusion jest modelem z podwójnym kosztem, co sprawia, że dynamika treningu jest trudniejsza. Wagi kosztów wymagają dostrojenia. Niedopasowanie harmonogramu między NTP a dyfuzją może spowodować dominację jednej głowy.

### Co leży dalej

Janus-Pro (Lekcja 12.15) udoskonala pomysł Transfusion przez rozdzielenie enkodera wizyjnego dla rozumienia i generacji — SigLIP dla jednego, VQ dla drugiego — przy jednoczesnym współdzieleniu ciała transformera. Show-o (Lekcja 12.14) zamienia dyfuzję na dyskretną dyfuzję (przewidywanie maskowane). Rodzina unified-generation rozgałęzia się szybko po Transfusion.

Produkcyjne VLM 2026 roku, które emitują obrazy — Gemini 3 Pro, GPT-5, ścieżka generacji obrazów Claude Opus 4.7 — prawie na pewno używają jakiegoś potomka tej rodziny. Szczegóły są zastrzeżone.

## Use It

`code/main.py` buduje zabawkowy Transfusion na małym problemie podobnym do MNIST:

- Podpisy tekstowe to krótkie sekwencje liczb całkowitych opisujące cyfrę (0-9).
- Obrazy to siatki 4x4 bajtów.
- Para współdzielonych liniowych projekcji wag działa jako zastępnik transformera; koszt NTP na tekście, koszt MSE na zaszumionych łatkach.
- Pętla treningowa przeplata dwa koszty, maska atencji jest jawna.
- Generacja produkuje podpis tekstowy i obraz 4x4 w jednym przejściu w przód.

Transformer jest zabawkowy. Instalacja dwóch kosztów, konstrukcja maski atencji i pętla inferencji to prawdziwe artefakty.

## Ship It

Ta lekcja produkuje `outputs/skill-two-loss-trainer-designer.md`. Na podstawie nowego zadania treningu multimodalnego (tekst + obraz, tekst + dźwięk, tekst + wideo) projektuje harmonogram dwóch kosztów (wagi kosztów, kształt maski, bloki współdzielone vs specyficzne dla modalności) i wskazuje ryzyka implementacyjne.

## Exercises

1. Model w stylu Transfusion trenuje na 70% tokenów tekstowych i 30% łaek obrazkowych. Koszt dyfuzji obrazu jest około 10 razy większy od kosztu NTP tekstu. Jakie wagi kosztów je zrównoważą?

2. Zaimplementuj blokowo-trójkątną maskę dla sekwencji: `[T, T, <image>, P, P, P, P, </image>, T]`. Oznacz każdy wpis 0 lub 1.

3. MMDiT ma wagi QKV specyficzne dla modalności. Jaki narzut liczby parametrów dodaje to w porównaniu z w pełni współdzielonym transformerem Transfusion? Przy 7B parametrów, czy to się opłaca?

4. Generacja: mając prompt tekstowy, model uruchamia NTP dla 50 tokenów, następnie trafia na `<image>`, następnie uruchamia dyfuzję na 256 łatkach przez 20 kroków odszumiania. Ile przejść w przód łącznie?

5. Przeczytaj Sekcję 3 artykułu SD3. Opisz sprecyzowany przepływ i dlaczego jest zbieżny w mniejszej liczbie kroków inferencji niż DDPM.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Trening z dwoma kosztami | "NTP + dyfuzja" | Pojedynczy transformer optymalizuje zarówno entropię krzyżową na tokenach tekstowych, jak i MSE na ciągłych łatkach obrazkowych w tym samym kroku gradientu |
| Dopasowanie przepływu | "Sprecyzowany przepływ" | Wariant dyfuzji przewidujący pole prędkości od szumu do czystych danych; prostsza matematyka niż DDPM |
| MMDiT | "Multimodalny DiT" | Architektura Stable Diffusion 3: wspólna atencja, MLP i normy specyficzne dla modalności |
| Maska blokowo-trójkątna | "Przyczynowy tekst + dwukierunkowy obraz" | Maska atencji przyczynowa w poprzek tekstu, ale dwukierunkowa w obrębie regionów obrazowych |
| Ciągła reprezentacja obrazu | "Bez VQ" | Łatki obrazkowe jako wektory o wartościach rzeczywistych, nie indeksy całkowite z kodbu |
| Przewidywanie prędkości | "v-parametryzacja" | Wyjście sieci to pole prędkości między szumem a danymi, a nie sam szum |

## Further Reading

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)