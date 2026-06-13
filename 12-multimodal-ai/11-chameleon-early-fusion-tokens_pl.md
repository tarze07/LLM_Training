# Chameleon i modele multimodalne oparte wyłącznie na tokenach z wczesną fuzją

> Każdy VLM, który do tej pory widzieliśmy, utrzymuje obrazy i tekst oddzielnie. Tokeny wizualne pochodzą z enkodera wizyjnego, przepływają przez projektor, a następnie spotykają tekst wewnątrz LLM-a. Słowniki wizyjne i tekstowe nigdy się nie pokrywają. Chameleon (Meta, maj 2024) zapytał: a gdyby tak się pokrywały? Wytrenuj VQ-VAE, który zamienia obraz w sekwencję dyskretnych tokenów ze wspólnego słownika. Każdy multimodalny dokument to teraz jedna sekwencja — tokeny tekstowe i obrazkowe przeplatane, z pojedynczym autogresywnym kosztem. Skutek uboczny: model może generować wyniki o mieszanej modalności — naprzemienne tokeny tekstowe i obrazkowe w pojedynczym wywołaniu inferencji. Ta lekcja czyta tezę wczesnej fuzji i buduje jej zabawkową wersję od początku do końca.

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## Learning Objectives

- Wyjaśnij, dlaczego wspólne słownictwo + pojedynczy koszt zmienia możliwości modelu.
- Opisz, w jaki sposób VQ-VAE tokenizuje obraz do dyskretnej sekwencji kompatybilnej z celem przewidywania następnego tokena w transformerze.
- Wymień triki stabilności treningowej Chameleona: QK-Norm, rozmieszczenie dropoutu, kolejność LayerNorm.
- Porównaj Chameleon z podejściem Q-Former z BLIP-2 i opisz, kiedy każde z nich jest właściwym wyborem.

## The Problem

Adapterowe VLM (LLaVA, BLIP-2, Qwen-VL) traktują tekst i obraz jako dwie różne rzeczy. Token tekstowy przechodzi przez `embed(text_token)`; obraz przechodzi przez `visual_encoder(image) → projector → ... pseudo_tokens`. Model ma dwie ścieżki wejściowe, które łączą się w połowie.

Trzy konsekwencje:

1. LLM może tylko konsumować obrazy, nie emitować ich. Wynik to tylko tekst.
2. Dokumenty o mieszanej modalności (naprzemienne akapity i obrazy, jak w artykule) są niewygodne — albo przetwarzasz multimodalne wejście poza modelem, albo łańcuchujesz generacje.
3. Niedopasowanie dystrybucyjne. Tokeny wizyjne i tekstowe żyją w różnych regionach przestrzeni ukrytej, tworząc subtelne problemy z dopasowaniem.

Chameleon odrzuca to założenie: obrazy to tylko sekwencje dyskretnych tokenów ze wspólnego słownika. Trenuj model na przeplatanych dokumentach, jeden koszt, jeden autogresywny dekoder, a generacja mieszanej modalności pojawia się za darmo.

## The Concept

### VQ-VAE jako tokenizator obrazów

Tokenizator to wektorowo-kwantowany autoenkoder wariacyjny. Architektura:

- Enkoder: CNN + ViT mapujący obraz na przestrzenną mapę cech, np. 32x32 cech o wymiarze 256.
- Kodbuk: nauczone słownictwo K wektorów (Chameleon używa 8192), również o wymiarze 256.
- Kwantyzacja: dla każdej cechy przestrzennej znajdź najbliższy wpis w kodbukowej odległości L2. Zastąp ciągłą cechę indeksem całkowitym.
- Dekoder: CNN przekształcający skwantowane cechy z powrotem w piksele.

Trening: koszt rekonstrukcji VAE + koszt zaangażowania + koszt kodbu. Indeksy kodbu tworzą dyskretny alfabet dla obrazów.

Dla Chameleona: jeden obraz staje się 32*32 = 1024 tokenami ze słownika 8192. Połącz z tokenami tekstowymi (ze słownika BPE LLM-a, np. 32000). Ostateczne słownictwo: 40192. Transformer widzi jedną sekwencję, jeden koszt.

### Wspólne słownictwo

Słownictwo Chameleona łączy tokeny tekstowe, tokeny obrazkowe i separatory modalności. Każdy token ma pojedyncze ID. Warstwa embeddowania wejściowego mapuje każde ID na ukryty wektor o wymiarze D. Projekcja wyjściowa mapuje ukryty wektor z powrotem na logity słownictwa. Softmax wybiera następny token, niezależnie od modalności.

Separatory mają znaczenie: znaczniki `<image>` i `</image>` otaczają sekwencję tokenów obrazkowych. Podczas generacji, jeśli model wyemituje `<image>`, oprogramowanie niższego poziomu wie, że następne 1024 tokenów to indeksy VQ do wysłania do dekodera w celu renderowania pikseli.

### Generacja o mieszanej modalności

Inferencja to przewidywanie następnego tokena we wspólnym słownictwie. Przykładowy prompt: "Narysuj kota i opisz go." Chameleon emituje:

```
<image> 4821 1029 2891 ... (1024 tokenów obrazkowych) </image>
Kot jest pomarańczowy, siedzi na parapecie...
```

Model samodzielnie wybiera kolejność — może wyprodukować obraz potem tekst, tekst potem obraz, lub przeplatać. Ten sam dekoder, ten sam koszt.

Porównaj z adapterowymi VLM, gdzie generacja jest tylko tekstowa. Chameleon ponownie otwiera pytanie o modalności wyjściowe modelu.

### Stabilność treningowa — QK-Norm, dropout, kolejność LayerNorm

Trening z wczesną fuzją jest niestabilny w skali. Artykuł Chameleona dokumentuje trzy triki:

- QK-Norm. Zastosuj LayerNorm do projekcji zapytań i kluczy wewnątrz atencji, przed iloczynem skalarnym. Zapobiega eksplozji wielkości logitów na głębokości. Używane przez wiele dużych modeli po 2024.
- Rozmieszczenie dropoutu. Dropout po każdym dodaniu residuum, nie tylko po atencji i MLP. Wymagana większa regularyzacja, gdy gradienty z tokenów obrazkowych mogą dominować.
- Kolejność LayerNorm. Pre-LN na gałęzi residuum (standardowo), plus dodatkowa LN na połączeniu skipowym ostatniego bloku. Stabilizuje przepływ gradientów w ostatniej warstwie.

Bez tych trików trening Chameleona 34B rozchodził się na wielu etapach. Z nimi jest zbieżny. Przepis treningowy jest równie ważny co architektura.

### Sufit rekonstrukcji tokenizatora

VQ-VAE jest stratny. Przy 8192 wpisach w kodbuk i 1024 tokenach na obraz 512x512, PSNR rekonstrukcji osiąga około 26-28 dB. To wystarcza do rozpoznawalnej generacji obrazów, ale wyraźnie gorsze od dyfuzji w przestrzeni ciągłej (Stable Diffusion 3 osiąga 32+ dB).

Tokenizator jest wąskim gardłem. Lepsze tokenizatory (MAGVIT-v2, IBQ, SBER-MoVQGAN) podnoszą sufit. Emu3 (Lekcja 12.12) osiąga jakość SDXL dzięki samemu lepszemu tokenizatorowi.

### Chameleon vs BLIP-2 / LLaVA

Chameleon (wczesna fuzja, wspólne słownictwo):
- Jeden koszt, jeden dekoder.
- Generuje wyniki o mieszanej modalności.
- Tokenizator jest sufitem jakości.
- Kosztowny: dekoder VQ-VAE na każdy wygenerowany obraz na ścieżce inferencji.

BLIP-2 / LLaVA (późna fuzja, osobne wieże):
- Tylko wizja na wejściu, tekst na wyjściu.
- Wykorzystuje pretrenowany LLM.
- Brak wąskiego gardła tokenizatora dla rozumienia.
- Tani: pojedyncze przejście w przód.

Wybierz według zadania. Jeśli potrzebujesz generacji obrazów, rodzina Chameleona. Jeśli potrzebujesz tylko rozumienia, adapter-VLM jest prostszy i wykorzystuje więcej pretrenowanej mocy obliczeniowej.

### Fuyu i AnyGPT

Fuyu (Adept, 2023) to pokrewne podejście: pomiń osobny enkoder wizyjny, podawaj surowe łatki obrazu przez projekcję wejściową LLM-a tak, jakby były tokenami, bez tokenizatora. Prostsze niż Chameleon, traci generację wyjściową ze wspólnym słownictwem.

AnyGPT (Zhan et al., 2024) rozszerza Chameleona na cztery modalności: tekst, obraz, mowę, muzykę. Ten sam trik VQ-VAE dla każdej, wspólny transformer. Generacja dowolna-do-dowolnej. Omówione szerzej w Lekcji 12.16.

## Use It

`code/main.py` buduje zabawkowy model wczesnej fuzji od początku do końca:

- Miniaturowy kwantyzator w stylu VQ-VAE, który mapuje łatki 8x8 na indeksy kodbu (K=16).
- Wspólne słownictwo (id tekstowe 0..31) + (id obrazkowe 32..47) + (separatory 48, 49).
- Zabawkowy autogresywny dekoder (tablica bigramów) trenowany na syntetycznych podpisach + sekwencjach tokenów obrazkowych.
- Pętla samplowania emitująca naprzemienne tokeny tekstowe + obrazkowe na podstawie promptu.

Kod celowo utrzymuje transformer w małym rozmiarze (bigramy), abyś mógł prześledzić przepływ sygnału od początku do końca.

## Ship It

Ta lekcja produkuje `outputs/skill-tokenizer-vs-adapter-picker.md`. Na podstawie specyfikacji produktu (tylko rozumienie vs rozumienie + generowanie, wymagana jakość obrazu, budżet kosztów) wybiera między rodziną Chameleona (wczesna fuzja) a rodziną LLaVA (późna fuzja) i uzasadnia wybór ilościowymi regułami kciuka.

## Exercises

1. Chameleon używa K=8192 wpisów w kodbuk i 1024 tokenów na obraz 512x512. Oszacuj współczynnik kompresji w porównaniu z 24-bitowym obrazem RGB. Czy to stratne? Jak stratne?

2. Obraz 4K (3840x2160) przy tej samej gęstości VQ-VAE produkuje ile tokenów obrazkowych? Czy model w stylu Chameleona może wygenerować obraz 4K w jednym wywołaniu inferencji? Co psuje się pierwsze — kontekst, jakość tokenizatora, czy pamięć podręczna KV?

3. Zaimplementuj QK-Norm w czystym Pythonie. Dla 64-wymiarowego zapytania i klucza pokaż iloczyn skalarny przed i po LayerNorm. Dlaczego kontrola wielkości jest ważna na głębokości?

4. Przeczytaj Sekcję 2.3 artykułu Chameleona o stabilności treningowej. Opisz dokładny tryb awarii, który artykuł zaobserwował przy 34B bez QK-Norm. Jaki był sygnał "eksplozji norm"?

5. Rozszerz zabawkowy dekoder, aby emitował odpowiedź o mieszanej modalności na podstawie promptu tylko tekstowego. Zmierz, jak często model wybiera obraz-pierwszy vs tekst-pierwszy, biorąc pod uwagę rozkład danych treningowych 60% tekst-pierwszy / 40% obraz-pierwszy.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Wczesna fuzja | "Ujednolicone tokeny" | Obrazy konwertowane na dyskretne tokeny dzielące słownictwo transformera od pierwszego kroku |
| VQ-VAE | "Tokenizator obrazów" | CNN + ViT + kodbuk mapujący obrazy na indeksy całkowite, które transformer może przewidywać |
| Wspólne słownictwo | "Jeden słownik" | Pojedyncza przestrzeń ID tokenów obejmująca tekst + obraz + separatory modalności |
| QK-Norm | "Stabilizator atencji" | LayerNorm zastosowany do zapytania i klucza przed ich iloczynem skalarnym, zapobiega eksplozji normy |
| Generacja o mieszanej modalności | "Wyjście tekst + obraz" | Inferencja, która autonomicznie produkuje przeplatane tokeny tekstowe i obrazkowe w jednym przebiegu |
| Rozmiar kodbu | "K wpisów" | Liczba dyskretnych wektorów, do których VQ-VAE może kwantyzować; kompromis między kompresją a wiernością |
| Sufit tokenizatora | "Limit rekonstrukcji" | Najlepszy PSNR osiągalny przez dekodowanie tokenów VQ; ogranicza jakość obrazu modelu |

## Further Reading

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)