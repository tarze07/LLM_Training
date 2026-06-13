# Spekulacyjne dekodowanie EAGLE-3 w produkcji

> Spekulacyjne dekodowanie łączy szybki model szkicowy z modelem docelowym. Szkic proponuje K tokenów; cel weryfikuje w jednym przejściu do przodu; zaakceptowane tokeny są darmowe. W 2026 roku EAGLE-3 jest produkcyjnym wariantem — trenuje głowę szkicową na ukrytych stanach modelu docelowego, a nie na surowych tokenach, podnosząc współczynnik akceptacji alpha do pasma 0,6-0,8 na ogólnym czacie. Właściwe pytanie nie brzmi „jak szybki jest szkic", ale „jaka jest alpha na moim ruchu?" Jeśli alpha spadnie poniżej ~0,55, spekulacyjne dekodowanie jest netto ujemne przy wysokiej współbieżności, ponieważ każdy odrzucony szkic kosztuje drugie przejście do przodu modelu docelowego. Ta lekcja uczy mierzyć alpha najpierw, a włączać flagę drugie.

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## Learning Objectives

- Wymień trzy generacje spekulacyjnego dekodowania i wyjaśnij, co EAGLE-3 zmienia w stosunku do EAGLE-2 i klasycznego modelu szkicowego.
- Zdefiniuj współczynnik akceptacji alpha, oblicz oczekiwane przyspieszenie z alpha i K (długość szkicu) oraz zidentyfikuj próg rentowności alpha dla docelowej współbieżności.
- Wyjaśnij, dlaczego spekulacyjne dekodowanie jest opcjonalne (nie domyślne) w vLLM 2026 i dlaczego włączanie go bez mierzenia alpha jest antywzorcem produkcyjnym.
- Napisz plan pomiaru: który benchmark, która dystrybucja promptów, który punkt współbieżności, którą metrykę bramkować.

## The Problem

Dekodowanie jest ograniczone pamięcią. Na H100 uruchamiającym Llama 3.3 70B FP8, każdy dekodowany token odczytuje ~140 GB/s wag i emituje jeden token. GPU jest prawie bezczynne podczas dekodowania — wąskim gardłem jest przepustowość HBM, a nie przepustowość matmul.

Spekulacyjne dekodowanie wykorzystuje tę lukę. Wygeneruj K tokenów kandydackich tanim modelem szkicowym, a następnie poproś model docelowy o zweryfikowanie wszystkich K w jednym przejściu do przodu. Każdy zweryfikowany token jest efektywnie darmowy (zamortyzowany w partię K przejść do przodu, które model docelowy musiałby i tak wykonać).

Klasyczne podejście z modelem szkicowym używa mniejszego modelu tej samej rodziny (Llama 3.2 1B szkicujący dla Llama 3.3 70B). Działa, ale współczynnik akceptacji jest przeciętny — dystrybucja mniejszego modelu odbiega od docelowego. EAGLE, następnie EAGLE-2, następnie EAGLE-3 trenują lekką głowę szkicową bezpośrednio na wewnętrznych stanach modelu docelowego, więc dystrybucja szkicu znacznie bliżej śledzi docelową. Dlatego alpha rośnie z 0,4 z modelem szkicowym do 0,6-0,8 z EAGLE-3.

Haczyk: EAGLE-3 jest opcjonalny w vLLM 2026. `speculative_config` musi być ustawiony jawnie. Brak flagi, brak przyspieszenia. Zespoły, które włączają go bez mierzenia alpha na swoim rzeczywistym ruchu, często widzą pogorszenie latencji ogona, a nie poprawę.

## The Concept

### Co faktycznie kupuje spekulacyjne dekodowanie

Bez spec decode, koszt na token to jedno przejście do przodu celu. Z spec decode przy długości szkicu K i akceptacji alpha, oczekiwane tokeny na przejście do przodu celu to `1 + K * alpha`. Przyspieszenie to `(1 + K * alpha) / (1 + epsilon)`, gdzie epsilon to narzut szkicu-plus-weryfikacji. Dla K=5, alpha=0,7: `(1 + 5*0,7) / (1 + 0,1) = 4,5 / 1,1 = 4,1x`. Rzeczywiste wartości oscylują wokół 2-3x, ponieważ alpha rzadko jest tak wysoka na ruchu produkcyjnym, a epsilon rośnie przy dużym rozmiarze partii.

### Dlaczego alpha jest jedyną metryką, która ma znaczenie

Odrzucone tokeny nie znikają — wymuszają drugie przejście do przodu celu dla pierwszego odrzuconego tokena. Na obciążeniu, gdzie alpha spada do 0,4, płacisz narzut szkicu plus weryfikację plus ponowne losowanie. Przy wysokiej współbieżności (powiedzmy 256 równoczesnych), partia dekodowania jest już wystarczająco duża, że luka w przepustowości pamięci między „sam cel" a „cel z weryfikacją" się kurczy. Poniżej alpha 0,55 na większości sprzętu 2026, spec decode jest netto ujemny.

Alpha zmienia się w zależności od obciążenia. Na ogólnym czacie w stylu ShareGPT, EAGLE-3 trenowany na ShareGPT osiąga 0,6-0,8. Na ruchu specyficznym dla domeny (kod, medyczny, prawny) głowa szkicowa trenowana na ogólnych danych spada do 0,4-0,6. Trenowanie głowy szkicowej specyficznej dla domeny odzyskuje alpha — to lekkie, szybkie zadanie treningowe w porównaniu do fine-tuningu celu.

### Generacje EAGLE w skrócie

- **Klasyczny model szkicowy**: mały model tej samej rodziny. Alpha 0,3-0,5. Prosta infrastruktura — dwa załadowane modele, szkic uruchamia K przejść do przodu na przejście do przodu celu.
- **EAGLE-1 (2024)**: pojedyncza głowa szkicowa trenowana na ukrytych stanach celu (ostatnia warstwa). Alpha ~0,5-0,6. Mały narzut parametrów na wierzchu celu.
- **EAGLE-2 (2025)**: adaptacyjna długość szkicu i szkice drzewiaste (weryfikuj wiele gałęzi w jednym przejściu celu). Alpha ~0,6-0,7. Bardziej złożony scheduler szkicu.
- **EAGLE-3 (2025-2026)**: głowa szkicowa trenowana na wielu warstwach celu (nie tylko ostatniej), lepsze dopasowanie. Alpha ~0,6-0,8 na ogólnym czacie.

### Przepis produkcyjny na 2026

1. Wdróż model docelowy czysto. Zmierz bazową TTFT, ITL, przepustowość przy docelowej współbieżności.
2. Włącz szkic EAGLE-3 przez `speculative_config` vLLM. Uruchom ponownie benchmark.
3. Loguj współczynnik akceptacji alpha. vLLM V1 raportuje to jako `spec_decode_metrics.accepted_tokens_per_request`. Podziel przez żądaną długość szkicu, aby uzyskać alpha.
4. Jeśli alpha < 0,55 na dystrybucji ruchu produkcyjnego, wyłącz spec decode lub trenuj głowę szkicową EAGLE-3 specyficzną dla domeny.
5. Przy produkcyjnej współbieżności, uruchom ponownie. Potwierdź, że P99 ITL się nie pogorszył.

### Pułapka produkcyjna: ogon P99

Średnia ITL spada z spec decode. P99 może się pogorszyć, jeśli nie dostroisz. Odrzucone szkice wyzwalają sekwencję dwuprzejściową (szkic + weryfikacja-niepowodzenie + ponowne losowanie). Przy pełnej partii, te dwa przejścia serializują się. Obserwuj P99 ITL, a nie P50.

### Gdzie EAGLE-3 jest już wdrożony

Google wdrożył spekulacyjne dekodowanie w AI Overviews w 2025 (ta sama jakość, szybsza odpowiedź). vLLM V1 dostarcza `speculative_config` jako udokumentowany interfejs; N-gram GPU spekulacyjne dekodowanie w V1 jest wariantem kompatybilnym z chunked prefill. SGLang wspiera EAGLE-3 jako rekomendowaną ścieżkę szkicu dla obciążeń z dużą liczbą prefiksów.

### Matematyka progu rentowności w jednej linii

Oczekiwane przyspieszenie: `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`. Ustawienie `S = 1` rozwiązuje dla alpha: `alpha_breakeven = verify_overhead / K`. Dla typowego verify_overhead ~0,15 i K=5: `alpha_breakeven = 0,03`. Ale to surowa matematyka dekodowania. Przy wysokiej współbieżności narzut weryfikacji rośnie, a partia dekodowania już amortyzuje odczyty pamięci między sekwencjami, więc efektywny alpha_breakeven wzrasta do ~0,45-0,55 w praktyce.

### Kiedy nie używać spekulacyjnego dekodowania

- Generowanie offline Batch-1, gdzie latencja nie ma znaczenia. Używaj czystego celu.
- Bardzo krótkie wyjścia (poniżej 50 tokenów). Narzut szkicu i koszt weryfikacji dominują.
- Wyspecjalizowane domeny bez głowy szkicowej trenowanej na domenie. Alpha zbyt niska.
- vLLM v0.18.0 plus spec decode z modelem szkicowym plus `--enable-chunked-prefill`. Ta kombinacja się nie kompiluje. Udokumentowanym wyjątkiem jest N-gram GPU spec decode w V1.

## Use It

`code/main.py` symuluje pętlę dekodowania z i bez spekulacyjnego dekodowania w zakresie wartości alpha i długości szkicu K. Wyświetla próg rentowności alpha, zmierzone przyspieszenie i zachowanie ogona. Uruchom na kilku kombinacjach (alpha, K), aby zobaczyć dokładnie, gdzie spekulacyjne dekodowanie przestaje się opłacać.

## Ship It

Ta lekcja produkuje `outputs/skill-eagle3-rollout.md`. Biorąc model docelowy, opis dystrybucji ruchu i docelową współbieżność, produkuje etapowy plan wdrożenia EAGLE-3 — zmierz bazę, włącz konfigurację, zmierz alpha, bramkuj na alpha >= 0,55, obserwuj P99 ITL.

## Exercises

1. Uruchom `code/main.py`. Przy K=5, jakiej alphy potrzebujesz dla 2x przyspieszenia? Dla 3x przyspieszenia? Jak wrażliwe jest to na verify_overhead?
2. Wyobraź sobie, że ruch produkcyjny dzieli się 70% ogólny czat, 30% kod. Ogólny czat osiąga alpha 0,7 z EAGLE-3 trenowanym na ShareGPT; kod osiąga alpha 0,4. Jaka jest mieszana alpha i czy spec decode jest netto dodatni?
3. Przeczytaj dokumentację `speculative_config` vLLM. Wymień trzy tryby (model szkicowy, EAGLE, N-gram) i który jest kompatybilny z chunked prefill.
4. Widzisz, że średnia ITL spadła o 25% po włączeniu EAGLE-3, ale P99 ITL wzrosło o 15%. Zdiagnozuj i zaproponuj mitigację.
5. Oblicz koszt pamięci głowy szkicowej EAGLE-3 dla Llama 3.3 70B. Jak wypada w porównaniu do uruchamiania Llama 3.2 1B jako klasycznego szkicu?

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Speculative decoding | „szkic plus weryfikacja" | Zaproponuj K tokenów tanim modelem, zweryfikuj wszystkie K w jednym przejściu celu |
| Acceptance rate alpha | „współczynnik akceptacji spec" | Frakcja tokenów szkicu zaakceptowanych przez cel; jedyna metryka, która ma znaczenie |
| Draft length K | „spec k" | Ile tokenów szkic proponuje na przejście celu; typowo 4-8 |
| Verify overhead epsilon | „narzut spec" | Dodatkowy koszt weryfikacji-i-ponownego losowania vs czyste przejście celu; rośnie z partią |
| EAGLE-3 | „najnowszy EAGLE" | Wariant 2025-2026; trenuje głowę szkicową na wielu warstwach celu; alpha 0,6-0,8 na ogólnym czacie |
| `speculative_config` | „konfiguracja spec vLLM" | Jawne włączenie w vLLM V1; brak domyślnego oznacza brak przyspieszenia |
| N-gram spec decode | „szkic N-gram" | Szkic po stronie GPU używający wyszukiwania N-gram w prompcie; kompatybilny z chunked-prefill |
| Break-even alpha | „alpha bez efektu" | Alpha, przy której spec decode daje zerowe przyspieszenie; obserwuj to przy produkcyjnej współbieżności |
| Rejected-draft two-pass | „koszt ponownego losowania" | Dwa przejścia celu, gdy szkice odrzucają; napędza ogon P99 |

## Further Reading

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — autorytatywne źródło o `speculative_config` i kompatybilności z chunked-prefill w V1.
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — dokładny zestaw pól.
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — oryginalna formulacja głowy szkicowej EAGLE.
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — adaptacyjne szkice i drzewa.
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — wydajny system LLM ze spekulacyjnym dekodowaniem.
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — lista kontrolna wdrożenia produkcyjnego.

(End of file - total 110 lines)