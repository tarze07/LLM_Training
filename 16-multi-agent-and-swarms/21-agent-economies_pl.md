# Ekonomie Agentów, Zachęty Tokenowe, Reputacja

> Długoterminowe autonomiczne agentów (krzywa pracy METR od 1 do 8 godzin) potrzebują sprawczości ekonomicznej. Wyłaniający się **pięciowarstwowy stos** to: **DePIN** (fizyczne obliczenia) → **Tożsamość** (W3C DIDs + kapitał reputacyjny) → **Kognicja** (RAG + MCP) → **Rozliczenia** (abstrakcja konta) → **Zarządzanie** (Agentowe DAO). Produkcyjne sieci motywacyjne dla agentów obejmują **Bittensor** (subsieci TAO nagradzają modele specyficzne dla zadań), **Fetch.ai / ASI Alliance** (ASI-1 Mini LLM + token FET) oraz **Gonka** (PoW oparty na transformerach, który realokuje moc obliczeniową do produktywnych zadań AI). Prace akademickie: zdecentralizowane LaMAS z AAMAS 2025 wykorzystuje **atrybucję kredytu opartą na wartości Shapleya** do sprawiedliwego nagradzania przyczyniających się agentów; Google Research "Mechanism design for large language models" proponuje **aukcje tokenów** z płatnością drugiej ceny przy monotonicznej agregacji. Ta lekcja buduje minimalny rynek agentów, stosuje atrybucję kredytu opartą na wartości Shapleya w wieloagentowym potoku i przeprowadza aukcję tokenów z drugą ceną, aby teoria gier stała się konkretna.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Problem

Systemy wieloagentowe stają się skomplikowane, gdy agenci wytwarzają wartość wspólnie, ale muszą być nagradzani indywidualnie. Klasyczne mechanizmy — równy podział, ostatni-kontrybutor-bierze-wszystko — są niesprawiedliwe lub podatne na manipulację. Nagradzanie oparte na koalicjach za pomocą wartości Shapleya jest z założenia sprawiedliwe, ale kosztowne obliczeniowo. Literatura z lat 2025–2026 proponuje użyteczne przybliżenia: próbkowanie Shapleya, aukcje z monotoniczną agregacją oraz reputację w łańcuchu bloków, która wynika z potwierdzonych wkładów.

Poza atrybucją kredytu, dziedzina zwróciła się w stronę faktycznych agentów ekonomicznych: Bittensor TAO nagradza moc obliczeniową wydobywania za dostrajanie modeli specyficznych dla subsieci, Fetch.ai/ASI nagradza użycie ASI-1 Mini LLM tokenami FET, Gonka realokuje transformatorowy proof-of-work w kierunku produktywnych zadań AI. Agenci, którzy dokonują autonomicznych transakcji, istnieją już dziś; pytanie brzmi, jak dostosować zachęty.

Ta lekcja traktuje ekonomie agentów jako konkretną rodzinę problemów — atrybucję kredytu, projektowanie mechanizmów i reputację — i buduje każdy z nich przy użyciu minimalnej matematyki, aby pomysły się utrwaliły.

## Koncepcja

### Pięciowarstwowy stos ekonomii agentów

1. **DePIN (fizyczne obliczenia).** Zdecentralizowana infrastruktura wynajmująca GPU, przestrzeń dyskową, przepustowość. Subsieci Bittensor, Render Network, Akash. Nie specyficzne dla agentów; agenci z tego korzystają.
2. **Tożsamość.** Zdecentralizowane Identyfikatory W3C (DID) nadają każdemu agentowi trwały identyfikator niezależny od platformy. Reputacja przypisana jest do DID. Agent Network Protocol (ANP) używa DID jako warstwy odkrywania.
3. **Kognicja.** Pętla rozumowania agenta: LLM + RAG + MCP. To właśnie budują inne fazy.
4. **Rozliczenia.** Abstrakcja konta (ERC-4337) pozwala agentom płacić za gaz z własnych sald bez posiadania ETH. Agenci mogą płacić za usługi, sobie nawzajem lub za moc obliczeniową.
5. **Zarządzanie.** Agentowe DAO: struktury zarządzania, w których ludzie *i* agenci głosują nad zmianami protokołu, z siłą głosu powiązaną z reputacją.

Nie każdy system produkcyjny wykorzystuje wszystkie pięć. Bittensor używa 1, 2, częściowo 3, częściowo 4, żadnego z 5. OpenAI agenci nie używają żadnego poza 3. Stos jest mapą referencyjną, a nie wymogiem.

### Bittensor, Fetch.ai, Gonka — co działa

**Bittensor (TAO).** Subsieci to wyspecjalizowane zadania (modelowanie języka, generowanie obrazów, prognozowanie). Górnicy przesyłają wyniki modeli. Walidatorzy je rankują; punktacja ważona stawką dystrybuuje nagrody TAO. Każda subsieć ma własną ewaluację. Lekcja ekonomiczna: płać za jakość wyników specyficznych dla zadania, a nie za wykorzystaną moc obliczeniową.

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM działa w sieci Fetch.ai; użytkownicy płacą tokenami FET za inferencję. Narracja agenci-jako-równorzędni-partnerzy jest tu silniejsza: agent na Fetch może wezwać innego do zadania i zapłacić w FET.

**Gonka.** Transformerowy proof-of-work: "praca" to przejścia w przód transformatora. Górnicy zarabiają, wykonując zadania inferencji, które mają znane poprawne wyniki (z danych treningowych). PoW produktywny dla zasobów zamiast PoW opartego na haszowaniu.

Wszystkie trzy są produkcjnie gotowe na kwiecień 2026. Dystrybucja wypłat jest różna. Bittensor nagradza jakość względem walidatorów subsieci; Fetch nagradza użyteczność mierzoną przez płacących użytkowników; Gonka nagradza weryfikowalną pracę inferencyjną.

### Atrybucja kredytu oparta na wartości Shapleya

Trzej agenci współpracują nad zadaniem. Wynik ma wartość 0.8. Kto wniósł jaki wkład?

Wartość Shapleya: unikalna alokacja kredytu spełniająca cztery aksjomaty (efektywność, symetria, liniowość, null). Dla agenta `i`:

```
shapley(i) = (1/N!) * suma po wszystkich permutacjach O z (v(S_i_O ∪ {i}) - v(S_i_O))
```

gdzie `S_i_O` to zbiór agentów przed `i` w permutacji `O`. W praktyce: wylicz wszystkie permutacje, zanotuj krańcowy wkład każdego agenta w każdej permutacji, uśrednij.

Dla N=3 agentów, jest 6 permutacji. Dla N=10, 3.6M — więc w praktyce próbkuje się permutacje, a nie wylicza wszystkie.

### Aukcja drugiej ceny dla agregacji

Google Research ("Mechanism design for large language models") proponuje aukcje tokenów z drugą ceną do agregacji wyników LLM. Konfiguracja: N agentów proponuje uzupełnienie; każdy ma prywatną wartość za bycie wybranym. Organizator aukcji wybiera ofertę o najwyższej wartości i płaci *drugą najwyższą* wartość. Przy monotonicznej agregacji (wartość zależy od tego, która oferta jest wybrana, a nie ile zostało złożonych), jest to prawdomówne — agenci podają swoją prawdziwą wartość.

Dlaczego to ma znaczenie dla systemów LLM: możesz zlecić zadania uzupełniania wielu agentom o różnych cenach; aukcja wybiera najlepszą + płaci uczciwie, a agenci nie mają motywacji do fałszywych raportów.

### Kapitał reputacyjny

Wynik reputacji związany z DID, który kumuluje się z potwierdzonych wkładów. Prosta reguła aktualizacji:

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

Ze współczynnikiem zaniku `alpha` bliskim 1. Reputacja:

- Jest tania do odczytu dla decyzji routingu ("wyślij trudne zadania do agentów o wysokiej reputacji").
- Jest kosztowna do sfałszowania (kumuluje się w czasie, związana z DID).
- Może być pomniejszana: wkłady, które nie przejdą weryfikacji, odejmują punkty.

### Zdecentralizowane LaMAS z AAMAS 2025

Propozycja LaMAS (AAMAS 2025) łączy: tożsamość DID, atrybucję kredytu opartą na wartości Shapleya oraz prosty mechanizm aukcyjny. Kluczowe twierdzenie: decentralizacja kroku atrybucji kredytu czyni system audytowalnym i odpornym na manipulacje z pojedynczego punktu.

### Gdzie ekonomia się załamuje

- **Manipulacja wyrocznią cenową.** Jeśli funkcja kredytu może być oszukana, agenci będą ją oszukiwać. Każdy mechanizm potrzebuje testu adversarialnego.
- **Ataki Sybili.** Jeden operator uruchamia N fałszywych agentów, aby zawyżyć swój własny wkład. DID spowalniają, ale nie powstrzymują tego; koszt sfałszowania reputacji jest środkiem zaradczym.
- **Koszt weryfikacji.** Atrybucja kredytu jest tylko tak sprawiedliwa, jak weryfikator. Jeśli weryfikacja jest tania (mały LLM), może być oszukana; jeśli droga (panel ludzi), system nie skaluje się.
- **Ryzyko regulacyjne.** Ekonomie agentów krzyżują się z regulacjami finansowymi. Bittensor, Fetch i Gonka działają w szarych strefach prawnych w niektórych jurysdykcjach od 2026 roku.

### Kiedy ekonomie agentów mają sens

- **Otwarte sieci z heterogenicznymi operatorami.** Żaden zespół nie kontroluje wszystkich agentów.
- **Weryfikowalne wyniki.** Bez weryfikacji atrybucja kredytu jest zgadywanką.
- **Długoterminowe przepływy pracy.** Zadania jednorazowe nie korzystają z kumulacji reputacji.
- **Tokenizowane płatności są prawnie wykonalne** w twojej jurysdykcji.

W zamkniętych systemach korporacyjnych ekonomia ustępuje prostszej alokacji (menedżerowie przydzielają pracę, metryki są wewnętrzne). Literatura ekonomiczna ma zastosowanie głównie do otwartych sieci.

## Build It

`code/main.py` implementuje:

- `shapley(value_fn, agents)` — dokładne obliczenie Shapleya przez wyliczenie dla małego N.
- `second_price_auction(bids)` — prawdomówny mechanizm; zwycięzca płaci drugą najwyższą.
- `Reputation` — reputacja związana z DID z wykładniczym zanikiem i pomniejszaniem.
- Demo 1: trzech agentów współpracuje, dokładny Shapley przypisuje kredyt.
- Demo 2: pięciu agentów licytuje o slot zadania; aukcja drugiej ceny wybiera zwycięzcę + płatność.
- Demo 3: 100 rund przydzielania zadań agentom o heterogenicznej reputacji; routing ważony reputacją bije losowy.

Uruchom:

```
python3 code/main.py
```

Oczekiwane wyniki: wartości Shapleya dla każdego agenta; wynik aukcji pokazujący równowagę prawdomównych ofert; routing ważony reputacją pokazujący 10-20% wzrost jakości nad losowym po rozgrzaniu.

## Use It

`outputs/skill-economy-designer.md` projektuje minimalną ekonomię agentów: wybór warstwy tożsamości, mechanizmu atrybucji kredytu, mechanizmu płatności, reguły reputacji.

## Ship It

Uruchamianie ekonomii agentów w 2026:

- **Zacznij od reputacji, nie tokenów.** Reputacja jest tania we wdrożeniu i wartościowa sama w sobie; tokeny dodają złożoności prawnej i ekonomicznej.
- **Weryfikuj, zanim nagrodzisz.** Nigdy nie dystrybuuj kredytu bez niezależnego kroku weryfikacji. Samozgłoszona jakość prowadzi do gier sybilskich.
- **Shapley-próbkuj, nie Shapley-dokładnie.** Próbkuj 100-1000 permutacji; dokładne wyliczanie nie skaluje się.
- **Ogranicz współczynnik zaniku i ustaw podłogę reputacji.** Nieograniczony zanik usuwa prawowitych kontrybutorów; zbyt wolny zanik nagradza nieaktualnych agentów o wysokiej reputacji.
- **Audytuj mechanizmy adversarialnie.** Przeprowadź scenariusze red-team przed otwarciem sieci. Każdy mechanizm ma teorię gier; chcesz znaleźć dziury, nie atakujących.

## Ćwiczenia

1. Uruchom `code/main.py`. Potwierdź, że wartości Shapleya sumują się do całkowitej wartości (aksjomat efektywności). Zmień funkcję wartości; czy alokacje Shapleya zmieniają się w oczekiwanym kierunku?
2. Zaimplementuj *próbkowanie* Shapleya (Monte Carlo po K permutacjach). Jak K wpływa na dokładność przybliżenia? Porównaj z dokładnym dla N=4.
3. Zaimplementuj krok tworzenia koalicji przed aukcją: agenci mogą łączyć się w zespoły i licytować jako jednostka. Które koalicje się tworzą? Czy wynik jest Pareto-lepszy niż indywidualne licytowanie?
4. Przeczytaj post Google Research o projektowaniu mechanizmów. Zidentyfikuj jedno założenie, którego naruszenie niszczy prawdomówność. Jak wygląda ten tryb awarii w kontekście LLM?
5. Przeczytaj artykuł o zdecentralizowanym LaMAS z AAMAS 2025. Zaimplementuj ich krok Shapleya dla 10 agentów na syntetycznym zadaniu. Ile czasu zajmuje dokładne obliczenie? Jak blisko jest próbkowanie przy 100 losowaniach?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|----------------|------------------------|
| DePIN | "Zdecentralizowana fizyczna infrastruktura" | Obliczenia/przestrzeń/przepustowość motywowane tokenami. Bittensor, Akash, Render. |
| DID | "Zdecentralizowany identyfikator" | Specyfikacja W3C dla przenośnych identyfikatorów. Reputacja agenta wiąże się z DID, a nie z platformą. |
| ERC-4337 | "Abstrakcja konta" | Konta kontraktowe, które mogą sponsorować gaz, umożliwiając płatności agentów. |
| Wartość Shapleya | "Sprawiedliwa atrybucja kredytu" | Unikalna alokacja spełniająca efektywność, symetrię, liniowość, null. |
| Aukcja drugiej ceny | "Aukcja Vickreya" | Prawdomówny mechanizm: zwycięzca płaci drugą najwyższą ofertę. Kompatybilna z monotoniczną agregacją. |
| Kapitał reputacyjny | "Skumulowany wynik jakości" | Wynik związany z DID z potwierdzonych wkładów; zanika w czasie. |
| Agentowe DAO | "Agenci + ludzie zarządzają" | DAO z agentami-głosującymi jako pierwszorzędnymi obywatelami, siła głosu związana z reputacją. |
| TAO / FET / GPU credits | "Nominały tokenów" | Bittensor TAO, Fetch.ai FET, różne tokeny DePIN. |

## Dalsza lektura

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — przegląd z 2026 pięciowarstwowego stosu ekonomii agentów
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — aukcje tokenów z monotoniczną agregacją
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — atrybucja kredytu oparta na wartości Shapleya
- [Dokumentacja Bittensor TAO](https://docs.bittensor.com/) — struktura subsieci i dystrybucja nagród
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM i token FET
- [Specyfikacja W3C Decentralized Identifiers (DIDs)](https://www.w3.org/TR/did-core/) — fundament tożsamości