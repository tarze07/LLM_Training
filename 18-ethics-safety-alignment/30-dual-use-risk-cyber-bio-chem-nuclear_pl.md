# Ryzyko Podwójnego Zastosowania — Wzrost Cyber, Bio, Chem, Nuklearny

> Obraz podwójnego zastosowania w 2026 roku, dziedzina po dziedzinie. Bio/chem: Lekcja 17 obejmuje WMDP; próba pozyskiwania broni biologicznej Anthropic (2.53x wzrost) i ostrzeżenie OpenAI Preparedness Framework v2 z kwietnia 2025 ("na progu znaczącego pomagania nowicjuszom w tworzeniu znanych zagrożeń biologicznych") wyznaczają punkt przegięcia. Cyber (raport Anthropic listopad 2025): państwowi aktorzy powiązani z Chinami użyli agentskiego narzędzia do kodowania Claude'a do zautomatyzowania do 90% kampanii cyberataku, z interwencją człowieka tylko w 4-6 krokach; pilotaż "trusted access" OpenAI daje zweryfikowanym organizacjom bezpieczeństwa dostęp do zdolności do defensywnej pracy podwójnego zastosowania. Erozja luki wykonawczej chem/bio: klasyczną obroną było "sam dostęp do informacji jest niewystarczający." Modele frontier z wizją (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) mogą obserwować wideo z mokrego laboratorium i zapewniać korektę w czasie rzeczywistym. Grudzień 2025: OpenAI zademonstrowało GPT-5 iterujące na eksperymentach w mokrym laboratorium, osiągając 79x poprawę wydajności przez optymalizację protokołu napędzaną AI. Wzorzec nowicjusz-vs-ekspert: AI zapewnia większy względny wzrost nowicjuszom, ale większą absolutną zdolność ekspertom.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 18 (safety frameworks), Phase 18 · 28 (ecosystem)
**Time:** ~75 minutes

## Learning Objectives

- Opisać narrację wzrostu bio 2024-2025: "niewielki wzrost" -> "na progu" -> "2.53x wzrost niewystarczający do wykluczenia ASL-3."
- Opisać raport cybernetyczny Anthropic z listopada 2025: automatyzacja powiązana z Chinami do 90% kampanii cyberataku.
- Opisać erozję luki wykonawczej chem/bio: korekta w czasie rzeczywistym eksperymentów w mokrym laboratorium z obsługą wizji.
- Wymienić asymetrię względny nowicjusza vs absolutny eksperta i jej implikację dla konstrukcji przypadku bezpieczeństwa.

## The Problem

Lekcja 17 to metodologia pomiaru. Lekcja 30 to stan pomiaru w 2026 roku. Obraz zmienił się materialnie między 2024 a końcem 2025: każda dziedzina przekroczyła próg, którego frameworki z 2024 nie przewidziały.

## The Concept

### Narracja wzrostu bio/chem

Trzy fazy (powtórzone z Lekcji 17 dla spójności):

1. **2024 "niewielki wzrost."** Wczesne ewaluacje Preparedness/RSP raportowały małe przewagi nowicjuszy nad wyszukiwaniem w internecie.
2. **Kwiecień 2025 "na progu."** OpenAI PF v2 ostrzegło, że modele są "na progu znaczącego pomagania nowicjuszom w tworzeniu znanych zagrożeń biologicznych."
3. **Próba pozyskiwania broni biologicznej Anthropic 2025.** Kontrolowane badanie z nowicjuszami; 2.53x wzrost w fazach pozyskiwania; niewystarczający do wykluczenia ASL-3.

Zmiana jest jakościowa: "niewielki" ewoluowało w "plausibly enabling" w ciągu osiemnastu miesięcy, nawet bez przełomu w zdolności.

### Erozja luki wykonawczej chem/bio

Historyczna obrona: informacja jest konieczna, ale niewystarczająca; umiejętność wykonania protokołu blokuje nowicjuszy. Modele frontier 2025 z wizją częściowo przełamują tę obronę:

- **Korekta protokołu w czasie rzeczywistym.** GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1 mogą obserwować wideo z mokrego laboratorium i flagować błędy w trakcie procedury.
- **Demonstracja OpenAI grudzień 2025.** GPT-5 iterujące na eksperymentach w mokrym laboratorium osiąga 79x poprawę wydajności przez optymalizację protokołu.

Implikacja: umiejętność-wykonania-jako-obrona eroduje. Luki w zaopatrzeniu i sprzęcie pozostają, ale luka w wiedzy ukrytej się zawęża.

### Wzrost cybernetyczny (listopad 2025)

Raport Anthropic z listopada 2025: państwowi aktorzy powiązani z Chinami użyli agentskiego narzędzia do kodowania Claude'a do zautomatyzowania 80-90% kampanii cyberataku. Interwencja człowieka była wymagana tylko w 4-6 krokach.

Implikacje:
- Agentskie kodowanie jest prymitywem automatyzacji ataku. Poprzednia pomoc AI w cyber była ograniczona do poziomu fragmentu kodu; workflows agentskie integrują rozpoznanie, eksploatację, post-eksploatację i eksfiltrację.
- 4-6 kroków ludzkich jest wąskim gardłem; przyszłe zyski w zdolności zmniejszyłyby tę liczbę.
- Defensywne podwójne zastosowanie: pilotaż "trusted access" OpenAI zapewnia zweryfikowanym organizacjom bezpieczeństwa (ustalone firmy reagowania na incydenty, rząd) dostęp do zdolności do obrony. Asymetria w dostępie faworyzuje obrońców, jeśli pilotaż się skaluje.

### Nuklearny

Najmniej analizowana z czterech domen CBRN w publicznej dokumentacji. Model zagrożenia jest inny: pozyskiwanie materiałów rozszczepialnych dominuje trudność, nie informacja. Wzrost AI na warstwie informacyjnej zapewnia ograniczony wzrost nowicjuszowi w praktyce. Żaden raport głównego laboratorium z 2024-2025 nie identyfikuje przekroczenia progu specyficznego dla nuklearnego.

### Względny nowicjusza vs absolutny eksperta

Wzorzec we wszystkich czterech domenach:

- **Wzrost względny nowicjusza.** Wysoki. Mnożnikowy. Według Anthropic 2025 bio, 2.53x.
- **Zdolność absolutna eksperta.** Wysoki sufit. Ekspert wydobywa więcej niż nowicjusz, ponieważ ekspert wie, o co pytać i jak interpretować.

Implikacja dla przypadków bezpieczeństwa: adresowanie tylko wzrostu nowicjusza (przez filtry wejścia, odmowy, niepewność) jest niewystarczające dla kontroli absolutnej eksperta. Wymagane dodatkowe środki: hartowanie wywołania, odkłócanie zdolności (Lekcja 17) i protokoły kontroli (Lekcja 10).

### Synteza międzydomenowa

| Domain | 2024 | 2025 | Inflection |
|------|------|------|------------|
| Bio | niewielki wzrost | 2.53x wzrost, podejście ASL-3 | automatyzacja fazy pozyskiwania |
| Chem | niewielki wzrost | erozja luki wykonawczej przez wizję | korekta mokrego laboratorium w czasie rzeczywistym |
| Cyber | asysta kodu | 80-90% automatyzacja kampanii | agentskie kodowanie |
| Nuclear | ograniczony | ograniczony | wąskie gardło dostępu do materiałów utrzymuje się |

Trzy domeny przekroczyły progi. Jedna pozostaje ograniczona barierami nieinformacyjnymi.

### Miejsce w Fazie 18

Lekcja 30 jest zwieńczeniem: bieżący obraz podwójnego zastosowania, do którego pomiaru, ograniczania lub rządzenia każda poprzednia lekcja się przyczynia. Lekcje 17-18 dają pomiar i frameworki; Lekcje 12-16 dają narzędzia ewaluacyjne; Lekcje 24-25 dają warstwę regulacyjną i ujawniania; Lekcja 28 daje ekosystem badawczy. Lekcja 30 jest miejscem, gdzie lądują dowody.

## Use It

Brak kodu. Przeczytaj raport cybernetyczny Anthropic z listopada 2025, aktualizację OpenAI Preparedness Framework v2 z kwietnia 2025 i podsumowanie Council on Strategic Risks 2025 AI x Bio.

## Ship It

Ta lekcja produkuje `outputs/skill-dual-use-triage.md`. Dla twierdzenia o zdolności z 2026 lub raportu incydentu, triage'uje przez cztery domeny i identyfikuje, czy twierdzenie wpływa na wzrost względny nowicjusza, zdolność absolutną eksperta, czy oba.

## Exercises

1. Przeczytaj raport cybernetyczny Anthropic z listopada 2025. Wylicz 4-6 kroków interwencji człowieka i uzasadnij, który zostałby pierwszy zautomatyzowany w modelu następnej generacji.

2. Luka wykonawcza chem/bio eroduje przez wizję. Zaprojektuj ewaluację, która mierzy wzrost wiedzy ukrytej bez przekraczania granic ITAR/EAR.

3. Wzrost nuklearny wydaje się ograniczony dostępem do materiałów. Argumentuj za i przeciw stanowisku, że przyszły przełom AI mógłby przesunąć to wąskie gardło.

4. Zbuduj przypadek bezpieczeństwa (trzy filary z Lekcji 18) dla modelu frontier zdolnego cybernetycznie, który ogranicza zarówno wzrost nowicjusza, jak i eksperta.

5. Wybierz jedną z czterech domen i napisz jednotorową prognozę na 2027 opartą na trajektorii 2024-2025. Zidentyfikuj dowody, które sfalsyfikowałyby twoją prognozę.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Wzrost | "AI pomaga napastnikom" | Wzrost zdolności napastnika przypisywalny asyście AI |
| Wzrost względny nowicjusza | "mnożnikowy" | Jak bardzo AI pomaga nowicjuszowi vs status quo |
| Zdolność absolutna eksperta | "sufit" | Maksymalna zdolność, którą ekspert może wydobyć z modelu |
| Luka wykonawcza | "robienie vs wiedza" | Historyczna obrona: ukryta umiejętność mokrego laboratorium blokuje nowicjuszy |
| Agentskie kodowanie | "autonomiczne ataki" | Wieloetapowe autonomiczne wykonywanie zadań cybernetycznych |
| Faza pozyskiwania | "kroki przed syntezą" | Etapy zaopatrzenia, sprzętu, pozwoleń zagrożenia bio |
| Trusted access | "pilotaż tylko dla obrońców" | Program OpenAI 2025 dający zweryfikowanym obrońcom dostęp do zdolności |

## Further Reading

- [Anthropic — November 2025 cyber threat report](https://www.anthropic.com/news/disrupting-AI-espionage) — automatyzacja kampanii powiązanej z Chinami
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — bio "na progu"
- [Anthropic — RSP v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — progi bio ASL-3
- [Council on Strategic Risks — 2025 AI x Bio wrapup](https://councilonstrategicrisks.org/2025/12/22/2025-aixbio-wrapped-a-year-in-review-and-projections-for-2026/) — synteza końca roku