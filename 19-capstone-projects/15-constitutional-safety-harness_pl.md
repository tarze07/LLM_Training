# Capstone 15 — Konstytucyjny Harness Bezpieczeństwa + Poligon Red-Team

> Konstytucyjne Klasyfikatory Anthropica, Llama Guard 4 Meta, ShieldGemma-2 Google, Nemotron 3 Content Safety NVIDIA i X-Guard dla pokrycia wielojęzycznego zdefiniowały stos klasyfikatorów bezpieczeństwa w 2026 roku. garak, PyRIT, NVIDIA Aegis i promptfoo stały się standardowymi narzędziami ewaluacji adversarialnej. NeMo Guardrails v0.12 łączy je w produkcyjny potok. Ten capstone łączy to wszystko: warstwowy harness bezpieczeństwa wokół docelowej aplikacji, autonomicznego agenta red-team uruchamiającego 6+ rodzin ataków i konstytucyjny przebieg samokrytyki produkujący mierzalną różnicę nieszkodliwości.

**Type:** Capstone
**Languages:** Python (safety pipeline, red team), YAML (policy configs)
**Prerequisites:** Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 18 (ethics, safety, alignment)
**Phases exercised:** P10 · P11 · P13 · P14 · P18
**Time:** 25 hours

## Problem

Granica bezpieczeństwa LLM w 2026 roku nie polega na tym, czy klasyfikatory działają (działają, mniej więcej), ale jak poprawnie je złożyć wokół aplikacji produkcyjnej bez nadmiernego odmawiania lub pozostawiania oczywistych luk. Llama Guard 4 obsługuje naruszenia polityki w języku angielskim. X-Guard (132 języki) obsługuje wielojęzyczny jailbreak. ShieldGemma-2 łapie wstrzykiwanie promptów oparte na obrazach. NVIDIA Nemotron 3 Content Safety obejmuje kategorie korporacyjne. Konstytucyjne Klasyfikatory Anthropica to osobne podejście używane podczas trenowania, a nie serwowania.

Ewolucja ataków również ma znaczenie. PAIR i TAP automatyzują odkrywanie jailbreaków. GCG uruchamia ataki sufiksów gradientowych. Ataki wieloobrotowe i code-switch wykorzystują pamięć agenta. Każdy wdrożony LLM potrzebuje poligonu red-team — garak i PyRIT to kanoniczne sterowniki — plus udokumentowane łagodzenia i wyniki ze skalą CVSS.

Utwardzisz docelową aplikację (albo 8B model instruowany, albo jeden z chatbotów RAG z innych capstone'ów), uruchomisz 6+ rodzin ataków przeciwko niej i wyprodukujesz pomiar nieszkodliwości przed/po.

## Koncepcja

Potok bezpieczeństwa ma pięć warstw. **Oczyszczanie wejścia**: usuń znaki zerowej szerokości, dekoduj base64/rot13, normalizuj Unicode. **Warstwa polityki**: szyny NeMo Guardrails v0.12 (poza domeną, toksyczność, ekstrakcja PII). **Bramka klasyfikatora**: Llama Guard 4 na wejściu, X-Guard na nie-angielskim, ShieldGemma-2 na wejściach obrazowych. **Model**: docelowy LLM. **Filtr wyjścia**: Llama Guard 4 na wyjściu, Presidio PII scrub, egzekwowanie cytowań tam, gdzie ma zastosowanie. **Warstwa HITL**: wyjścia oznaczone jako wysokiego ryzyka trafiają do kolejki Slack.

Poligon red-team działa zgodnie z harmonogramem. PAIR i TAP autonomicznie odkrywają jailbreaks. GCG uruchamia ataki sufiksów gradientowych. Ataki kodowania ASCII / base64 / rot13. Ataki wieloobrotowe (adoptowanie osobowości, wykorzystanie pamięci). Ataki code-switch (mieszanie angielskiego z suahili lub tajskim). Każdy przebieg produkuje strukturalny plik wyników z punktacją CVSS i harmonogramem ujawnienia.

Przebieg konstytucyjnej samokrytyki to interwencja w czasie trenowania. Weź 1k promptów próbujących szkodzić, niech model naszkicuje odpowiedź, skrytykuj ją względem pisemnej konstytucji (zasady nie-szkodzenia) i trenuj ponownie na pętli krytyki. Zmierz różnicę nieszkodliwości przed/po na wstrzymanej ewaluacji.

## Architektura

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## Stack

- Klasyfikatory bezpieczeństwa: Llama Guard 4, ShieldGemma-2, NVIDIA Nemotron 3 Content Safety, X-Guard
- Framework Guardrail: NeMo Guardrails v0.12 + OPA
- Sterowniki red-team: garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo
- Agenty jailbreak: PAIR (Chao et al., 2023), Tree-of-Attacks (TAP), GCG suffix
- Trenowanie konstytucyjne: pętla samokrytyki w stylu Anthropic + SFT na krytykach
- Czyszczenie PII: Presidio
- Cel: 8B model instruowany lub jeden z chatbotów RAG z innych capstone'ów

## Build It

1. **Konfiguracja celu.** Uruchom 8B model instruowany na vLLM (lub użyj ponownie chatbota RAG z innego capstone). To jest aplikacja pod testem.

2. **Otoczka potoku bezpieczeństwa.** Podłącz pięciowarstwowy potok wokół celu. Zweryfikuj, że każda warstwa jest indywidualnie obserwowalna (span na warstwę w Langfuse).

3. **Pokrycie klasyfikatorów.** Załaduj Llama Guard 4, X-Guard (wielojęzyczny), ShieldGemma-2 (obrazy). Uruchom każdy na małym oznaczonym zestawie, aby ustalić baseline.

4. **Harmonogram red-team.** Zaplanuj garak, PyRIT, agenta PAIR, agenta TAP, runnera GCG, atakującego wieloobrotowego i atakującego code-switch. Każdy działa na osobnej kolejce.

5. **Zestaw ataków.** Sześć rodzin ataków: (1) zautomatyzowany jailbreak PAIR, (2) drzewo ataków TAP, (3) sufiks gradientowy GCG, (4) kodowanie ASCII / base64 / rot13, (5) wieloobrotowa persona, (6) wielojęzyczny code-switch. Raportuj wskaźnik sukcesu na rodzinę.

6. **Konstytucyjna samokrytyka.** Przygotuj 1k promptów próbujących szkodzić. Dla każdego, cel szkicuje odpowiedź. Krytyczny LLM punktuje przeciwko pisemnej konstytucji ("nie szkodzić", "cytuj dowody", "odmawiaj nielegalnych żądań"). Prompty, do których krytyk się sprzeciwia, są przepisywane; cel dostraja się na parach ulepszonych przez krytykę. Zmierz nieszkodliwość przed/po na wstrzymanej ewaluacji.

7. **Pomiar nadmiernego odmawiania.** Śledź wskaźnik fałszywie pozytywnych na zestawie nieszkodliwych promptów (np. XSTest). Cel musi pozostać pomocny na nieszkodliwych pytaniach.

8. **Punktacja CVSS.** Dla każdego udanego jailbreaka, punktuj według CVSS 4.0 (wektor ataku, złożoność, wpływ). Wyprodukuj harmonogram ujawnienia i plan łagodzenia.

9. **Automatyzacja poligonu.** Wszystko powyżej działa na cronie; wyniki zapisują się do kolejki; alarmy regresji nadmiernego odmawiania odpalają do Slack.

## Use It

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## Ship It

`outputs/skill-safety-harness.md` jest rezultatem. Produkcyjny warstwowy potok bezpieczeństwa plus powtarzalny poligon red-team z różnicami nieszkodliwości przed/po.

| Waga | Kryterium | Jak jest mierzone |
|:-:|---|---|
| 25 | Pokrycie powierzchni ataku | 6+ rodzin ataków przećwiczonych, 2+ języki |
| 20 | Kompromis prawdziwie pozytywnych / fałszywie pozytywnych | Wskaźnik blokowania ataków vs wskaźnik przepustowości nieszkodliwych XSTest |
| 20 | Różnica samokrytyki | Nieszkodliwość przed/po na wstrzymanej ewaluacji |
| 20 | Dokumentacja i ujawnienie | Wyniki z punktacją CVSS z harmonogramem |
| 15 | Automatyzacja i powtarzalność | Wszystko działa na cronie z alarmami |
| **100** | | |

## Ćwiczenia

1. Uruchom wtyczkę garak do wstrzykiwania promptów na chatbot RAG i porównaj wskaźnik sukcesu ataku z i bez warstwy filtra wyjścia.

2. Dodaj siódmą rodzinę ataków: pośrednie wstrzykiwanie promptów przez pobrane dokumenty. Zmierz dodatkową obronę wymaganą.

3. Zaimplementuj tryb "odmów-z-pomocą": gdy guardrail blokuje, cel oferuje bezpieczniejszą pokrewną odpowiedź zamiast płaskiej odmowy. Zmierz różnicę XSTest.

4. Luka w pokryciu wielojęzycznym: znajdź język, w którym X-Guard osiąga gorsze wyniki. Zaproponuj zestaw danych do dostrajania celujący w niego.

5. Uruchom konstytucyjną samokrytykę na modelu 30B i zmierz, czy różnica się skaluje.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|------------------------|
| Layered safety | "Obrona w głąb" | Wielokrotne guardraile na wejściu, bramce, wyjściu, HITL |
| Llama Guard 4 | "Klasyfikator bezpieczeństwa Meta" | Referencyjny klasyfikator treści wejścia/wyjścia 2026 |
| PAIR | "Agent jailbreaku" | Artykuł (Chao et al.) o odkrywaniu jailbreaków przez LLM |
| TAP | "Tree-of-Attacks" | Wariant PAIR z przeszukiwaniem drzewa |
| GCG | "Greedy coordinate gradient" | Atak sufiksów adversarialnych oparty na gradiencie |
| Constitutional self-critique | "Trenowanie w stylu Anthropic" | Cel szkicuje -> krytyk punktuje -> przepisz -> trenuj ponownie |
| XSTest | "Zestaw nieszkodliwych prób" | Benchmark do regresji nadmiernego odmawiania |
| CVSS 4.0 | "Wynik ważności" | Standardowa punktacja podatności dla wyników bezpieczeństwa |

## Dalsza lektura

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — training-time reference
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — the 2026 input/output classifier
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — image + multimodal safety
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — enterprise reference
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132-language multilingual safety
- [garak](https://github.com/NVIDIA/garak) — NVIDIA red-team toolkit
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft red-team framework
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail framework
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — jailbreak agent paper