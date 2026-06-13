# Biblioteki umiejętności i uczenie się przez całe życie (Voyager)

> Voyager (Wang i in., TMLR 2024) traktuje wykonywalny kod jako umiejętność. Umiejętności są nazwane, możliwe do wyszukania, komponowalne i udoskonalane przez informację zwrotną ze środowiska. Jest to architektura referencyjna dla umiejętności Claude Agent SDK, skillkit i wzorca biblioteki umiejętności w 2026 roku.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## Learning Objectives

- Wymień trzy komponenty Voyagera — automatyczny program nauczania, bibliotekę umiejętności, iteracyjne promptowanie — i rolę każdego z nich.
- Wyjaśnij, dlaczego Voyager czyni przestrzeń akcji kodem, a nie prymitywnymi poleceniami.
- Zaimplementuj w stdlib bibliotekę umiejętności z rejestracją, wyszukiwaniem, kompozycją i udoskonalaniem napędzanym błędami.
- Odwzoruj wzór Voyagera na umiejętności Claude Agent SDK 2026 i ekosystem skillkit.

## The Problem

Agenci, którzy odbudowują każdą zdolność od zera w każdej sesji, robią trzy rzeczy źle:

1. **Marnują tokeny.** Każde zadanie ponownie wywołuje to samo rozumowanie.
2. **Tracą postępy.** Korekta nauczona w sesji A nie przenosi się do sesji B.
3. **Nie radzą sobie z kompozycją długiego horyzontu.** Złożone zadania wymagają hierarchii zdolności; jednorazowe prompty nie są w stanie ich wyrazić.

Odpowiedź Voyagera: traktuj każdą wielokrotnego użytku zdolność jako nazwany fragment kodu przechowywany w bibliotece, możliwy do wyszukania przez podobieństwo, komponowalny z innymi umiejętnościami i udoskonalany przez informację zwrotną z wykonania.

## The Concept

### Trzy komponenty

Voyager (arXiv:2305.16291) organizuje agenta wokół:

1. **Automatyczny program nauczania.** Proponujący kierowany ciekawością wybiera następne zadanie na podstawie obecnego zestawu umiejętności agenta i stanu środowiska. Eksploracja jest oddolna.
2. **Biblioteka umiejętności.** Każda umiejętność to wykonywalny kod. Nowe umiejętności są dodawane, gdy zadanie się powiedzie. Umiejętności są wyszukiwane przez podobieństwo zapytania do opisu.
3. **Mechanizm iteracyjnego promptowania.** W przypadku niepowodzenia agent otrzymuje błędy wykonania, informację zwrotną ze środowiska i wynik samo-weryfikacji, a następnie udoskonala umiejętność.

Ewaluacja w Minecrafcie (Wang i in., 2024): 3.3x więcej unikalnych przedmiotów, 8.5x szybsze kamienne narzędzia, 6.4x szybsze żelazne narzędzia, 2.3x dłuższa nawigacja po mapie w porównaniu do linii bazowych. Liczby są specyficzne dla Minecrafta, ale wzór się przenosi.

### Przestrzeń akcji = kod

Większość agentów emituje prymitywne polecenia. Voyager emituje funkcje JavaScript. Umiejętność to:

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Złożona z pod-umiejętności. Przechowywana z kluczem opisu i embeddingu. Wyszukiwana jako program, nie prompt.

Jest to umiejętność Claude Agent SDK 2026: nazwany, możliwy do wyszukania fragment kodu plus instrukcje, które agent ładuje na żądanie.

### Wyszukiwanie umiejętności

Nowe zadanie "zrób diamentowy kilof." Agent:

1. Embeduje opis zadania.
2. Przepytuje bibliotekę umiejętności o top-k podobnych umiejętności.
3. Pobiera `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable` itd.
4. Komponuje nową umiejętność z pobranych prymitywów + nowej logiki.

Jest to wzór, który implementują zasoby MCP (Faza 13) i umiejętności Agent SDK: wyszukiwanie po powierzchni wiedzy/kodu, ograniczone do bieżącego zadania.

### Udoskonalanie iteracyjne

Pętla informacji zwrotnej Voyagera:

1. Agent pisze umiejętność.
2. Umiejętność jest uruchamiana w środowisku.
3. Zwracany jest jeden z trzech sygnałów: `success`, `error` (ze śladem stosu), `self-verification failure`.
4. Agent przepisuje umiejętność, używając sygnału jako kontekstu.
5. Pętla aż do sukcesu lub maksymalnej liczby rund.

Jest to Self-Refine (Lekcja 05) zastosowane do generowania kodu z weryfikacją ugruntowaną w środowisku. CRITIC (Lekcja 05) to ten sam wzór z zewnętrznymi narzędziami jako weryfikatorem.

### Program nauczania i eksploracja

Moduł programu nauczania Voyagera proponuje zadania takie jak "zbuduj schronienie w pobliżu jeziora" na podstawie tego, co agent ma i czego jeszcze nie zrobił. Proponujący używa stanu środowiska + inwentarza umiejętności, aby wybrać zadanie tuż powyżej obecnych możliwości — słodki punkt eksploracji.

Dla agentów produkcyjnych przekłada się to na operator "czego brakuje": biorąc pod uwagę obecną bibliotekę umiejętności i domenę, jakich umiejętności jeszcze nie pokrywamy? Zespoły zazwyczaj implementują to ręcznie jako przegląd programu nauczania.

### Gdzie ten wzór zawodzi

- **Gnicie biblioteki umiejętności.** Ta sama umiejętność dodana 10 razy z nieco innymi opisami. Dodaj deduplikację przy zapisie; wyszukiwanie zwraca tylko jedną.
- **Dryf komponowanej umiejętności.** Umiejętność nadrzędna zależy od podrzędnej, która została udoskonalona. Wersjonuj umiejętności; umiejętność nadrzędna przypięta do v1 nie magicznie przejmuje v3.
- **Jakość wyszukiwania.** Wyszukiwanie wektorowe po opisach umiejętności degraduje się, gdy biblioteka przekroczy kilkaset pozycji. Uzupełnij filtrami tagów i twardymi ograniczeniami ("tylko umiejętności z `category=tooling`").

## Build It

`code/main.py` implementuje w stdlib bibliotekę umiejętności:

- `Skill` — nazwa, opis, kod (jako string), wersja, tagi, zależności.
- `SkillLibrary` — rejestracja, wyszukiwanie (nakładanie tokenów), kompozycja (sortowanie topologiczne zależności) i udoskonalanie (zwiększenie wersji przy aktualizacji).
- Skryptowany agent, który rejestruje trzy prymitywne umiejętności, komponuje czwartą, napotyka błąd i udoskonala.

Uruchom:

```
python3 code/main.py
```

Ślad pokazuje zapisy do biblioteki, wyszukiwanie, kompozycję, nieudane wykonanie i udoskonalenie do v2 — pętlę Voyagera od końca do końca.

## Use It

- **Umiejętności Claude Agent SDK** (Anthropic) — referencja 2026: każda umiejętność ma opis, kod i instrukcje; ładowana na żądanie podczas sesji agenta.
- **skillkit** (npm: skillkit) — między-agentowe zarządzanie umiejętnościami dla 32+ agentów kodowania AI.
- **Niestandardowe biblioteki umiejętności** — specyficzne dla domeny (umiejętności SQL dla agentów danych, umiejętności Terraform dla agentów infrastruktury). Wzór Voyagera skaluje się w dół.
- **OpenAI Agents SDK `tools`** — na niskim końcu; każde narzędzie to lekka umiejętność.

## Ship It

`outputs/skill-skill-library.md` generuje bibliotekę umiejętności w kształcie Voyagera z rejestracją, wyszukiwaniem, wersjonowaniem i udoskonalaniem dla dowolnego środowiska docelowego.

## Exercises

1. Dodaj detektor cykli zależności do `compose()`. Co się dzieje, gdy umiejętność A zależy od B, która zależy od A? Błąd vs ostrzeżenie?
2. Zaimplementuj przypinanie wersji dla poszczególnych umiejętności. Gdy umiejętność nadrzędna komponuje podrzędną `crafting@1`, udoskonalenie do `crafting@2` nie może po cichu uaktualnić nadrzędnej.
3. Zastąp wyszukiwanie przez nakładanie tokenów embeddingami sentence-transformers (lub implementacją BM25 w stdlib). Zmierz retrieval@5 na zabawkowej bibliotece 50 umiejętności.
4. Dodaj agenta "programu nauczania": biorąc pod uwagę obecną bibliotekę i opis domeny, zaproponuj 5 brakujących umiejętności. Wywołuj co tydzień.
5. Przeczytaj dokumentację umiejętności Claude Agent SDK od Anthropic. Przenieś zabawkową bibliotekę na schemat umiejętności SDK. Co zmienia się w kwestii odkrywalności?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Umiejętność | "Zdolność wielokrotnego użytku" | Nazwany fragment kodu + opis, możliwy do wyszukania przez podobieństwo |
| Biblioteka umiejętności | "Pamięć agenta jak-to-zrobić" | Trwały magazyn umiejętności, przeszukiwalny i komponowalny |
| Program nauczania | "Proponujący zadania" | Oddolny generator celów napędzany bieżącą luką w zdolnościach |
| Kompozycja | "DAG umiejętności" | Umiejętności wywołujące umiejętności; sortowane topologicznie przy wykonaniu |
| Udoskonalanie iteracyjne | "Pętla samokorekty" | Informacja zwrotna ze środowiska + błędy + samo-weryfikacja składane do następnej wersji |
| Przestrzeń akcji jako kod | "Akcje programistyczne" | Emituj funkcje, nie prymitywne polecenia, dla czasowo rozszerzonego zachowania |
| Dedup przy zapisie | "Kolaps umiejętności" | Prawie-duplikaty opisów zwijają się do jednej kanonicznej umiejętności |

## Further Reading

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — oryginalny artykuł o bibliotece umiejętności
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — umiejętności jako produktyzacja w 2026
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — umiejętności i podagenci w praktyce
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — pętla udoskonalania leżąca u podstaw Voyagera