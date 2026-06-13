# Benchmarki: WebArena i OSWorld

> WebArena testuje zdolności agentów internetowych w czterech samodzielnie hostowanych aplikacjach. OSWorld testuje zdolności agentów desktopowych w Ubuntu, Windows, macOS. W momencie wydania (2023–2024) oba pokazywały dużą lukę między najlepszymi agentami a ludźmi. Luka się zmniejsza; tryby awarii się nie zmieniły.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 minutes

## Learning Objectives

- Opisz cztery samodzielnie hostowane aplikacje WebArena i dlaczego ewaluacja oparta na wykonaniu ma znaczenie.
- Wyjaśnij, dlaczego OSWorld używa rzeczywistych zrzutów ekranu OS zamiast API dostępności.
- Wymień dwa podstawowe tryby awarii OSWorld: ugruntowanie GUI i wiedza operacyjna.
- Podsumuj, co OSWorld-G i OSWorld-Human dodają do podstawowego benchmarku.

## Problem

Agenci-generaliści mogą wywoływać narzędzia. Czy potrafią sterować przeglądarką przez 20 kliknięć, aby sfinalizować zakupy? Czy potrafią skonfigurować Linuksa używając tylko klawiatury i myszy? To są pytania, na które odpowiadają WebArena i OSWorld.

## Koncepcja

### WebArena (Zhou i in., ICLR 2024)

- 812 długoterminowych zadań w czterech samodzielnie hostowanych aplikacjach internetowych: stronie sklepowej, forum, narzędziu deweloperskim w stylu GitLab, firmowym CMS.
- Plus narzędzia: mapa, kalkulator, notatnik.
- Ewaluacja oparta na wykonaniu przez API gym — czy zamówienie zostało złożone, czy zgłoszenie zostało zamknięte, czy strona CMS została zaktualizowana?
- W momencie wydania: najlepszy agent GPT-4 osiągnął 14,41% sukcesu vs ludzkie 78,24%.

Samodzielne hostowanie ma znaczenie — benchmark nie jest zawodny, ponieważ docelowe aplikacje są przypięte i powtarzalne.

### Rozszerzenia

- **VisualWebArena** — zadania ugruntowane wizualnie, gdzie sukces zależy od interpretacji obrazów (zrzuty ekranu jako obserwacje pierwszej klasy).
- **TheAgentCompany** (grudzień 2024) — dodaje terminal + kodowanie; bardziej przypomina rzeczywiste środowisko pracy zdalnej.

### OSWorld (Xie i in., NeurIPS 2024)

- 369 rzeczywistych zadań komputerowych na Ubuntu, Windows, macOS.
- Swobodna kontrola klawiatury i myszy rzeczywistych aplikacji.
- Zrzuty ekranu 1920×1080 jako obserwacja.
- W momencie wydania: najlepszy model 12,24% vs ludzkie 72,36%.

### Podstawowe tryby awarii

1. **Ugruntowanie GUI.** Mapowanie piksel → element. Modele mają trudności z niezawodną lokalizacją elementów UI w 1920×1080.
2. **Wiedza operacyjna.** Które menu ma ustawienie, który skrót klawiaturowy, które okno preferencji. Wiedza ogonowa, którą ludzie budują przez lata.

### Kontynuacje

- **OSWorld-G** — zestaw ugruntowujący 564 próbki + zestaw treningowy Jedi. Dekomponuje ugruntowanie od planowania, aby można je było mierzyć osobno.
- **OSWorld-Human** — ręcznie wyselekcjonowane złote trajektorie działań. Pokazuje, że najlepsi agenci używają 1,4–2,7× więcej kroków niż potrzeba (luka wydajności trajektorii).

### Dlaczego to ma znaczenie

Claude computer use, OpenAI CUA, Gemini 2.5 Computer Use (Lekcja 21) wszystkie trenują na obciążeniach ukształtowanych przez WebArenę i OSWorld. Benchmarki są celem; modele produkcyjne są wysłaną odpowiedzią.

### Gdzie benchmarkowanie zawodzi

- **Ewaluacje tylko na zrzutach ekranu.** OSWorld jest napędzany zrzutami ekranu; ewaluacja agenta używającego DOM lub API dostępności na OSWorld pomija wyzwanie ugruntowania.
- **Ignorowanie długości trajektorii.** Ocenianie tylko wskaźnika sukcesu pomija 1,4–2,7× nieefektywność kroków, którą ujawnia OSWorld-Human.
- **Nieaktualne samodzielnie hostowane aplikacje.** Aplikacje WebAreny przypinają konkretne wersje; aktualizacja bez ponownej kurateli psuje porównywalność.

## Build It

`code/main.py` implementuje zabawkowy szkielet agenta internetowego:

- Minimalna maszyna stanów „aplikacji sklepowej": list_items, add_to_cart, checkout.
- Złote trajektorie dla 3 zadań.
- Skryptowany agent, który próbuje każde zadanie.
- Ewaluator oparty na wykonaniu (sprawdzenie stanu) i metryka wydajności trajektorii (kroki vs złote).

Uruchom:

```
python3 code/main.py
```

Wynik: wskaźnik sukcesu na zadanie i wydajność trajektorii, odzwierciedlające metodologię OSWorld-Human.

## Use It

- **WebArena Verified** samodzielnie hostowana na wewnętrznym klastrze do ciągłej ewaluacji.
- **OSWorld** w flocie maszyn wirtualnych dla agentów desktopowych.
- **Agenci computer use** (Lekcja 21) — Claude, OpenAI CUA, Gemini — wszyscy trenują na obciążeniach takich jak te.
- **Własne przepływy produktowe** — przechwytuj złote trajektorie dla swoich 20 najważniejszych zadań; uruchamiaj agentów co tydzień.

## Ship It

`outputs/skill-web-desktop-harness.md` buduje szkielet agenta internetowego/desktopowego z ewaluacją opartą na wykonaniu i metryką wydajności trajektorii.

## Ćwiczenia

1. Rozszerz zabawkowy szkielet o drugą aplikację (forum). Napisz 3 zadania plus złote trajektorie.
2. Dodaj raportowanie wydajności trajektorii na zadanie. Na twojej zabawce, czy agent jest 1×, 2× czy 3× ponad złotą?
3. Zaimplementuj narzędzie „rozpraszające" — takie, którego złota trajektoria nigdy nie używa. Czy skryptowany agent ulega pokusie?
4. Przeczytaj OSWorld-G. Jak oddzieliłbyś awarie ugruntowania od awarii planowania we własnych ewaluacjach?
5. Przeczytaj plik README aplikacji WebArena. Co się psuje, gdy uaktualnisz jedną z przypiętych wersji aplikacji?

## Key Terms

| Termin | Co ludzie mówią | Co naprawdę znaczy |
|------|----------------|------------------------|
| WebArena | „Benchmark agentów internetowych" | 812 zadań w 4 samodzielnie hostowanych aplikacjach; ewaluacja w stylu gym |
| VisualWebArena | „Wizualna WebArena" | WebArena ugruntowana wizualnie; zrzuty ekranu są obserwacjami |
| OSWorld | „Benchmark agentów desktopowych" | 369 zadań na rzeczywistym Ubuntu/Windows/macOS |
| Ugruntowanie GUI | „Mapowanie piksel-na-element" | Model lokalizujący elementy UI w 1920×1080 |
| Wiedza operacyjna | „Znajomość OS" | Które menu, który skrót, które okno preferencji |
| OSWorld-G | „Zestaw ugruntowujący" | 564 próbek tylko ugruntowania + zestaw treningowy |
| OSWorld-Human | „Złote trajektorie" | Ręczne sekwencje działań eksperta do pomiaru wydajności |
| Wydajność trajektorii | „Kroki ponad złotą" | Liczba kroków agenta podzielona przez ludzkie minimum |

## Dalsza lektura

- [Zhou i in., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — benchmark internetowy czterech aplikacji
- [Xie i in., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — benchmark desktopowy między systemami
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — zdolność ukształtowana przez benchmarki Claude'a
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — wyniki OSWorld i WebArena