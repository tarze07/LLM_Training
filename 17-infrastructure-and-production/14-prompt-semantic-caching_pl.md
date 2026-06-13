# Buforowanie promptów i ekonomika buforowania semantycznego

> **Zrzut cen z 2026-04.** Liczby poniżej odzwierciedlają stawki dostawców w momencie publikacji tej lekcji; zweryfikuj z połączonymi dokumentami przed zacytowaniem ich dalej.

> Buforowanie działa na dwóch warstwach. L2 (na poziomie dostawcy) buforowanie promptów/prefiksów ponownie używa attention KV dla powtarzających się prefiksów — buforowanie promptów Anthropic reklamuje do 90% redukcji kosztów i 85% redukcji latencji na długich promptach; dla Claude 3.5 Sonnet odczyty z pamięci podręcznej to $0,30/M vs $3,00/M świeże z 5-minutowym TTL i 2x premią za zapis dla opcji 1-godzinnego TTL (docs.anthropic.com, 2026-04). Buforowanie promptów OpenAI stosuje się automatycznie dla promptów ≥1024 tokenów i wycenia buforowane wejście zgrubnie z 90% zniżką vs świeże (platform.openai.com, 2026-04); dokładna stawka buforowana na model zależy od aktualnej tabeli stawek. L1 (na poziomie aplikacji) buforowanie semantyczne pomija LLM całkowicie przy trafieniu podobieństwa embeddingu. „95% dokładności" dostawcy odnosi się do poprawności dopasowania, a nie współczynnika trafień — zgłaszane produkcyjne współczynniki trafień wahają się od 10% (otwarty czat) do 70% (strukturyzowane FAQ); żaden dostawca nie publikuje oficjalnej linii bazowej, więc traktuj je jako telemetrię społeczności, a nie gwarancje. Pułapki produkcyjne: parallelizacja zabija buforowanie (N równoległych żądań wysłanych przed pierwszym zapisem do pamięci podręcznej może wielokrotnie zwiększyć wydatki), a dynamiczna treść wewnątrz prefiksu uniemożliwia trafienia w pamięci podręcznej. ProjectDiscovery zgłosiło przejście z 7% do 74% współczynnika trafień (2025-11) przez przeniesienie dynamicznego tekstu poza możliwy do buforowania prefiks.

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Learning Objectives

- Odróżnij L2 buforowanie promptów/prefiksów (ponowne użycie KV u dostawcy) od L1 buforowania semantycznego (pominięcie LLM na podobnych promptach).
- Wyjaśnij jawne oznaczanie `cache_control` Anthropic i dwie opcje TTL (5-min vs 1-godzina) z ich mnożnikami cen.
- Oblicz oczekiwane miesięczne oszczędności, mając współczynnik trafień, mieszankę promptów/odpowiedzi i ceny tokenów.
- Nazwij antywzorzec parallelizacji, który zawyża rachunki 5-10x, i antywzorzec dynamicznej treści, który załamuje współczynnik trafień.

## The Problem

Dodajesz buforowanie promptów do swojej usługi RAG. Rachunek pozostaje taki sam. Mierzysz współczynnik trafień; wynosi 7%. Twoje prompty wyglądają na statyczne, ale nie są — prompt systemowy zawiera bieżącą datę sformatowaną co minutę, ID żądania i losową zmianę kolejności przykładów dla różnorodności. Każde żądanie zapisuje nowy wpis w pamięci podręcznej, odczytuje zero.

Osobno, Twój agent uruchamia dziesięć równoległych wywołań narzędzi na pytanie użytkownika. Wszystkie dziesięć dociera do dostawcy, zanim pierwszy zapis do pamięci podręcznej zostanie ukończony. Dziesięć zapisów, zero odczytów. Twój rachunek jest 5-10x wyższy, niż „z buforowaniem" miał kosztować.

Buforowanie to protokół, a nie flaga. Dwie warstwy, dwa różne tryby awarii.

## The Concept

### L2 — buforowanie promptów/prefiksów u dostawcy

Dostawca przechowuje attention KV dla możliwego do buforowania prefiksu i ponownie go używa przy następnym żądaniu, które pasuje do prefiksu. Płacisz koszt zapisu raz, odczyty prawie darmowe.

**Anthropic (seria Claude 3.5 / 3.7 / 4)**: jawny znacznik `cache_control` w żądaniu. Oznaczasz, które bloki są możliwe do buforowania. TTL: 5-minut (zapis kosztuje 1,25x podstawy) lub 1-godzinny (zapis kosztuje 2x podstawy). Odczyty z pamięci podręcznej: $0,30/M na Claude 3.5 Sonnet vs $3,00/M świeże — 10x taniej (docs.anthropic.com, stan na 2026-04). Stawki różnią się w zależności od modelu (Opus/Haiku publikowane osobno); zawsze sprawdzaj krzyżowo żywą stronę cenową.

**OpenAI**: automatyczne buforowanie dla promptów ≥1024 tokenów (platform.openai.com, 2026-04). Bez jawnej flagi. Buforowane wejście jest zgrubnie 10x tańsze niż świeże na obecnych stawkach gpt-4o/gpt-5. Ani dokumentacja, ani notatki wydania nie publikują oficjalnej linii bazowej współczynnika trafień; raporty społeczności oscylują wokół 30-60% przy starannym projekcie promptów. Monitoruj `usage.cached_tokens`, aby zmierzyć własne.

**Google (Gemini)**: buforowanie kontekstu przez jawne API; kontekst 1M tokenów oznacza, że buforowanie opłaca się jeszcze bardziej.

**Samo-hostowane (vLLM, SGLang)**: Faza 17 · 06 obejmuje RadixAttention — ten sam wzorzec na własnym sprzęcie.

### L1 — buforowanie semantyczne na poziomie aplikacji

Przed wywołaniem LLM w ogóle, haszuj prompt, osadź go i poszukaj podobnego buforowanego żądania (podobieństwo cosinusowe powyżej progu, typowo 0,95+). Przy trafieniu, zwróć buforowaną odpowiedź. Przy chybieniu, wywołaj LLM i buforuj wynik.

Otwarte źródło: Redis Vector Similarity, GPTCache, Qdrant. Komercyjne: Portkey Cache, Helicone Cache.

Twierdzenia dostawców o dokładności odnoszą się do tego, jak często zwrócona buforowana odpowiedź była semantycznie odpowiednia — nie jak często trafiasz. Produkcyjne współczynniki trafień:

- Otwarty czat: 10-15%.
- Strukturyzowane FAQ / wsparcie: 40-70%.
- Pytania kodowe: 20-30% (małe warianty zabijają trafienia).
- Agenci głosowi powtarzający prompty: 50-80% (normalizacja głosu ustalony zestaw).

### Antywzorzec parallelizacji

Twój agent wykonuje 10 wywołań narzędzi równolegle. Wszystkie 10 ma ten sam 4K-tokenowy prompt systemowy. Zapis do pamięci podręcznej Anthropic jest na żądanie; pierwszy zapis kończy się około 300 ms po tym, jak dostawca zobaczy prompt. Żądania 2-10 docierają w tym samym oknie milisekundowym i każde widzi chybienie pamięci podręcznej. Płacisz 10 premii za zapis, 0 zniżek za odczyt.

Poprawka: batch z sekwencyjnym-first — wykonaj żądanie 1 samodzielnie, a następnie wyślij 2-10, gdy pamięć podręczna po 1 jest już wypełniona. Dodaje 300 ms do pierwszego wywołania narzędzia; oszczędza 5-10x rachunku.

### Antywzorzec dynamicznej treści

Twój prompt systemowy wygląda tak:

```
Jesteś pomocnym asystentem. Aktualny czas to 14:32:17.
ID użytkownika: abc123. Dzisiaj jest wtorek...
```

Każde żądanie jest unikalne. Każde żądanie zapisuje. Zero trafień.

Poprawka: przenieś wszystko naprawdę statyczne do możliwego do buforowania prefiksu; dołącz dynamiczną treść po granicy pamięci podręcznej:

```
[do buforowania]
Jesteś pomocnym asystentem. [zasady, przykłady, instrukcje]
[/do buforowania]
[dynamiczne, niebuforowane]
Aktualny czas: 14:32:17. Użytkownik: abc123.
```

ProjectDiscovery przeszło z 7% do 74% współczynnika trafień w ten sposób i opublikowało anatomię.

### Układaj wsadowo + buforuj dla obciążeń nocnych

Batch API (Faza 17 · 15) daje 50% zniżki przy 24-godzinnym czasie realizacji. Buforowane wejście na wierzchu daje ~10x na dodatek. Nocne klasyfikacje, etykietowanie i generowanie raportów mogą spaść do ~10% kosztu synchronicznego niebuforowanego przez układanie.

### Liczby, które powinieneś zapamiętać

Punkty cenowe są zebrane 2026-04 z połączonych dokumentów dostawców i dryfują co kilka miesięcy — sprawdź ponownie przed poleganiem na nich.

- Buforowany odczyt Anthropic: $0,30/M na Claude 3.5 Sonnet, zgrubnie 10x taniej niż świeże wejście (docs.anthropic.com).
- Premia za zapis do pamięci podręcznej Anthropic: 1,25x (TTL 5-min) lub 2x (TTL 1-godzina).
- Auto-buforowanie OpenAI: dotyczy promptów ≥1024 tokenów; buforowane wejście wycenione zgrubnie na 10% świeżego wejścia na obecnych stawkach (platform.openai.com).
- Współczynnik trafień pamięci podręcznej semantycznej (zgłaszany przez społeczność): ~10% otwarty czat; do ~70% strukturyzowane FAQ. Nie jest udokumentowaną linią bazową dostawcy.
- ProjectDiscovery: 7% → 74% współczynnik trafień przez przeniesienie dynamicznej treści poza prefiks (blog projektu, 2025-11).
- Antywzorzec parallelizacji: typowe raporty 5-10x zawyżenia rachunku, gdy N równoległych żądań trafia na pierwszy zapis do pamięci podręcznej.

## Use It

`code/main.py` symuluje buforowanie L1 + L2 na mieszanych obciążeniach. Raportuje współczynniki trafień, rachunek i pokazuje karę parallelizacji.

## Ship It

Ta lekcja produkuje `outputs/skill-cache-auditor.md`. Biorąc szablon promptu i ruch, audytuje możliwość buforowania i rekomenduje restrukturyzację.

## Exercises

1. Uruchom `code/main.py`. Przełącz flagę parallelizacji. O ile zmienia się rachunek?
2. Twój prompt systemowy ma datę. Przenieś ją na zewnątrz. Pokaż matematykę współczynnika trafień przed/po.
3. Oblicz próg rentowności dla TTL 1-godzina (2x zapis) vs TTL 5-minut (1,25x zapis), mając swoją szybkość przybywania żądań.
4. Buforowanie semantyczne przy progu 0,95 trafia 20%. Przy 0,85 trafia 50%, ale widzisz niepoprawne buforowane odpowiedzi. Wybierz właściwy próg i uzasadnij.
5. Batchujesz 10 równoległych podzapytań na pytanie użytkownika. Przepisz dla przyjazności buforowaniu bez dodawania latencji od końca do końca.

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| L2 prompt cache | „bufor prefiksu" | Dostawca przechowuje KV dla powtarzającego się prefiksu |
| `cache_control` | „znacznik pamięci podręcznej Anthropic" | Jawny atrybut oznaczający możliwe do buforowania bloki |
| Cache write premium | „podatek zapisu" | Dodatkowy koszt za pierwszy zapis do pamięci podręcznej (1,25x lub 2x) |
| L1 semantic cache | „bufor embeddingu" | Poziom aplikacji: haszuj i osadź przed wywołaniem LLM |
| GPTCache | „biblioteka buforowania LLM" | Popularna biblioteka OSS L1 cache |
| Cache hit rate | „trafienia / wszystkie" | Frakcja żądań obsłużonych z pamięci podręcznej |
| Parallelization anti-pattern | „pułapka N-zapisów" | N równoległych żądań chybia pamięć podręczną N razy |
| Dynamic content trap | „pułapka czasu w prompcie" | Dynamiczne bajty w prefiksie zabijają współczynnik trafień |
| RadixAttention | „bufor wewnątrz repliki" | Implementacja buforowania prefiksów SGLang |

## Further Reading

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — oficjalna semantyka `cache_control` i TTL.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — automatyczne zachowanie buforowania i kwalifikowalność.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)

(End of file - total 134 lines)