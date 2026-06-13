# Zarządzane platformy LLM — Bedrock, Vertex AI, Azure OpenAI

> Trzej hiperskalerzy, trzy odrębne strategie. AWS Bedrock to marketplace modeli — Claude, Llama, Titan, Stability, Cohere za jednym API. Azure OpenAI to ekskluzywna współpraca z OpenAI plus Provisioned Throughput Units (PTU) dla dedykowanej przepustowości. Vertex AI to platforma stawiająca na Gemini z najlepszym wsparciem dla długiego kontekstu i multimodalności. W 2026 roku Artificial Analysis mierzy Azure OpenAI z medianą ~50 ms, a Bedrock z ~75 ms na odpowiednikach Llama 3.1 405B — PTU tłumaczą różnicę, ponieważ dedykowana przepustowość bije na głowę współdzieloną na żądanie. Zasadą decyzyjną nie jest „który jest najszybszy", ale „który katalog modeli i powierzchnia FinOps pasują do mojego produktu". Ta lekcja uczy wybierać z zapisanymi kompromisami, a nie na podstawie przeczuć.

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## Learning Objectives

- Wymień trzy strategie platform (marketplace vs ekskluzywna vs stawiająca na Gemini) i dopasuj każdą do przypadku użycia produktu.
- Wyjaśnij, co kupują Provisioned Throughput Units (PTU) w Azure OpenAI i dlaczego Bedrock na żądanie typowo jest ~25 ms wolniejszy w skali 405B.
- Narysuj diagram powierzchni atrybucji FinOps dla każdej platformy (Bedrock Application Inference Profiles vs Vertex projekt-na-zespół vs Azure zakresy + rezerwacje PTU).
- Zapisz politykę „minimum dwóch dostawców" i wyjaśnij, dlaczego uzależnienie od jednego dostawcy jest kosztownym błędem w 2026 roku.

## The Problem

Wybrałeś Claude 3.7 Sonnet dla swojego produktu. Teraz musisz go serwować. Możesz wywołać bezpośrednio API Anthropic, lub wywołać go przez AWS Bedrock, lub przejść przez bramę. Bezpośrednie API jest najprostsze; Bedrock dodaje BAA, punkty końcowe VPC, IAM i atrybucję CloudWatch. Brama dodaje failover, ujednolicone rozliczenia i limity szybkości u dostawców.

Głębsze pytanie dotyczy katalogu. Jeśli potrzebujesz Claude, Llama i Gemini w tym samym produkcie, nie możesz kupić ich wszystkich z jednego miejsca, chyba że tym miejscem jest Bedrock plus Vertex plus Azure OpenAI jednocześnie. Hiperskalerzy nie są wymienni — każdy postawił inną stawkę na to, kto kontroluje warstwę modeli.

Ta lekcja mapuje trzy stawki, różnicę w latencji, różnicę w FinOps i ryzyko uzależnienia.

## The Concept

### Trzy strategie

**AWS Bedrock** — marketplace. Claude (Anthropic), Llama (Meta), Titan (AWS pierwszej strony), Stability (obrazy), Cohere (embeddingi), Mistral, plus podkatalogi obrazów i embeddingów. Jedno API, jedna powierzchnia IAM, jeden eksport CloudWatch. Zakład Bedrock: klienci chcą opcjonalności bardziej niż pojedynczego modelu.

**Azure OpenAI** — ekskluzywne partnerstwo. Otrzymujesz GPT-4 / 4o / 5 / o-series, DALL·E, Whisper i fine-tuning modeli OpenAI w centrach danych Azure. Żadnych modeli spoza OpenAI w katalogu „Azure OpenAI Service" — te trafiają do Azure AI Foundry (oddzielny produkt). Zakład Azure: OpenAI pozostaje na froncie, a klienci chcą korporacyjnych kontroli dla tej konkretnej relacji.

**Vertex AI** — Gemini na pierwszym miejscu, wszystko inne na drugim. Gemini 1.5 / 2.0 / 2.5 Flash i Pro, plus Model Garden (strony trzecie). Zakład Vertex: multimodalny długi kontekst — 1-milionowy kontekst Gemini jest wyróżnikiem.

### Różnica w latencji na dużą skalę

Artificial Analysis prowadzi ciągłe benchmarki. Na równoważnych wdrożeniach Llama 3.1 405B (współdzielone na żądanie), mediana latencji pierwszego tokena Azure OpenAI wynosi około 50 ms; Bedrock około 75 ms. Różnica nie jest porażką AWS — to różnica w modelu pojemności. Azure sprzedaje PTU (Provisioned Throughput Units), które rezerwują pojemność GPU dla Twojego dzierżawcy. Odpowiednik Bedrock (Provisioned Throughput) istnieje, ale zaczyna się od ~21 USD za godzinę za jednostkę, a większość klientów pozostaje przy współdzielonym na żądanie.

Współdzielona pojemność na żądanie konkuruje z ruchem każdego innego klienta. Dedykowana pojemność nie. Jeśli SLA Twojego produktu wymaga TTFT < 100 ms na P99, albo kupujesz PTU w Azure, albo kupujesz Bedrock Provisioned Throughput, albo akceptujesz domyślną wariancję.

### Ekonomika Provisioned Throughput

Azure PTU: zarezerwowany blok mocy obliczeniowej do inferencji. Do ~70% oszczędności w porównaniu do na żądanie dla przewidywalnych obciążeń. Koszt stały za godzinę niezależnie od ruchu — płacisz za rezerwację nawet gdy jest bezczynna. Próg rentowności wynosi zwykle około 40-60% stałego wykorzystania.

Bedrock Provisioned Throughput: 21-50 USD za godzinę, w zależności od modelu i regionu. Podobna matematyka — próg rentowności około połowy szczytowego wykorzystania. Wymagane zobowiązanie miesięczne.

Vertex provisioned capacity jest sprzedawana na SKU Gemini; ceny różnią się w zależności od modelu i regionu i są mniej publicznie reklamowane.

### Powierzchnia FinOps — prawdziwy wyróżnik

**Bedrock Application Inference Profiles** to najczystsza atrybucja na rynku. Oznacz profil tagami `team`, `product`, `feature`; kieruj wszystkie wywołania modeli przez niego; CloudWatch rozbija koszt na profil bez potrzeby post-processingu. Dodane w 2025, wciąż najbardziej granularne natywne rozwiązanie hiperskalera.

**Vertex** atrybucja to projekt-na-zespół plus etykiety wszędzie. Modelujesz każdy zespół jako projekt GCP, umieszczasz etykiety na każdym zasobie i używasz BigQuery Billing Export + DataStudio do podsumowań. Więcej pracy, ale BigQuery daje dowolne SQL na danych kosztowych.

**Azure** opiera się na zakresach subskrypcji/grup zasobów plus tagach, z rezerwacjami PTU jako pierwszorzędnym obiektem kosztowym. Tagi są dziedziczone z grup zasobów, a nie żądań, więc atrybucja na żądanie wymaga niestandardowych metryk Application Insights lub bramy, która dodaje nagłówki.

Wzorzec: Bedrock jest najczystszy natywnie, Vertex najbardziej elastyczny przez BigQuery, Azure najbardziej nieprzezroczysty, chyba że przeprowadzisz instrumentację.

### Ryzyko uzależnienia w 2026

Zaangażowanie u jednego hiperskalera było w porządku, gdy dominował jeden model. W 2026 roku front zmienia się co miesiąc — Claude 3.7 jednego kwartału, Gemini 2.5 następnego, GPT-5 kolejnego. Zablokowanie się na jednej platformie blokuje dostęp do dwóch trzecich frontu.

Wzorzec przyjęty przez dobrze działające zespoły: minimum dwóch dostawców dla każdego krytycznego wywołania LLM. Bedrock plus Azure OpenAI to popularna para — Claude od jednego, GPT od drugiego, failover między nimi, ta sama brama. Wzrost kosztów jest znikomy, ponieważ brama optymalizuje routing; wzrost dostępności podczas awarii (jak incydent Azure OpenAI ze stycznia 2025, awaria AWS us-east-1) jest decydujący.

### Rezydencja danych, BAA i branże regulowane

Bedrock: BAA w większości regionów; punkty końcowe VPC; zabezpieczenia. Typowy domyślny wybór w fintech.
Azure OpenAI: HIPAA, SOC 2, ISO 27001; rezydencja danych w UE; domyślny wybór dla przedsiębiorstw regulowanych.
Vertex: HIPAA, GDPR, rezydencja danych na region; stos zgodności Google Cloud.

Wszystkie trzy spełniają podstawowe wymagania. Różnice dotyczą polityk przechowywania danych, sposobu obsługi logów i tego, czy monitorowanie nadużyć czyta Twój ruch (domyślnie włączone u większości; możliwość wyłączenia dla przedsiębiorstw).

### Liczby, które powinieneś zapamiętać

- Mediana TTFT Azure OpenAI na odpowiednikach Llama 3.1 405B: ~50 ms (z PTU).
- Mediana TTFT Bedrock na żądanie: ~75 ms.
- Bedrock Provisioned Throughput: 21-50 USD/h za jednostkę.
- Próg rentowności PTU Azure: ~40-60% stałego wykorzystania.
- Oszczędności PTU vs na żądanie przy wysokim wykorzystaniu: do 70%.

## Use It

`code/main.py` porównuje trzy platformy na syntetycznym obciążeniu — modeluje ekonomikę na żądanie vs PTU, wariancję TTFT i wierność atrybucji kosztów. Uruchom, aby zobaczyć, gdzie PTU się opłacają i gdzie szerokość katalogu modeli przewyższa różnicę w TTFT.

## Ship It

Ta lekcja produkuje `outputs/skill-managed-platform-picker.md`. Biorąc profil obciążenia (potrzebne modele, SLA TTFT, dzienny wolumen, wymagania zgodności), rekomenduje platformę podstawową, zapasową i plan instrumentacji FinOps.

## Exercises

1. Uruchom `code/main.py`. Przy jakim stałym wykorzystaniu Azure PTU bije na żądanie dla modelu klasy 70B? Oblicz próg rentowności i porównaj z reklamowanym przedziałem 40-60%.
2. Twój produkt potrzebuje Claude 3.7 Sonnet i GPT-4o. Zaprojektuj wdrożenie u dwóch dostawców — który trafia do którego hiperskalera, jaka brama stoi z przodu, jaka jest polityka failover?
3. Regulowany klient z opieki zdrowotnej wymaga BAA, rezydencji danych w US-East i P99 TTFT poniżej 100 ms. Wybierz platformę i uzasadnij trzema konkretnymi funkcjami.
4. Odkrywasz, że Twój rachunek za Bedrock wzrósł 4-krotnie w tym miesiącu bez zmiany ruchu. Bez Application Inference Profiles, jak znalazłbyś winowajcę? Z profilami, ile to zajmuje?
5. Przeczytaj strony cenowe Azure OpenAI i Bedrock. Dla obciążenia Claude 100M tokenów/miesiąc, co jest tańsze — bezpośrednie API Anthropic, Bedrock na żądanie czy Bedrock Provisioned Throughput?

## Key Terms

| Term | Co ludzie mówią | Co to naprawdę znaczy |
|------|----------------|------------------------|
| Bedrock | „usługa LLM AWS" | Marketplace modeli od Claude, Llama, Titan, Mistral, Cohere |
| Azure OpenAI | „ChatGPT Azure" | Ekskluzywne modele OpenAI w centrach danych Azure z kontrolami korporacyjnymi |
| Vertex AI | „LLM Google" | Platforma stawiająca na Gemini z Model Garden dla modeli stron trzecich |
| PTU | „dedykowana przepustowość" | Provisioned Throughput Unit — zarezerwowane GPU do inferencji, wycenione za godzinę |
| Application Inference Profile | „tagowanie Bedrock" | Profil kosztów/użycia na produkt z tagami, natywny w CloudWatch |
| Model Garden | „katalog Vertex" | Sekcja modeli stron trzecich Vertex AI, oddzielona od Gemini |
| Two-provider minimum | „redundancja LLM" | Polityka uruchamiania każdej krytycznej ścieżki LLM u ≥2 hiperskalerów |
| BAA | „papierologia HIPAA" | Business Associate Agreement; wymagany dla PHI; zapewniany przez wszystkich trzech |
| Abuse monitoring | „obserwator logów" | Skanowanie bezpieczeństwa promptów/wyjść po stronie dostawcy; do wyłączenia w przedsiębiorstwie |

## Further Reading

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — autorytatywna tabela stawek i ceny Provisioned Throughput.
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — ekonomika PTU i tabele stawek.
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — poziomy Gemini i dopłaty Model Garden.
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — ciągłe benchmarki latencji i przepustowości u dostawców.
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — ramy decyzyjne dla przedsiębiorstw.
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — mechanika atrybucji obok siebie.

(End of file - total 118 lines)