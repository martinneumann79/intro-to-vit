# Presentationsplan: Vision Transformers på 45 minuter

Tidsplan, förslag på bilder, demomanus och svar på vanliga frågor. Faktan finns i
[presentation-info.md](presentation-info.md) och [vit-in-practice.md](vit-in-practice.md).

## Tre saker publiken ska minnas

1. **ViT gör en bild till en "mening" av 16×16-rutor** och använder sedan samma transformer som
   språkmodellerna.
2. **CNN har inbyggd kunskap om bilder och vinner när data är knapp.** ViT måste lära sig mer
   själv, men är lättare att skala upp och tål brus bättre än en klassisk CNN. Designval och
   träningsrecept spelar minst lika stor roll som etiketten "transformer" eller "CNN".
3. **Nästan alla stora bildmodeller i dag bygger på ViT**, men CNN lever kvar där det måste gå
   snabbt på liten hårdvara.

## Tidsplan

| Tid (min) | Avsnitt | Innehåll | Underlag |
|---|---|---|---|
| 0–3 | Inledning | Fråga publiken vad ChatGPT och en modell som hittar cancer i vävnadsprover har gemensamt. Svaret är transformern. Visa agendan | |
| 3–9 | CNN | Hur en dator ser en bild. Filter som glider över bilden. Hierarkin från kanter till objekt. AlexNet 2012. Inbyggda antaganden | [§2](presentation-info.md#2-cnn-convolutional-neural-network) |
| 9–15 | Transformers | Tokens, embedding och self-attention med exemplet "djuret … det". Flera huvuden. Varför transformern tog över. Kvadratisk kostnad | [§1](presentation-info.md#1-transformers) |
| 15–22 | ViT | 16×16-rutor blir tokens, [CLS], flödesbilden. Resultatet från 2020: förlorar på lite data, vinner med 300 miljoner bilder. DeiT, Swin, CLIP, SAM, DINO | [§3](presentation-info.md#3-vision-transformer-vit) |
| 22–27 | Skillnader, för- och nackdelar | Jämförelsetabellen. Inbyggda antaganden som "förkunskap". Kostnad och upplösning | [§4–5](presentation-info.md#4-skillnader) |
| 27–35 | Demo | Se demomanuset nedan | [§7](presentation-info.md#7-demon-när-vinner-vilken) |
| 35–39 | När använda vilken, användningsområden | Beslutstabellen och 4–5 utvalda projekt | [§6](presentation-info.md#6-när-ska-man-använda-vilken), [projekt](vit-in-practice.md) |
| 39–42 | Är ViT state of the art? | Ja, med tre förbehåll | [SOTA](vit-in-practice.md#är-vit-state-of-the-art) |
| 42–45 | Sammanfattning och frågor | De tre budskapen ovan | [Vanliga frågor](#vanliga-frågor-och-svar) |

Behöver du korta: stryk historiken i CNN-delen och visa bara två projekt. Behöver du förlänga:
kör fler bilder live, eller visa jämförelsen med sex modeller.

## Förslag på bilder

1. Titel
2. Inledningsfrågan: ChatGPT, Segment Anything och en självkörande bil har transformern gemensamt
3. Agenda
4. Hur en dator ser en bild: 224 × 224 × 3 = 150 528 tal
5. CNN: ett 3×3-filter som glider över bilden
6. CNN: hierarkin från kanter till texturer till objekt
7. Transformers: meningen med "djuret … det" och pilar som visar attention
8. Self-attention: query, key och value i en enkel figur
9. "An image is worth 16×16 words": en bild uppdelad i rutor
10. Flödet i ViT (diagrammet i [§3](presentation-info.md#3-vision-transformer-vit))
11. Data avgör: 1,3 miljoner bilder mot 300 miljoner
12. Familjen: DeiT, Swin, CLIP, MAE, SAM, DINOv3
13. Skillnader (tabell)
14. För- och nackdelar
15. Demo del 1: från grunden på CIFAR-10 vinner CNN
16. Demo del 2: live i webbläsaren
17. Grafen med brus: [images/noise-accuracy.png](images/noise-accuracy.png). En mörk variant finns som `noise-accuracy-dark.png`
18. Nyans: ocklusion, ConvNeXt och träningsrecept, med grafen
    [images/noise-accuracy-models.png](images/noise-accuracy-models.png)
19. När använda vilken (beslutstabell)
20. Användningsområden, gärna som ett bildkollage
21. Är ViT state of the art?
22. Sammanfattning
23. Frågor

Reservbilder om livedemon strular: [images/noise-examples.jpg](images/noise-examples.jpg) och
graferna i [images/](images/).

## Demomanus

### Före presentationen

```bash
make install       # installerar beroenden
make download      # hämtar testbilder (~340 MB) och modellvikter (~200 MB)
make robustness    # gör graferna som appen visar (~5 minuter)
make demo          # öppnar appen på http://127.0.0.1:7860
```

- **Testa utan internet.** Stäng av wifi och kör `HF_HUB_OFFLINE=1 make demo`. Appen ska starta
  och svara direkt.
- **Öppna appen innan du börjar prata.** Första bilden tar några sekunder medan modellerna laddas.
- **Ha reservbilderna** i presentationen ifall något strular.

### Under demon (cirka 8 minuter)

1. **Från grunden vinner CNN (1 min).** Visa tabellen från CIFAR-10. Säg: "Två små modeller,
   samma träning, 50 000 bilder. CNN:en vinner med 13 procentenheter, för att den redan vet hur
   bilder fungerar."
2. **Presentera modellerna (1 min).** Öppna appen. Säg: "Nu två förtränade modeller. Samma 1,3
   miljoner bilder, nästan samma storlek, och båda känner igen 81 % av rena bilder. Det enda som
   skiljer är arkitekturen."
3. **Dra upp bruset (2 min).** Använd första exempelbilden, mannen med fisken. Vid brusnivå 0,3
   klarar båda bilden. Vid 0,45 säger ViT "tench" (sutare) och ResNet-50 "stingray" (rocka).
4. **Visa att det inte är handplockat (1 min).** Ställ bruset på 0,35 och klicka "Slumpa en bild"
   några gånger. Ibland har CNN rätt och ViT fel, och det är bra att säga högt. Öppna sedan
   "Resultat på 500 osedda bilder" och visa grafen: 62 % mot 43 % vid brusnivå 0,3.
5. **Nyans 1: ocklusion (1,5 min).** Välj "Ocklusion i linje med ViT:s rutnät" på 50 %. ViT klarar
   sig bra. Byt till "Ocklusion förskjuten 8 px". Nu faller ViT, eftersom varje patch får svarta
   kanter. Säg: "Robusthet beror på vad man mäter."
6. **Nyans 2: modern CNN och träningsrecept (1,5 min).** Visa grafen med sex modeller,
   [images/noise-accuracy-models.png](images/noise-accuracy-models.png). ConvNeXt, en CNN från
   2022 som lånat designidéer från ViT, tål brus lika bra som ViT, men inte oskärpa. Samma
   ResNet-50 tränad med 2015 års recept klarar bara 7 % vid brusnivå 0,3, mot 43 % med ett
   modernt recept.
7. **Valfritt: webbkameran.** Ta en bild på något i rummet, till exempel en kaffekopp, banan eller
   ett tangentbord, och dra upp bruset.

Vill du visa nyans 2 live: starta appen med `uv run robustness-demo --models vit convnext`.

### Valfritt: ViT på satellitbilder (cirka 2 minuter)

Passar bäst under användningsområden, vid jordobservation. Visa kartan som `make satellite-demo`
gör, eller [images/satellite-stockholm-map.jpg](images/satellite-stockholm-map.jpg). Säg: "Samma
sorts ViT, förtränad på vanliga foton, har finjusterats på drygt 21 000 Sentinel-2-bilder från
Copernicus på åtta minuter. Den känner igen 99 % av de 5 400 bilder den inte sett. Här klassar den en färsk bild av
Stockholm ruta för ruta: innerstaden, Värtahamnen, vattnet och Djurgården." Nämn också
svagheten ärligt: rutor som blandar hus, vägar och grönska blir ofta "väg".

Förbered med `make satellite-train` och `make satellite-demo` dagen innan. Hämtningen kräver ett
gratis konto hos Copernicus Data Space Ecosystem, se [satellite-demo.md](satellite-demo.md).

## Vanliga frågor och svar

**Är ViT alltid bättre än CNN?**
Nej. Från grunden på lite data vinner CNN, se demo del 1. Vid samma data och beräkningsbudget är
skillnaden liten, och en modern CNN som ConvNeXt tål brus lika bra i demon. På mobiler är CNN
ofta snabbare.

**Varför tål ViT brus bättre än ResNet?**
Self-attention fungerar som ett lågpassfilter som jämnar ut snabba variationer mellan pixlar.
Faltningar fungerar mer som högpassfilter ([Park & Kim 2022](https://arxiv.org/abs/2202.06709)).
Brus är just snabba variationer. Men CNN:er kan göras lika robusta med rätt designval: dela in
bilden i rutor, använda större filter och färre aktiveringslager
([Wang m.fl. 2023](https://arxiv.org/abs/2206.03452)). ConvNeXt gör allt detta.

**Är jämförelsen i demon rättvis?**
Modellerna är tränade på samma data, är ungefär lika stora (22 respektive 26 miljoner
parametrar) och har samma träffsäkerhet på rena bilder. Testbilderna har ingen av dem sett.
Träningsrecepten skiljer sig i detaljer, men ResNet-50 har tränats med det bästa kända receptet
för just den modellen ([ResNet strikes back](https://arxiv.org/abs/2110.00476)). Resultatet håller
även på 10 000 bilder i alla 1000 klasser (ImageNetV2), se [robustness-demo.md](robustness-demo.md).

**Varför just 16×16 pixlar?**
Det är en avvägning. Mindre rutor ger fler tokens och mer detaljer, men kostnaden växer
kvadratiskt. ViT-B/32 ger 49 tokens och är snabb men sämre. ViT-H/14 ger 256 tokens och är bättre
men dyrare.

**Hur vet ViT var i bilden en ruta satt?**
Varje position har en inlärd positionsvektor. Modellen lär sig själv att närliggande rutor hör
ihop.

**Vad är skillnaden mellan ViT och Swin?**
Swin räknar attention inom små fönster som flyttas mellan lagren och bygger en hierarki som en
CNN. Kostnaden växer linjärt med bildstorleken, vilket passar detektion och segmentering.

**Kan man kombinera CNN och transformer?**
Ja, hybrider är vanliga: CoAtNet, MobileViT, FastViT och BEVFormer. Även YOLO26, som i grunden är
en CNN, har ett attention-block.

**Behöver man en GPU?**
För att träna stora modeller, ja. Demon körs på en vanlig laptop. Den använder Apples grafikchip
om det finns, annars processorn, vilket går långsammare men fungerar.

**Kan man se vad en ViT tittar på?**
Attention-kartor ger en fingervisning, men de kan innehålla artefakter
([Registers](https://arxiv.org/abs/2309.16588)). Samma försiktighet gäller Grad-CAM för CNN.

**Vad är DINO, CLIP och SAM?**
Förtränade ViT-modeller. DINO lär sig från bilder utan etiketter, CLIP kopplar ihop bild och
text, och SAM segmenterar objekt.

**Vilken ska jag välja för mitt projekt?**
Börja med en förtränad modell som passar uppgiften, se beslutstabellen i
[presentation-info.md](presentation-info.md#6-när-ska-man-använda-vilken). Testa den på din egen
data, inklusive de störningar som finns i verkligheten.
