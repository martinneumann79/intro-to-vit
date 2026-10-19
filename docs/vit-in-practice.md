# ViT i praktiken: användningsområden och state of the art

Del av underlaget för presentationen, se även [presentation-info.md](presentation-info.md).
Alla länkar kontrollerades i oktober 2026.

## Användningsområden med existerande projekt

Mönstret är nästan alltid detsamma. En stor ViT förtränas en gång på enorma mängder bilder och
används sedan som "öga" i många olika tillämpningar.

### Generella bildmodeller (foundation models)

- **DINOv3** (Meta, augusti 2025). En ViT med 7 miljarder parametrar, tränad utan etiketter på
  1,7 miljarder bilder. Enligt Meta ger en och samma frysta modell toppresultat på detektion,
  segmentering och djupskattning. Meta har också destillerat mindre ConvNeXt-modeller (CNN) ur den
  för enheter med mindre beräkningskraft.
  [GitHub](https://github.com/facebookresearch/dinov3) · [Projektsida](https://ai.meta.com/dinov3/)
- **DINOv2** (Meta, 2023). Föregångaren, som många av projekten nedan bygger på.
  [GitHub](https://github.com/facebookresearch/dinov2)
- **CLIP** (OpenAI, 2021) och den öppna varianten **OpenCLIP**. Kopplar ihop bild och text, så
  att man kan söka bland bilder med vanliga ord eller klassificera utan träningsexempel.
  [CLIP](https://github.com/openai/CLIP) · [OpenCLIP](https://github.com/mlfoundations/open_clip)
- **SigLIP 2** (Google DeepMind, 2025). Efterföljaren till CLIP-idén och "ögat" i många
  multimodala språkmodeller. [Artikel](https://arxiv.org/abs/2502.14786) ·
  [GitHub](https://github.com/google-research/big_vision)

### Segmentering

- **SAM 3** (Meta, november 2025). Hittar och segmenterar varje förekomst av ett begrepp, till
  exempel "gul skolbuss", i bilder och video. [GitHub](https://github.com/facebookresearch/sam3) ·
  [Projektsida](https://ai.meta.com/sam3/)
- **SAM 2** (Meta, 2024). Segmentering som följer objekt genom video.
  [GitHub](https://github.com/facebookresearch/sam2)
- **MedSAM** (2023–2024). Segment Anything anpassad för medicinska bilder, tränad på 1,57
  miljoner bild- och maskpar från 10 bildtyper som CT, MR och ultraljud.
  [GitHub](https://github.com/bowang-lab/MedSAM)

### Objektdetektion i realtid

- **RF-DETR** (Roboflow, ICLR 2026). Detektor med en DINOv2-ViT som ryggrad. Enligt Roboflow
  den första realtidsdetektorn över 60 AP på benchmarken COCO.
  [GitHub](https://github.com/roboflow/rf-detr)
- **Motpol: YOLO26** (Ultralytics, januari 2026). I grunden en CNN, med ett enda litet
  attention-block på lägsta upplösningen. Minsta varianten har 2,6 miljoner parametrar och är
  snabb på CPU och små enheter. Fortfarande förstahandsval i många inbyggda system.
  [GitHub](https://github.com/ultralytics/ultralytics) ·
  [Modellbeskrivning](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26.yaml)

### Djup och 3D

- **Depth Anything 3** (ByteDance, november 2025). Visar att en vanlig DINO-ViT räcker för att
  skatta djup och kamerapositioner från en eller flera bilder.
  [GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Depth Pro** (Apple, 2024). Skarp djupkarta i verklig skala från en enda bild på 0,3 sekunder
  på ett vanligt grafikkort, byggd på en ViT i flera skalor. [GitHub](https://github.com/apple/ml-depth-pro)

### Språkmodeller som ser, dokument och bildsökning

- **Qwen3-VL** (Alibaba, 2025). En SigLIP-2-ViT gör bilden till tokens som språkmodellen läser
  tillsammans med texten. [GitHub](https://github.com/QwenLM/Qwen3-VL)
- **DeepSeek-OCR** (oktober 2025). En ViT-baserad bildkodare komprimerar en dokumentsida till
  ett hundratal bild-tokens, som en språkmodell sedan läser av.
  [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- **Immich** (öppen källkod). Självhostat fotobibliotek där sökningar som "hund på stranden"
  körs med en CLIP-ViT. [GitHub](https://github.com/immich-app/immich) ·
  [Dokumentation](https://docs.immich.app/features/searching)

### Medicin

- **UNI** (Mahmood Lab, Harvard, 2024). ViT tränad på över 100 miljoner vävnadsutsnitt från över
  100 000 patologibilder i 20 vävnadstyper. [GitHub](https://github.com/mahmoodlab/UNI)
- **Prov-GigaPath** (Microsoft och Providence, Nature 2024). ViT för hela patologiska snitt.
  [GitHub](https://github.com/prov-gigapath/prov-gigapath)
- **Virchow2** (Paige). ViT för patologi. [Hugging Face](https://huggingface.co/paige-ai/Virchow2)
- **RETFound** (Moorfields och UCL, Nature 2023). ViT förtränad på 1,6 miljoner bilder av
  näthinnan, för att upptäcka ögonsjukdomar. [GitHub](https://github.com/rmaphoh/RETFound_MAE)
- **Swin UNETR** (MONAI och NVIDIA). Swin-baserad 3D-segmentering, till exempel av hjärntumörer.
  [GitHub](https://github.com/Project-MONAI/research-contributions)

### Jordobservation, klimat och väder

- **Prithvi** (NASA och IBM, 2023). Transformer förtränad på över 1 TB satellitbilder från
  Landsat och Sentinel-2. Används för översvämningar, brandområden och grödor.
  [Hugging Face](https://huggingface.co/ibm-nasa-geospatial)
- **World Resources Institute med DINOv3.** Mäter trädhöjd från satellitbilder för att följa
  avskogning. Enligt Meta sjönk medelfelet i en region i Kenya från 4,1 till 1,2 meter jämfört
  med DINOv2. [Källa](https://ai.meta.com/dinov3/) · [Trädhöjdsmodellen på GitHub](https://github.com/facebookresearch/HighResCanopyHeight)
- **Aurora** (Microsoft, Nature 2025). Modell för hela jordsystemet byggd på en 3D Swin
  Transformer. Prognoser för väder, luftkvalitet, havsvågor och tropiska cykloner.
  [GitHub](https://github.com/microsoft/aurora)

### Natur och biologi

- **BioCLIP 2** (Imageomics, 2025). ViT tränad på 214 miljoner bilder av organismer för att
  artbestämma över hela livets träd. [GitHub](https://github.com/Imageomics/bioclip-2)
- **Motexempel: SpeciesNet** (Google, 2025). Klassificerar djur i viltkamerabilder med
  EfficientNetV2-M, alltså en CNN. [GitHub](https://github.com/google/cameratrapai)

### Robotik

- **OpenVLA** (Stanford m.fl., 2024). Robotstyrning där bilden kodas av DINOv2 och SigLIP och en
  språkmodell väljer nästa rörelse. Tränad på 970 000 robotdemonstrationer.
  [GitHub](https://github.com/openvla/openvla)
- **V-JEPA 2** (Meta, 2025). ViT tränad på över 1 miljon timmar video, som också används för att
  planera robotrörelser. [GitHub](https://github.com/facebookresearch/vjepa2)

### Bild- och videogenerering

- **DiT** (2022). Diffusionsmodell där det vanliga U-Net:et, en CNN, byts mot en transformer
  som arbetar på patchar. [GitHub](https://github.com/facebookresearch/DiT) Samma idé finns i
  Stable Diffusion 3, FLUX ([GitHub](https://github.com/black-forest-labs/flux)) och OpenAI:s Sora.

### Industri och kvalitetskontroll

- **Dinomaly** (2024–2025). Hittar defekter som repor och sprickor med en förtränad ViT. Den
  tränas bara på felfria exempel. [GitHub](https://github.com/guojiajeremy/Dinomaly)

### Självkörande fordon: ofta hybrider

- **BEVFormer** (2022). En transformer slår ihop bilderna från flera kameror till en vy ovanifrån.
  Bildryggraden är ofta en CNN, som ResNet. Bra exempel på att CNN och transformer kombineras i
  praktiken. [GitHub](https://github.com/fundamentalvision/BEVFormer)

## Är ViT state of the art?

**Kort svar:** Ja, i den meningen att de flesta toppresultat och nästan alla stora förtränade
bildmodeller 2025–2026 bygger på ViT eller andra transformer-varianter. Men ViT är inte bäst i
alla lägen, och arkitekturen förklarar bara en del av försprånget.

### Det som talar för ja

- **Bildklassificering.** ImageNet är i praktiken mättat runt 90–91 %. Toppnoteringarna är
  ViT-baserade eller hybrider: [CoCa](https://arxiv.org/abs/2205.01917) med en ViT-kodare når
  91,0 % och hybriden [CoAtNet](https://arxiv.org/abs/2106.04803) 90,88 %.
- **Stora förtränade modeller.** DINOv3, SAM 3, SigLIP 2, CLIP och Depth Anything 3 är alla
  ViT-baserade. De är vad man laddar ner när man börjar ett nytt bildprojekt i dag.
- **Språkmodeller som ser.** Bildkodaren i modeller som Qwen3-VL är en ViT.
- **Bild- och videogenerering** har gått från U-Net (CNN) till diffusionstransformers.
- **Realtidsdetektion.** RF-DETR med ViT-ryggrad leder i noggrannhet på COCO.

### Det som talar emot, eller nyanserar

- **Vid samma data och beräkning presterar CNN lika bra.** DeepMind visade 2023 att CNN:er
  (NFNet) förtränade på 4 miljarder bilder matchar ViT vid samma beräkningsbudget och når 90,4 %
  på ImageNet ([artikel](https://arxiv.org/abs/2310.16764)). ConvNeXt V2 når 88,9 %
  ([artikel](https://arxiv.org/abs/2301.00808)). ViT:s dominans handlar alltså mycket om att den
  är lätt att skala och passar ihop med språkmodeller.
- **På små enheter är CNN ofta bättre.** YOLO26, MobileNetV4
  ([artikel](https://arxiv.org/abs/2404.10518)) och EfficientNet är snabbare och har bättre stöd
  på mobiler och kameror. Även Meta släppte ConvNeXt-versioner av DINOv3 för just det.
- **Hybrider är vanliga.** Swin, CoAtNet, FastViT och BEVFormer blandar idéerna. Även
  YOLO26, som i grunden är en CNN, har ett attention-block.
- **Det finns ingen central topplista längre.** Papers with Code, som var standardkällan för
  benchmarkresultat, stängdes av Meta i juli 2025. Jämförelser måste i dag göras utifrån
  originalartiklarna.

### Slutsats att säga på scenen

> ViT är grunden i nästan alla ledande bildmodeller i dag, framför allt för att den skalar och
> talar samma språk som textmodellerna. Men vid samma data och beräkning är skillnaden mot en
> modern CNN liten, och på små enheter är CNN fortfarande ofta rätt val.
