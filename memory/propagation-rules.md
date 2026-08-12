---
name: propagation-rules
description: Enerji yayılmasının dört kuralı — çarpımsal sönümleme, benzerlikle oranlı bölüşüm, toplamalı birikim, eşikle durma
metadata:
  type: project
---

Dört karar, dördü de gerekçeleriyle kesinleşti. Bunlar ayar değil **kimlik**;
değiştirilecekse bilinçli değiştirilmeli.

**1. Sönümleme çarpımsaldır** (`E_out = E_in × damping`), çıkarma değil.
İlk taslakta enerji sabit miktarda azalıyordu (10 → 8 → 6 → 4 → 0). Bu, zayıf bağ
ile güçlü bağa aynı bedeli ödetiyordu. Çarpmada zayıf bağ hızla söner, güçlü
zincir uzağa gider — "zayıf bağ uzağa gitmesin" niyeti ancak böyle matematiğe
geçiyor. Varsayılan `damping = 0.60`: düğüm enerjisinin %60'ını dağıtır, %40'ını
tutar.

**2. Enerji komşulara bölüştürülür, kopyalanmaz** — ve bölüşüm bağ gücüyle
orantılıdır. Tek komşu varsa ~%60 alır; eşit uzaklıkta beş komşu varsa her biri
~%12 alır. Kopyalama seçilseydi üç hop'ta binlerce düğüm aktive olur, kutu
taşardı. Kabul edilen bedeli [[known-risks]] içindeki hub cezası.

**3. Birikim toplamalıdır.** Bir atoma iki ayrı yoldan enerji gelirse toplanır.
*Converging evidence* bunun bedava yan ürünü: bağımsız iki zayıf yol aynı düğümde
buluşunca o düğüm yükselir. Kanonik örnekte `D` düğümünün, sorguya en benzer atom
olmadığı hâlde 3. sıraya çıkmasının sebebi tam olarak budur — ve bu terfi,
projenin `top-k`'ya karşı sattığı tek şeydir.

**4. Durma eşikle olur**, sayıyla değil. Enerji eşiğin (seed `10.0` iken `1.5`)
altına inince o dal ölür. Kasıtlı olarak "kaç sonuç getir" parametresi yok; ağ
kendi kendine sönüyor. `max_hop` sadece emniyet freni, birincil durdurucu değil.

Bağlam: [[core-concept]]. Kanonik sayısal örnek CLAUDE.md ve design spec içinde.
