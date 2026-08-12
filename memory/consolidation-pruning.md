---
name: consolidation-pruning
description: Periyodik offline "uyku" fazı — 1. fazda sadece ölü bağların budanması, atom birleştirme 2. faza ertelendi
metadata:
  type: project
---

Graf periyodik olarak, sorgu dışı bir zamanda kendini toparlar. Biyolojik
karşılığı uyku sırasındaki konsolidasyon.

**Karar: 1. fazda sadece budama.** Hiç enerji taşımamış bağlar silinir. Index
küçülür, seyrek matris daha seyrek olur, yayılma hızlanır.

**Birleştirme 2. faza ertelendi.** Hep birlikte ateşlenen atomları tek bir
süper-atomda toplamak güçlü bir sıkıştırma ama **geri döndürülemez** — birleşen
atomlar ayrılamaz. Ölçüm yapılmadan alınacak bir karar değil.

Bu mekanizma [[learned-layer-hebbian]] ile aynı madalyonun iki yüzü: biri
kullanılan bağı güçlendirir, diğeri kullanılmayanı temizler. İkisi birlikte
olmazsa öğrenilmiş katman sınırsız büyür.

Uyarı: budama, "henüz sorulmamış soruların" yollarını da kesebilir. Bir bağın hiç
enerji taşımamış olması, gereksiz olduğu anlamına gelmez — sadece o tür sorunun
henüz gelmediği anlamına gelebilir. Bu yüzden budama eşiği agresif olmamalı ve
silinen bağlar (en azından 1. fazda) geri yüklenebilir şekilde saklanmalı.
