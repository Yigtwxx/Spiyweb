---
name: corpus-lint
description: KB sağlık teftişi — yetim kümeler, aşırı yüklü hub'lar, çelişki haritası, kopya yoğunluğu; Faz 2 ürün adayı (D37)
metadata:
  type: project
---

**Karar (2026-08-12, D37): "corpus lint" resmi Faz 2 ürün adayı.**
Retrieval'dan bağımsız, offline bir teftiş modu: grafın topolojisinden
**yetim kümeler** (hiçbir şeye köprüsü olmayan bilgi adaları), **aşırı yüklü
hub'lar** (hub cezasının en çok vurduğu düğümler, [[known-risks]] #2),
**çelişki haritası** ([[contradiction-detection]] çıktılarının toplu görünümü)
ve **kopya yoğunluğu** (oy dağılımından) raporlanır.

**Why:** RAG gözlemlenebilirliği cevap seviyesinde mevcut (Ragas vb.) ama
**graf-topoloji seviyesinde corpus teftişi** aracı bulunamadı — gerçek boşluk
adayı. [[corpus-gap-detection]] zaten "2. fazda ayrı ürün olabilir" diyordu;
bu karar onu resmileştirir. Multi-hop kazancı marjinal çıkarsa
([[prior-art-and-differentiation]]) projenin **B planı** budur: ürün argümanı
"daha iyi retrieval"dan "bilgi tabanı kalite analizi"ne kayar.

**Zamanlama:** Faz 2. Faz 1 metrikleri ve timebox değişmez; Faz 1'in tek
katkısı, eval çıktılarının bu istatistikleri üretmeye yetecek veriyi zaten
kaydediyor olmasıdır ([[output-contract]]).

İlgili: [[explained-abstention]] — sorgu-başına gerekçenin corpus-geneli hali.
