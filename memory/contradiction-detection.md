---
name: contradiction-detection
description: Çelişki tespiti index anında küçük çok-dilli NLI modeliyle edges/ içinde yapılır; core/ yalnız hazır işaretli negatif kenarları işler
metadata:
  type: project
---

[[contradiction-handling]] çelişkiyle *ne yapılacağını* söylüyordu (negatif yük +
seçenekli soru); bu dosya *nasıl bulunacağını* kapatır ([[known-risks]] #1'in
seçilmemiş kalan yolu).

**Karar (2026-08-12): index anında NLI.** Küçük, çok dilli (TR+EN) bir NLI
modeli index sırasında aday çiftleri — esas olarak yüksek benzerlikli **önerme**
çiftlerini ([[node-layers-and-mass]]) — tarar; "contradiction" çıkanlara
**negatif kenar** üretir. Üretim `edges/` içinde yaşar (`edges/nli.py`);
`core/` yalnızca hazır işaretli veriyi işler, saflık korunur
([[architecture-boundaries]]).

**Why:** Çelişki tespiti chunk üstünde bulanık, önerme üstünde nettir; NLI bu
işin standart ve LLM'siz-çalışma-anı çözümüdür. Index anında çalıştığı için
sorgu gecikmesine maliyet eklemez.

**How to apply:** NLI çağrısı asla `core/` içine girmez; `core/` sadece işaretli
negatif kenarları tüketir. Sorgu anında NLI yok.

Elenen alternatifler:
- **Kural/negasyon sezgileri:** ucuz ama recall düşük; TR ve EN için ayrı kural
  seti gerektirir.
- **Faz 1'de tespiti dışarıdan verme:** mekanizma test edilirdi ama "dürüst
  retriever" konumlandırması tespit olmadan eksik kalırdı.

Açık kalanlar: NLI model seçimi ve aday çifti eşiği — [[open-questions]].
İzlenecek risk: NLI recall'ü ölçülmeden "çelişki körlüğü kapandı" denemez.
