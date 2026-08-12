---
name: explained-abstention
description: Güven düşükken LLM'siz yapısal red gerekçesi — "A ve B kümeleri arasında köprü yok, şu tür kaynak eksik" (D35, Faz 1)
metadata:
  type: project
---

**Karar (2026-08-12, D35): açıklamalı red, Faz 1 kapsamında.** Güven skoru
düşük kaldığında `retrieve()` sadece sayı döndürmez; **LLM'siz, şablonla
üretilen yapısal bir gerekçe** döndürür: hangi entity kümeleri aktive oldu,
aralarında köprü var mı, enerji nerede öldü, ne tür bir kaynak eksik.

**Why:** Abstention literatürü var ama *yapısal ve LLM'siz* eksik-köprü
raporuna rastlanmadı — niş içinde yüksek özgünlük. Mevcut makinelerin
([[corpus-gap-detection]] + [[confidence-and-abstention]] +
[[output-contract]]) üstüne ince bir katman: **neredeyse bedava** ve "dürüst
retriever" konumlandırmasının vitrini.

**How to apply:** `output.py` içinde; [[corpus-gap-detection]]'ın corpus
seviyesindeki uyarısının **sorgu-başına** versiyonu. Şablonlar `core/` dışında
([[contradiction-handling]] soru şablonlarıyla aynı yerde). Config'ten
kapatılabilir.

İlgili: [[negative-knowledge-atoms]] — negatif atom uyarısı da aynı rapor
yapısında taşınır.
