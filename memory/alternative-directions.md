---
name: alternative-directions
description: Dört alternatif yön değerlendirildi — A/B/C kabul edildi (D12/D10/D15), D (öğrenilmiş sönümleme) 2. faza ertelendi
metadata:
  type: project
---

Mevcut tasarımın üstüne düşünülmüş yönler. **Güncelleme (2026-08-12): A, B ve C
karara bağlandı ve tasarıma girdi** — A → D12 [[multi-seed-colors]], B → D10
[[node-layers-and-mass]], C → D15 [[contradiction-handling]]. Yalnız D açık ve
2. faza ertelendi. Aşağıdaki metinler, kararların ilk düşünüldüğü hâliyle
korunuyor.

**A. Çok tohumlu, renkli enerji (KARAR: kabul, D12).**
Tek topçuk yerine sorguyu alt-sorulara veya varlıklarına ayırıp **birden çok
topçuk** at, her birine farklı bir "renk" ver. Renkler ayrı ayrı yayılsın. Bir
atomda **iki farklı renk buluşuyorsa**, o atom sorunun iki ayrı parçasını
birbirine bağlayan **köprü**dür — ve multi-hop bir sorunun cevabı tam olarak
oradadır. Şu anki toplamalı birikim bunu renk ayrımı olmadan yapıyor; renk
eklemek "neden yükseldi" bilgisini de veriyor, yani hem daha güçlü hem daha
açıklanabilir. Maliyeti: sorgu ayrıştırma adımı (LLM veya kural tabanlı).

**B. Atom = chunk yerine önerme (KARAR: kabul, D10 — iki katman birden).**
Düğümler 300-500 token'lık chunk yerine **atomik olgu cümleleri** olsun.
Kazanç: "aynı şeyi söylüyor" kavramı bir chunk için bulanık, bir önerme için
nettir — [[redundancy-as-vote]] mekanizması ancak o zaman tam anlamıyla çalışır.
Ayrıca çelişki tespiti de önermeler üstünde çok daha kolaydır
([[known-risks]] #1). Maliyeti: index anında önerme çıkarımı (LLM), yani pahalı
bir ön işleme.

**C. Çelişki = negatif yük (KARAR: kabul, D15; tespit için [[contradiction-detection]]).**
Zıt şey söyleyen iki aktif atom birbirinin enerjisini artırmak yerine
**söndürsün** ya da ikisi birden "çelişkili" işaretiyle context'e girsin ve LLM'e
"burada anlaşmazlık var" bilgisi verilsin. Metafora da oturuyor: pozitif ve
negatif yük. Bu, [[known-risks]] içindeki en ciddi kör noktayı doğrudan
kapatıyor ve tek başına yayınlanabilir bir katkı olabilir.

**D. Öğrenilmiş sönümleme (KARAR: 2. faza ertelendi).**
`damping` sabit `0.60` yerine kenar türüne göre değişsin: varlık kenarı uzağa
taşısın, yapısal kenar çabuk sönsün. İleri adımı: eval setinden öğrenilen katman
ağırlıkları. 2. fazın konusu; 1. fazda elle ayar yeterli.

Not: A ve C birlikte, projeyi "PPR'nin bir varyantı" olmaktan çıkarıp ayrı bir
şeye dönüştürebilir. B ise mevcut mekanizmayı olduğu gibi bırakıp altındaki
zemini sağlamlaştırıyor.
