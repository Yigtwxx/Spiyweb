---
name: redundancy-as-vote
description: Tekrar eden vektörün bağı sıfırlanır ve fikrin oy sayısı artar — projenin bilinen literatürde karşılığı olmayan tek parçası
metadata:
  type: project
---

Yayılma sırasında bir atom, zaten aktif olan bir atomla neredeyse aynı şeyi
söylüyorsa: **o bağın ağırlığı 0'lanır** (enerji oraya akmaz, context'e girmez)
ve karşılığında **kaynak fikrin oy sayısı 1 artar**.

İki kazanç aynı anda:
- Aynı fikrin parafrazları context'i şişirmez, token yemez, LLM'in dikkatini
  dağıtmaz.
- Tekrar çöpe gitmez; "bu fikir corpus'ta birden çok yerde destekleniyor"
  sinyaline dönüşür.
- Üçüncüsü hediye: bağ kesildiği için o daldan yayılma da durur, hesap ucuzlar.

**Neden MMR değil:** Maximal Marginal Relevance tekrarı sadece atar ve unutur —
sinyal kaybolur. Burada tekrar *kanıta* çevriliyor. Fark küçük görünüyor ama
projenin özgün olduğu tek nokta burası; PPR tabanlı graph-RAG zaten var
([[prior-art-and-differentiation]]).

**Tespit dinamiktir, sorgu anındadır** — index anında statik kümeleme değil.
Gerekçe: "benzer" kavramı sorguya görecelidir. İki chunk bir soruda birbirinin
yerine geçebilirken başka bir soruda ayrı şeyler söylüyor olabilir. Maliyeti
düşük, çünkü aynı anda aktif düğüm sayısı birkaç yüzü geçmez.

**Oy sayımı doküman/kaynak bazındadır, chunk bazında değil.** Aksi hâlde
corpus'ta aynı metnin 50 kopyası varsa kazanan en iyi desteklenen fikir değil, en
çok kopyalanan fikir olur.

Bu mekanizmanın bilinen kör noktası: cosine 0.9 olan iki chunk **zıt** şey
söylüyor olabilir ve bu tasarım onu "mutabakat" sayar. Ayrıntı [[known-risks]].
