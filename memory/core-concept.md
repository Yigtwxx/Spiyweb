---
name: core-concept
description: Spiyweb'in çekirdek fikri — sorgunun vektör kutusuna enerji tohumu olarak atılıp sönümlenerek yayılması
metadata:
  type: project
---

Kapalı bir kutu düşün, içi oksijen atomlarıyla dolu ve hepsi cansız. Her atom
veritabanındaki bir vektör (bir chunk). Kullanıcı soru sorduğunda dışarıdan
kutuya **canlı bir topçuk** giriyor. Dokunduğu atomlar canlanıyor; her canlanan
atom kendi canının bir kısmını bağlı olduğu atomlara aktarıyor; enerji her
adımda azalıyor ve sıfıra indiğinde ağ büyümeyi durduruyor.

Güçlü bağlarla bağlı atomlar hemen canlanır. Zayıf bağlı olanlar birkaç sıçrama
ötede canlanır — ve yeterince ince iplik aynı atomda buluşursa o atom yine de
canlanır. Cevap tek bir kümeden değil, **ağın tamamından** üretilir.

Bunun çözdüğü problem: `top-k` sert bir kesme yapar ve **dolaylı ilgililik**
kavramı yoktur. Sorguya zayıf benzeyen ama gerçekten ilgili bir chunk'a bağlı
olan bir chunk, `top-k` için görünmezdir. `k`'yı büyütmek çözmez — aynı sonucun
parafrazlarıyla context'i doldurur.

Teknik karşılığı: sparse graf üzerinde **spreading activation**, matematiksel
olarak damping faktörlü **Personalized PageRank**'e denk. Tekrarlı seyrek
matris-vektör çarpımına indirgeniyor; ucuz ve sayısal olarak kararlı.

Kurallar [[propagation-rules]], kenar tanımı [[hybrid-edge-layers]], en özgün
parça [[redundancy-as-vote]].
