---
name: learned-layer-hebbian
description: Ağ kullanıldıkça öğrenir ama öğrenme ana grafı kirletmez — ayrı, kapatılabilir bir "öğrenilmiş katman"da tutulur
metadata:
  type: project
---

Örümcek kullandığı ipliği kalınlaştırır, kullanmadığı kopar. Spiyweb'de de enerji
taşımış bağlar zamanla güçlenir, hiç taşımamış olanlar incelir. Sinir bilimindeki
karşılığı Hebbian öğrenme: *birlikte ateşlenen birlikte bağlanır*.

**Karar: öğrenme ayrı bir katmanda tutulur.** `edges/` altındaki `semantic`,
`entity`, `structural` katmanlarının yanına dördüncü bir `learned` katmanı gelir
([[hybrid-edge-layers]]). Ana graf hiç değişmez.

Gerekçe — üç madde:
- **Geri alınabilir.** Öğrenme kötü gidiyorsa katman silinir, index yeniden
  kurulmaz.
- **Ölçülebilir.** Ablation'da `learned` ağırlığı 0 yapılır, katkısı tek başına
  görülür.
- **Yanlılık izole.** Popülerlik yanlılığı (çok sorulan konunun bağları sürekli
  güçlenir) ana grafa sızmaz; en kötü ihtimalle bir katman bozulur.

Elenen alternatif: her başarılı sorguda ana grafı doğrudan güncellemek. Reddedildi
çünkü bir kez oluşan yanlılık geri döndürülemez ve hangi sonucun graftan hangisinin
öğrenmeden geldiği ayırt edilemez hale gelir.

Unutma katsayısı zorunlu: pekiştirme sınırsız birikirse graf zamanla tek bir
"popüler bölge"ye çöker. [[consolidation-pruning]] ile birlikte çalışır.
