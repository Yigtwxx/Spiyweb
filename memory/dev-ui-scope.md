---
name: dev-ui-scope
description: UI 1. fazda var ama ürün arayüzü değil — damping ve eşiği gözle ayarlamak için salt-okunur teftiş aracı
metadata:
  type: project
---

UI 1. faza dahil edildi. Gerekçe teknik: bu algoritmanın parametreleri (damping,
eşik, katman ağırlıkları) **kör deneyerek** ayarlanamaz. Ağın nasıl yayıldığını
görmeden hangi sayının neden işe yaradığını anlamak mümkün değil.

Ama kapsamı dar: **geliştirici aracı**, ürün yüzeyi değil. Salt-okunur, tek
sayfa, Streamlit.

Gösterecekleri:
- seed atomlar ve enerjileri
- kenar ağırlıkları, katmanına göre renklendirilmiş ([[hybrid-edge-layers]])
- kesilen tekrar bağları kesikli çizgiyle ([[redundancy-as-vote]])
- oy sayıları
- `damping` / `threshold` / `max_hop` kaydırıcıları
- aktive olan ağ ile düz `top-k` sonuç listesi **yan yana**

Son madde en önemlisi: farkı gözle görmeden benchmark sayısını yorumlamak zor.

Sınır: `ui/` paketin parçası değil, bağımlılığı da değil. `pip install spiyweb`
diyen birinin Streamlit'e ihtiyacı olmamalı ([[architecture-boundaries]]).
Ürünleşme 2. fazın konusu, o da talep gelirse ([[roadmap-and-gates]]).
