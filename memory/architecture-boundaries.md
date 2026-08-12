---
name: architecture-boundaries
description: core/ hiçbir I/O bilmez — 1. fazdan 2. faza terfiyi yeniden yazım olmaktan çıkaran tek kural
metadata:
  type: project
---

Dört sınır kuralı var; ilki diğer üçünden daha önemli.

**1. `core/` dış dünyayı tanımaz.** Vektör store yok, LLM yok, embedding modeli
yok, dosya sistemi yok, ağ yok. İçeri dizi ve config girer, dışarı sayı çıkar.
Bu saflık, [[roadmap-and-gates]] içindeki 1 → 2 terfisini *bedava* yapan şeydir;
kirlendiği anda terfi yeniden yazıma döner. Oraya bir tane `requests` import'u
girerse proje sessizce pahalılaşır.

**2. `edges/` çoğuldur, bilerek.** Hibrit karar config'te yaşar, kodda değil. Yeni
bir kenar türü eklemek `core/`'a dokunmayı gerektiriyorsa sınır yanlış çizilmiş
demektir.

**3. `evaluation/` 1. fazın ürünüdür.** Yardımcı script değil. 2. fazda
regression testine dönüşür; hiçbir aşamada çöpe atılmaz. (Adı `eval` değil
`evaluation` — Python builtin'ini gölgelememek için, 2026-08-12.)

**4. `ui/` paketin parçası değildir.** `pip install spiyweb` diyen birinin
Streamlit'e ihtiyacı olmamalı ([[dev-ui-scope]]). UI, opsiyonel extra olarak
paketlenir: `pip install spiyweb[ui]` (2026-08-12).

Yerleşim kararı (2026-08-12): **`src/spiyweb/` layout** — test izolasyonu ve
paketleme hatalarını public'e çıkmadan yakalar.

Ek kural: kodda sihirli sayı yok. Her ayarlanabilir değer `config.py` içinde bir
dataclass alanı olarak, belgelenmiş varsayılanıyla durur. Bu hem UI'ın slider
üretebilmesi hem de deneylerin tekrarlanabilir olması için gerekli.

3. fazın (framework) geriye sızmasının ilk belirtisi şudur: 1. fazda "ileride
lazım olur" diye yazılmış soyutlamalar. Görülürse silinir.
