---
name: negative-knowledge-atoms
description: Olumsuz önermeler kalıcı negatif kutuplu atom olur — tersini iddia eden sorgunun enerjisini emer ve "corpus itiraz ediyor" uyarısı üretir (D34)
metadata:
  type: project
---

**Karar (2026-08-12, D34): negatif bilgi birinci sınıf vatandaş.** "X, Y'ye
sebep olmaz", "Z artık desteklenmiyor" gibi **olumsuz önermeler** index anında
tespit edilir ve kutuya **kalıcı negatif kutuplu atom** olarak girer. Tersini
iddia eden bir sorgu bu atomlara değdiğinde enerji emilir ve çıktıya
**"corpus bu iddiaya itiraz ediyor"** uyarısı düşer.

**Why:** Embedding'ler negasyonu taşımaz — literatür bunu "Semantic Collapse"
diye adlandırıyor ve çözümler çok erken aşamada. Mevcut negatif yük mekanizması
([[contradiction-handling]], [[query-profiles-and-negative-seeds]]) zaten var;
bu, projenin **en güçlü özgünlük adayı** ve ucuz bir uzantısı.

**Zamanlama:** Tasarıma ve şemaya **şimdi** girer (node şemasına `polarity`
alanı + config bayrağı); uygulama **Faz 1'in ilk ölçümünden SONRA** ablation
olarak yapılır. 2 haftalık timebox korunur.

**How to apply:** Polarite tespiti index anında, `core/` dışında
(önerme çıkarımı + NLI hattının parçası, [[contradiction-detection]]).
`core/` yalnız `polarity` etiketli atomu işler. Config'ten tek bayrakla
kapatılabilir — ablation şartı.

Açık kalan: negatif önerme tespit yöntemi — [[open-questions]].
