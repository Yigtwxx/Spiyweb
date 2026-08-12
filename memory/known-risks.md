---
name: known-risks
description: Tasarımın bilinen üç kör noktası — çelişki körlüğü, hub cezası, tekrarın doğrulukla karıştırılması
metadata:
  type: project
---

Üçü de biliniyor ve kabul edilerek devam edildi. Sürpriz değiller; sürpriz
olurlarsa bu dosya okunmamış demektir.

**1. Çelişki körlüğü — en ciddisi.** Cosine benzerliği `0.9` olan iki chunk
tamamen **zıt** şey söylüyor olabilir ("X güvenlidir" / "X güvenli değildir").
Embedding'ler **olumsuzlamayı** taşımaz. [[redundancy-as-vote]] bu ikisini
"mutabakat" sayar ve yanlış bir konsensüsü *güçlendirir*. Yani mekanizma, en çok
işe yaradığı yerde en tehlikeli. **Çözüm yolu seçildi (2026-08-12): index anında
NLI** — [[contradiction-detection]]. Risk "açık" değil "izleniyor" statüsünde:
NLI recall'ü ölçülene kadar kapanmış sayılmaz.

**2. Hub cezası.** Enerjinin komşulara oranlı **bölüştürülmesi**
([[propagation-rules]]), çok komşulu yani bilgi yoğun düğümleri haksız yere
cezalandırır: bağlantısı fazla olan atom her komşusuna az enerji verir. Bilinçli
bir takas — alternatifi olan kopyalama, üç hop'ta kutuyu taşırıyordu. Bilinen
yumuşatma yolu: benzerliği `sim ** alpha` ile eğmek.

**3. Güçlü fikir ≠ doğru fikir.** Oy sayısı corpus'taki desteği ölçer, doğruluğu
değil. Aynı pazarlama metninin 50 kopyası varsa "en çok desteklenen fikir" o
olur. Doküman/kaynak bazlı oy sayımı bunu **sınırlar ama yok etmez**.

**4. Fayda büyüklüğü belirsizliği.** Projenin tek gerçek bilinmezi
uygulanabilirlik değil, farkın büyüklüğü. `top-k` sürpriz biçimde güçlü bir
baseline ve iteratif retrieval de rakip ([[prior-art-and-differentiation]]).
1. fazın amacı bu sayıyı erken öğrenmek; negatif sonuç da değerli bir sonuçtur
([[roadmap-and-gates]]).
