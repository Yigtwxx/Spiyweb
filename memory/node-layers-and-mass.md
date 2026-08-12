---
name: node-layers-and-mass
description: Düğümler iki katmanlı — hem chunk hem önerme, aralarında bağ; ayrıca atomlar uzunlukla orantılı kütle taşıyor
metadata:
  type: project
---

İki karar birlikte alındı ve birbirini etkiliyor.

**1. İki katmanlı düğüm yapısı.** Graf hem **chunk** düğümleri hem **önerme
(proposition)** düğümleri içerir, aralarında bağ vardır. Önerme = atomik olgu
cümlesi.

Kazancı: "aynı şeyi söylüyor" kavramı bir chunk için bulanık, bir önerme için
nettir. [[redundancy-as-vote]] mekanizması ancak önermeler üstünde tam anlamıyla
çalışır — ve [[contradiction-handling]] için de doğru zemin budur; bir önerme
temiz biçimde olumsuzlanabilir, 400 token'lık bir chunk olumsuzlanamaz.

Bedeli dürüstçe: index anında **LLM ile önerme çıkarımı** gerekiyor. 1. fazın
maliyeti bu kararla arttı. Hafifletici etken, 1. fazın sabit ve sınırlı bir
benchmark corpus'u üzerinde koşacak olması ([[phase1-settings]]).

**2. Atom kütlesi uzunlukla orantılı.** Uzun chunk geç canlanır ama canlandığında
daha uzağa taşır; kısa chunk tersi.

**Bu iki karar birlikte bir tuzak üretiyor** ve kural gerektiriyor: önermeler
tanım gereği kısa, chunk'lar uzun. Kütle ham uzunluktan hesaplanırsa önerme
katmanı hiç canlanmaz ya da hiç yayamaz — ilk kararın tüm faydası kaybolur.

**Kural: kütle her zaman kendi katmanı içinde normalize edilir.** Bir önermenin
kütlesi diğer önermelere göre, bir chunk'ın kütlesi diğer chunk'lara göre
hesaplanır. Katmanlar arası bağlarda kütle etkisi devre dışıdır.

Bu kural yazılmazsa iki katmanlı yapı sessizce tek katmanlı gibi davranır ve
hata ayıklaması çok zor olur.
