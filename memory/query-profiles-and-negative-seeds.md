---
name: query-profiles-and-negative-seeds
description: Sorgu tipine göre 2-3 hazır yayılma profili; olumsuz istekler için enerji emen negatif tohum
metadata:
  type: project
---

Her istek aynı şekilde değil, ve her istek pozitif değil.

**1. Sorgu profilleri.** Kesin bir olgu sorusu **küçük, hızlı sönen** bir top
ister (kesinlik). Keşif sorusu **büyük, yavaş sönen** bir top ister (kapsam).
Tek bir global `damping` bu ikisini aynı anda iyi yapamaz.

Karar: config'te tanımlı **2-3 hazır profil** — `precise`, `explore`, `compare`.
Her profil kendi damping, eşik ve seed genişliği setini taşır. Seçimi çağıran
yapar.

Elenen alternatif: profili LLM seçsin. Reddedildi çünkü `core/`'a LLM bağımlılığı
sokar ve mimari kuralı bozar ([[architecture-boundaries]]). Çağıran zaten sorgu
tipini biliyorsa gereksiz bir çağrı olur.

**2. Negatif tohum (anti-enerji).** "X hariç", "Y olmadan", "bunun dışında" gibi
istekler için **enerji emen** bir tohum enjekte edilir.

Klasik filtrelemeden farkı kritik: filtre sadece istenmeyen düğümü sonuçtan
çıkarır. Negatif tohum, **o bölgeye giden yolları da söndürür** — yani X'e komşu
olduğu için gelmiş, X'ten başka bir şey anlatmayan 5 chunk da temizlenir. Filtre
bunu yapamaz çünkü onların neden geldiğini bilmez.

Metafora da oturuyor: pozitif ve negatif yük. Aynı mekanizma
[[contradiction-handling]] içinde de kullanılıyor.
