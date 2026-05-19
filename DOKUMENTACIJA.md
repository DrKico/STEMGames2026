# STEM Games 2026 – Rekonstrukcija oblaka točaka (Point Cloud)

## Sadržaj
1. [Opis problema](#opis-problema)
2. [Pristup rješavanju](#pristup-rješavanju)
3. [Skripte i njihova uloga](#skripte-i-njihova-uloga)
4. [Matematička pozadina](#matematička-pozadina)
5. [Rezultati](#rezultati)
6. [Ograničenja i opažanja](#ograničenja-i-opažanja)
7. [Pokretanje](#pokretanje)
8. [Izlazne datoteke](#izlazne-datoteke)

---

## Opis problema

Zadatak je bio rekonstruirati **3D pozicije piksela** (oblak točaka, eng. *point cloud*) iz četiri skupa fotografija istih scena snimljenih iz različitih kutova i pozicija.

Skupovi slika su:
| Skup | Koordinate kamere | Rezolucija |
|------|-------------------|------------|
| **Box** | Poznate | 1920 × 1080 |
| **Entrance** | Poznate | 1920 × 1080 |
| **Statue** | Nepoznate | 1920 × 1080 |
| **Fountain** | Nepoznate | 3072 × 2048 |

Za skupove **Box** i **Entrance** dane su nam pozicije i orijentacije kamera u tekstualnoj datoteci, što uvelike olakšava rekonstrukciju. Za skupove **Statue** i **Fountain** dane su nam samo unutarnje karakteristike kamere (matrica K), a položaje kamera moramo sami procijeniti.

---

## Pristup rješavanju

### Slučaj 1: Poznate koordinate kamera (Box i Entrance)

Kada su nam poznate pozicija i orijentacija kamere, svaki piksel na slici možemo opisati kao **zraku u 3D prostoru** (eng. *ray*). Ta zraka polazi od položaja kamere i prolazi kroz traženi piksel.

Metoda:
1. Za svaku sliku izvučemo karakteristične točke pomoću algoritma **SIFT** (Scale-Invariant Feature Transform).
2. Za svaki par slika pronađemo odgovarajuće parove točaka (eng. *feature matching*) koristeći Lowe-ov omjer (ratio test, prag 0.80).
3. Za svaki par odgovarajućih točaka iz dvaju kamera, konstruiramo dvije zrake u 3D prostoru koristeći funkciju `GetLineEquation` zadanu u opisu zadatka.
4. Pronađemo **najbližu točku između dviju zraka** (triangulacija). Središte tog najkraćeg segmenta je naša procjena 3D pozicije točke.
5. Odbacujemo točke kod kojih su zrake previše daleko jedna od druge (loši parovi) ili koje su predaleko od kamera.
6. Statistički uklanjamo grube pogreške (eng. *outliers*) koristeći medijan i MAD (Median Absolute Deviation).

### Slučaj 2: Nepoznate koordinate kamera (Statue i Fountain)

Kada koordinate kamera nisu poznate, koristimo tehniku **Structure from Motion (SfM)** uz pomoć biblioteke **pycolmap** (Python sučelje za COLMAP).

Metoda:
1. COLMAP sam pronalazi karakteristične točke i opisnike na svim slikama (SIFT, do 20 000 točaka po slici).
2. Sve parove slika uspoređujemo iscrpno (eng. *exhaustive matching*) kako bismo pronašli odgovarajuće točke.
3. Iz odgovarajućih točaka izračunava se **esencijalna matrica** (Essential Matrix) koja opisuje geometrijski odnos između kamera.
4. Iz esencijalne matrice dobivamo relativnu rotaciju i translaciju kamere (*recoverPose*).
5. COLMAP provodi **inkrementalni SfM** — postupno dodaje slike jednu po jednu, za svaku procjenjujući položaj kamere, te triangulira 3D točke.
6. Na kraju se provodi **bundle adjustment** (globalna optimizacija) koji istovremeno minimizira pogreške projekcije za sve kamere i sve 3D točke.

---

## Skripte i njihova uloga

### `reconstruct_known_cameras.py`
Rekonstrukcija za skupove **Box** i **Entrance** s poznatim kamerama.

Ključne funkcije:
- `parse_camera_file(filepath)` — parsira tekstualnu datoteku s podacima kamera (pozicija, Forward/Right/Up vektori)
- `get_ray_direction(pixel_row, pixel_col, ...)` — implementacija `GetLineEquation` iz zadatka; računa smjer zrake za dani piksel
- `triangulate_rays(o1, d1, o2, d2)` — triangulacija dviju zraka; traži najbližu točku između njih
- `remove_outliers(points)` — statistički filtar grubih pogrešaka (MAD metoda)
- `reconstruct(...)` — glavni tok: učitavanje, ekstrakcija značajki, uparivanje, triangulacija, filtriranje

### `reconstruct_unknown_cameras.py`
Python-bazirana rekonstrukcija (bez COLMAP-a) za **Statue** i **Fountain** — korisna za razumijevanje algoritma.

Ključne funkcije:
- `parse_K(k_file)` — parsira matricu K iz datoteke `K.txt`
- `incremental_sfm(images, K, ...)` — dvofazni SfM: (1) izgradnja lanca poza kamera iz uzastopnih parova, (2) triangulacija unutar kliznog prozora

### `reconstruct_colmap.py`
Rekonstrukcija visoke kvalitete koristeći **pycolmap** za sva četiri skupa.

Ključne funkcije:
- `reconstruct_unknown_cam(...)` — pokretanje punog COLMAP SfM cjevovoda (ekstrakcija → uparivanje → inkrementalno mapiranje s bundle adjustment)
- `extract_colmap_points(reconstruction)` — izvlačenje XYZ+RGB iz COLMAP Reconstruction objekta

### `reconstruct_statue_split.py`
Specijalizirana skripta za **Statue** koja prepoznaje da su slike podijeljene u dvije vizualno različite hemisfere.

### `final_summary.py`
Konsolidira najbolje rezultate iz svih metoda i generira kombinirani vizualni pregled.

---

## Matematička pozadina

### Zraka kamere (Ray equation)

Za kameru s poznatim parametrima, smjer zrake za piksel `(row, col)` računa se kao:

```
coeffRight = 2 * (col - ResX/2 + 0.5) / ResX
coeffUp    = -2 * (row - ResY/2 + 0.5) / ResY * (ResY / ResX)

direction = Forward + coeffRight * Right + coeffUp * Up
```

Ovo pretpostavlja vidno polje (FoV) od 90° i bez distorzije objektiva.

### Triangulacija dviju zraka

Dvije zrake opisane su kao:
```
P1(t) = O1 + t * d1
P2(s) = O2 + s * d2
```

Najbliže točke na dvjema zrakama dobivamo rješavanjem sustava:
```
d1d2 = dot(d1, d2)
denom = 1 - d1d2²
t = (dot(O2-O1, d1) - d1d2 * dot(O2-O1, d2)) / denom
s = (d1d2 * dot(O2-O1, d1) - dot(O2-O1, d2)) / denom
```

Konačna 3D točka je središte između `P1(t)` i `P2(s)`.

### Matrica kamere (Camera Matrix K)

Matrica unutarnjih parametara kamere:
```
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]
```

Za 90° horizontalni FoV i rezoluciju 1920×1080:
- `fx = fy = 960` (žarišna duljina u pikselima)
- `cx = 960`, `cy = 540` (glavna točka = središte slike)

Za Fountain (iz K.txt): `fx=2759.48`, `fy=2764.16`, `cx=1520.69`, `cy=1006.81`

### Esencijalna matrica (Essential Matrix)

Veza između odgovarajućih točaka u dvjema kalibriranim kamerama:
```
x2^T * E * x1 = 0
```

Iz esencijalne matrice dobivamo relativnu rotaciju `R` i translaciju `t` kamere. Translacija je poznata samo do skaliranja (jedinična norma).

### Bundle Adjustment

Globalna nelinearna optimizacija koja minimizira sumu kvadrata pogrešaka reprojekcije:
```
min Σ_i Σ_j || x_ij - π(R_i, t_i, X_j) ||²
```

gdje je `x_ij` opažena 2D točka, a `π(...)` projekcija 3D točke `X_j` kroz kameru `i`.

---

## Rezultati

| Skup | Broj točaka | Metoda | Srednja pogreška reprojekcije |
|------|------------|--------|-------------------------------|
| **Box** | ~2 800 | Ray triangulacija (Python) | — |
| **Entrance** | ~12 000 | Ray triangulacija (Python) | — |
| **Statue** | ~2 500 | COLMAP SfM + Bundle Adjustment | 0.21 px |
| **Fountain** | ~25 000 | COLMAP SfM + Bundle Adjustment | 0.32 px |

Za Box i Entrance koordinate su u **svjetskom koordinatnom sustavu** (iste jedinice kao pozicije kamera u ulaznoj datoteci). Za Statue i Fountain koordinate su u **relativnom koordinatnom sustavu** (COLMAP normalizira skalu).

---

## Ograničenja i opažanja

### Box
Scena je renderirana igrana okolina (drvena kutija na pješčanoj podlozi). Površina kutije ima relativno malo teksture, što SIFT algoritmu otežava pronalazak karakterističnih točaka. Rekonstruirani oblak sadrži točke i s kutije i s pozadinskog zida od opeke.

### Entrance
Bogata scena s mnogo geometrijskih detalja (kipovi, zid, automobil, luk). Daje najbolje rezultate od skupova s poznatim kamerama (~12 000 točaka).

### Statue
Kip je fotografiran u punom krugu od 360°. Slike 1–7 i 18 prikazuju **prednju stranu** kipa (pozadina: zid od opeke s markerima u crno-bijelim uzorcima). Slike 8–17 prikazuju **stražnju stranu** kipa (pozadina: pustinjski krajolik s karakterističnim stijenama).

Ovaj nagli vizualni prijelaz otežava uzastopno uparivanje slika — slike 7→8 i 17→18 dijele malo zajedničkih karakteristika. COLMAP to rješava **zatvaranjem petlje** (eng. *loop closure*): prepoznaje da slike 1 i 18 prikazuju gotovo identičan pogled i koristi tu informaciju za globalno poravnanje.

Uz COLMAP registrirano je 16 od 18 slika (2 slike s prijelaznog dijela nisu mogle biti registrirane).

### Fountain
Realna fotografija (ne renderirana scena) ornamentalne fontane. Bogata tekstura opeke i ukrasa daje izvrsne rezultate — COLMAP registrira svih 11 slika i producira ~25 000 točaka visoke točnosti.

---

## Pokretanje

### Preduvjeti
```bash
pip3 install opencv-python numpy scipy matplotlib pillow pycolmap
```

### Izvođenje

```bash
cd "/Users/kmatic/IdeaProjects/STEMgames 2026"

# Box i Entrance (poznate kamere, svjetske koordinate)
python3 reconstruct_known_cameras.py

# Statue i Fountain (COLMAP, visoka kvaliteta)
python3 reconstruct_colmap.py

# Konsolidacija rezultata i vizualizacija
python3 final_summary.py
```

---

## Izlazne datoteke

### Primarne datoteke za predaju (`final_output/`)

| Datoteka | Opis |
|----------|------|
| `box_final_points.txt` | 3D točke za Box (X Y Z R G B) |
| `entrance_final_points.txt` | 3D točke za Entrance |
| `statue_final_points.txt` | 3D točke za Statue |
| `fountain_final_points.txt` | 3D točke za Fountain |
| `box_final_points.ply` | PLY format za prikaz u MeshLab/CloudCompare |
| `entrance_final_points.ply` | PLY format |
| `statue_final_points.ply` | PLY format |
| `fountain_final_points.ply` | PLY format |
| `all_reconstructions.png` | Kombinirani vizualni pregled sva 4 oblaka |

### Format tekstualne datoteke
```
X Y Z R G B
26.8944 -692.1920 44.6369 131 77 47
99.3281 -712.5330 113.1627 195 141 77
...
```
- `X, Y, Z` — koordinate u 3D prostoru (float)
- `R, G, B` — boja točke preuzeta iz originalne slike (0–255)

### Vizualizacija PLY datoteka
PLY datoteke možete otvoriti u:
- **MeshLab** (besplatno, preporučeno)
- **CloudCompare** (besplatno)
- Python: `pip3 install open3d` i koristiti `open3d.io.read_point_cloud()`
