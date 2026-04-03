# Tuning I/O výkonu md RAID5 na inferno-7

Datum: 2026-04-01
Stroj: inferno-7 (256 CPU, 4x NVMe 3.6TB v md RAID5, XFS, Docker overlay2)

## Problém

Docker kontejnery (zejména ComfyUI a nově startující model kontejnery) si při
vysokém I/O vzájemně blokovaly zápisy. Latence zápisů na md0 dosahovala
**645 ms** a queue depth přes **2200 požadavků**, přestože jednotlivé NVMe disky
jely na pouhých 15–20 % využití. Disky tedy nebyly bottleneck — bottleneck byl
v md RAID5 vrstvě samotné.

## Konfigurace

- 4x NVMe SSD (Samsung 990 PRO 4TB, Seagate ZP4000, Kingston SFYRD4000,
  Samsung MZQL2) v md RAID5
- chunk size 512 KB, XFS filesystem
- Docker overlay2 storage driver
- Mount: /var/lib/docker na /dev/md0p1

## Diagnostika a kroky

### Krok 1: Identifikace kdo zapisuje

Nástroj: `iotop -baoP`

Zjištění: `dockerd` (PID 3053442) zapisoval až 1.5 GB/s. Dva ComfyUI kontejnery
(sc2nk20b0evw9u-3 a coosz1s4evlmar-3) zapsaly za 2 týdny po ~337 GB každý.
Nově startující kontejner (Qwen VL model) způsoboval aktuální burst stahováním
model layers.

### Krok 2: Identifikace kde je bottleneck

Nástroj: `iostat -xm`

Zjištění:
- md0 w_await: 372–645 ms (extrémně vysoká latence zápisů)
- md0 aqu-sz: 2221 (obrovská fronta čekajících požadavků)
- NVMe w_await: 0.14–1.87 ms (disky samy o sobě rychlé)
- NVMe %util: 15–20 % (disky nevytížené)

Závěr: Disky čekaly na md vrstvu, ne naopak.

### Krok 3: stripe_cache_size 256 → 8192 (později 32768)

Soubor: `/sys/block/md0/md/stripe_cache_size`
Původní hodnota: **256** (kernel default)

md RAID5 zpracovává I/O po "stripech" (512 KB across disků). Stripe cache drží
aktivní stripy v paměti. Při hodnotě 256 se cache okamžitě zaplnila a nové
zápisy se řadily do fronty. Proto ten queue depth 2221.

Zvýšení na 32768 umožňuje více současně zpracovávaných stripů. Paměťová náročnost
je ~1 GB RAM (32768 × 512K chunk / počet disků), což je na tomto stroji zanedbatelné.

Efekt: Queue depth klesl ~10x, latence na polovinu.

### Krok 4: read_ahead_kb 3072 → 512

Soubor: `/sys/block/md0/queue/read_ahead_kb`
Původní hodnota: **3072 KB**

Read-ahead je optimalizace pro sekvenční čtení — kernel předčítá data dopředu.
Docker overlay2 workloady jsou převážně random I/O (stahování layers, zápisy
do kontejnerů, model inference). Vysoký read_ahead plýtval šířkou pásma čtením
dat, která se nikdy nevyužila.

Efekt: Snížení zbytečného diskového provozu.

### Krok 5: Odstranění write-intent bitmap

Příkaz: `mdadm --grow /dev/md0 --bitmap=none`
Původní stav: **bitmap zapnutý** (28 stránek, 64 MB chunk)

Write-intent bitmap sleduje, které oblasti pole se změnily od posledního
synchronizovaného stavu. Při každém zápisu se musí NEJDŘÍVE aktualizovat bitmap
a POTOM zapsat data. Tato serializace přidávala latenci ke každému zápisu.

Trade-off: Bez bitmapy bude po nečistém vypnutí (výpadek proudu) resync celého
pole pomalejší — musí projít celých 10.5 TB místo jen změněných oblastí. Pokud
je stroj na UPS nebo je akceptovatelný delší resync, je to bezpečné.

Efekt: Snížení write latence odstraněním serializačního bodu.

### Krok 6: group_thread_cnt 0 → 8

Soubor: `/sys/block/md0/md/group_thread_cnt`
Původní hodnota: **0** (= veškerá práce na jednom md0_raid5 vlákně)

TOTO BYL HLAVNÍ PROBLÉM.

md RAID5 musí pro každý zápis spočítat paritu (XOR přes všechny disky). Při
group_thread_cnt=0 tuto práci dělá JEDINÉ kernel vlákno (md0_raid5, PID 2167).
Toto vlákno běželo na **95.8 % CPU** — na 256-jádrovém stroji jeden core na 100 %
bottleneckoval celý 4-diskový RAID array.

Proto disky jely na 15–20 % — nedostávaly dost požadavků, protože je jeden
CPU thread nestíhal připravovat.

Zvýšení na 8 threadů rozložilo parity výpočty na více jader. CPU zátěž md
vzrostla celkově (z 2 % na ~16 % system), ale rozloženě přes více cores.

Efekt:
- md0_raid5 CPU: 95.8 % → 19.2 %
- md0 w throughput: 675 MB/s → 1248 MB/s (téměř 2x)
- NVMe utilization: 15–20 % → 35–55 % (konečně využité)
- md0 w_await: 645 ms → 165 ms (4x zlepšení)

### Krok 7: NVMe I/O scheduler none → BFQ

Soubor: `/sys/block/nvme*n1/queue/scheduler`
Původní hodnota: **none**

BFQ (Budget Fair Queueing) scheduler umožňuje cgroups v2 io.weight —
kernel férově rozděluje diskový čas mezi kontejnery podle váhy. Bez BFQ
(scheduler=none) cgroups I/O weight nefunguje a kontejnery soutěží neregulovaně.

S BFQ a defaultním weight=100 na všech kontejnerech:
- Když je disk volný, kontejner dostane veškerý dostupný throughput
- Když více kontejnerů soutěží, dělí se férově podle váhy
- Nové kontejnery (i ty spuštěné RunPodem) automaticky dostanou default weight

Efekt: Žádný měřitelný dopad na throughput, ale férové sdílení I/O.

## Výsledné srovnání

| Metrika             | Před tunigem  | Po tuningu    | Zlepšení |
|---------------------|---------------|---------------|----------|
| md0 w_await         | 645 ms        | 165 ms        | 4x       |
| md0 w throughput    | 675 MB/s      | 1248 MB/s     | 1.85x    |
| md0 queue depth     | 2221          | 232           | 10x      |
| md0 %util           | 75 %          | 33 %          | 2.3x     |
| NVMe %util          | 15–20 %       | 35–55 %       | ~3x      |
| I/O férovost        | žádná         | BFQ weight    | —        |

## Persistence

Změny jsou uloženy v:
- `/etc/rc.local` — md0 parametry (stripe_cache, read_ahead, group_thread_cnt, bitmap)
- `/etc/udev/rules.d/60-nvme-bfq.rules` — BFQ scheduler na NVMe
- BFQ modul se loaduje v rc.local (`modprobe bfq`)

## Zbývající omezení

Latence md0 (165 ms) je stále vyšší než latence samotných NVMe (5–13 ms).
To je fundamentální vlastnost RAID5 — každý zápis vyžaduje read-modify-write
cyklus na paritu (4 diskové operace na 1 logický zápis). Toto nelze eliminovat
bez změny RAID levelu (např. RAID10 nemá parity overhead, ale nabízí jen 50 %
kapacity místo 75 %).
