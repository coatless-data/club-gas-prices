# Club Gas Prices

Posted fuel prices at warehouse-club gas stations, collected four times a day from each
chain's own public website, stored by day, and published as GitHub Releases.

Currently collecting every Costco station in the United States, Canada, Mexico, the
United Kingdom, Australia, Japan and Taiwan. A Sam's Club US feed is implemented and
turned off; see [Sources](#sources).

**Dashboard:** <https://dashboard.thecoatlessprofessor.com/club-gas-prices/>
 · built from this repository's releases by
[coatless-dashboard/club-gas-prices](https://github.com/coatless-dashboard/club-gas-prices)

> [!IMPORTANT]
> Unofficial. Prices are collected from public websites and may differ from the price at
> the pump.
>
> Not affiliated with, endorsed by, or connected to Costco Wholesale Corporation.
>
> Not affiliated with, endorsed by, or connected to Sam's West, Inc., Sam's Club, or
> Walmart Inc.

## The dashboard

The site lives in its own repository and holds no collection code. It reads five files
this repository publishes into the `current` release under a `site-` prefix, each
checksummed in `manifest.json`: `site-meta.json`, `site-latest.json`,
`site-stations.json`, `site-summary-daily.parquet` and `site-history.parquet`.
`publish` builds them from the same frames it writes the dataset from, inside the same
transaction, so the site can never show one capture's map over another capture's
history. What each page does is documented there.

## Getting the data

Everything is published as release assets. Nothing but code, config and a small status
file lives in git.

```
current                 rolling, all-time, marked Latest
├── club-gas-all.parquet            daily grain, every day so far
├── club-gas-all.csv.gz             the same rows as CSV
├── club-gas-all-captures.parquet   capture grain, every row so far
├── club-gas-latest.csv             every row of each station's newest capture
├── stations.csv                      one row per station_key ever seen
├── fx.csv                            one row per capture and non-USD currency
├── site-*.json, site-*.parquet       the five files the dashboard reads
└── manifest.json                     newest status.json, closed periods, SHA-256 of every asset above

data-YYYY-MM            one release per month, a prerelease while the month is open
├── club-gas-YYYY-MM-DD.csv.gz      capture grain, one file per UTC day
├── capture-<capture_id>.tar.gz       one bundle per capture: raw responses, inputs, config, rows
├── manifest-YYYY-MM.json             one entry per capture merged into the month
├── club-gas-YYYY-MM.parquet        daily grain, written when the month closes
├── club-gas-YYYY-MM.csv.gz         the same rows as CSV
└── club-gas-YYYY-MM-captures.parquet   capture grain, written when the month closes

data-YYYY               one release per year, written when the year closes
├── club-gas-YYYY.parquet           daily grain
├── club-gas-YYYY.csv.gz            the same rows as CSV
├── club-gas-YYYY-captures.parquet  capture grain
└── manifest-YYYY.json                SHA-256 of each input month file, both grains
```

Download the whole rolling release with the GitHub CLI:

```bash
gh release download current -R coatless-data/club-gas-prices -D state/current
```

### The two grains

- **Capture grain** — one row per capture, station and grade. Its key is
  (`capture_id`, `station_key`, `grade_raw`), and `station_key` carries the chain. Use
  it to see intraday changes.
- **Daily grain** — one row per (`capture_date`, `station_key`, `grade_raw`), taken
  from the capture with the greatest `captured_at_utc` that UTC day. It carries every
  capture-grain column plus `n_captures`, `price_min` and `price_max` for that day.

### Schema (capture grain, in file order)

| Column | Type | Description |
|---|---|---|
| `capture_id` | string | `YYYY-MM-DDTHHMMZ`, the UTC minute the capture started. |
| `capture_date` | date | The UTC date of `capture_id`. Daily files are keyed by it. |
| `captured_at_utc` | timestamp | When this country's last price response arrived. |
| `local_date` | date | The date of `captured_at_utc` in `timezone`. |
| `country` | string | `US`, `CA`, `MX`, `GB`, `AU`, `JP`, `TW`. Puerto Rico is `US` with `region = "PR"`. |
| `brand` | string | The chain: `COSTCO` or `SAMS`. |
| `station_key` | string | `{country}-{brand}-{source_station_id}`, e.g. `US-COSTCO-1364`, `JP-COSTCO-Tomiya`. |
| `source_station_id` | string | The station's id at its source. |
| `source` | string | `costco-us-gasprices`, `costco-ca-lookup-us`, `costco-ca-lookup`, `costco-ca-gasprices`, `costco-occ` or `sams-clubfinder`. |
| `name` | string | English or romanized name. |
| `name_local` | string, nullable | Native-script name (JP, TW). |
| `address` | string, nullable | Street address. |
| `city` | string, nullable | Town or municipality. |
| `region` | string, nullable | State, province, prefecture or equivalent. |
| `postcode` | string, nullable | Postal code. |
| `lat`, `lon` | float64, nullable | WGS84. |
| `timezone` | string | IANA name. |
| `grade_raw` | string | The label the source used. |
| `grade` | string | `regular`, `premium`, `diesel` or `other`. |
| `price_raw` | string | Exactly as published. |
| `price` | float64 | Parsed, in the published unit. |
| `price_unit` | string | `USD/gal`, `USD/L`, `CAD/L`, `MXN/L`, `GBp/L`, `AUD/L`, `JPY/L`, `TWD/L`. |
| `currency` | string | ISO 4217 code of the major unit. |
| `price_local_per_litre` | float64 | Major currency units per litre. |
| `fx_usd_per_unit` | float64, nullable | `1 / units_per_usd` at capture time. |
| `fx_rate_date` | date, nullable | The reference date of that rate. |
| `fx_source` | string, nullable | Where the rate came from. |
| `fx_fetched_at_utc` | timestamp, nullable | When the rate was fetched. |
| `price_usd_per_litre` | float64, nullable | `price_local_per_litre × fx_usd_per_unit`. |
| `price_usd_per_gallon` | float64, nullable | `price_usd_per_litre × 3.785411784`. |

`stations.csv` holds one row per `station_key` ever seen: identity and metadata,
`grades_seen` (pipe-joined `grade_raw` values), `first_seen_utc`, `last_seen_utc`,
`status` (`active` or `missing`) and `superseded_by`.

`fx.csv` holds one row per capture and non-USD currency: `capture_id`, `currency`,
`units_per_usd`, `fx_usd_per_unit`, `fx_rate_date`, `fx_source`, `fx_fetched_at_utc`.

## Data conventions

**Units.** Prices are stored exactly as published in `price_raw`, and again as a number
in `price` with the unit in `price_unit`. The US posts USD per US gallon, except Puerto
Rico, which posts USD per litre; every other country posts per litre, and the UK posts
pence. `price_local_per_litre` normalizes all of them: `USD/gal` is divided by
3.785411784, `GBp/L` by 100, and the rest are already per litre. Puerto Rico's per-litre
unit is inferred, not stated by the source: #335 posted 1.097 for regular on 2026-09-15,
about 4.15 per gallon, and Puerto Rico retail is priced per litre.

**Grades.** `grade_raw` is the source's own label and is never interpreted away.
`grade` is the mapped bucket, by exact label, from `config/grades.csv`:

| Feed | Labels | Notes |
|---|---|---|
| US-COSTCO | `regular`, `premium`, `diesel`, `clear` | `clear` maps to `other`. The source states no octane or blend. |
| CA-COSTCO | `regular`, `premium`, `diesel` | The source states no octane. |
| MX-COSTCO | `Regular`, `Premium` | No diesel sold. Regular is at least 87 ([RON+MON]/2), reported by CNE. |
| GB-COSTCO | `5301`, `5302`, `5303` | Unleaded, premium unleaded and premium diesel; E10, E5 97 and B7 are reported, not stated. |
| AU-COSTCO | `Unleaded 91`, `E10`, `Premium 98`, `Diesel` | `E10` and `Unleaded 91` both map to `regular`; `Unleaded 91` wins when a station lists both. Diesel is premium diesel (reported). |
| JP-COSTCO | `Regular`, `Premium`, `Diesel`, `Kerosene` | Kerosene is heating fuel: it maps to `other` and is excluded from comparisons. |
| TW-COSTCO | `95`, `98`, `Diesel` | The names are octane numbers. No 92 is sold. |

Each row of `config/grades.csv` carries `spec`, `spec_source` and `spec_source_url`.
`spec_source` is `source` when the retailer itself states the specification, and `reported`
when it comes from a third party, in which case the URL is required. The About page
shows the attribution. Grades are not equivalent across countries; comparing "regular"
between them compares different fuels.

**Exchange rates.** One rate per currency is fetched at capture time and stored with
every row of that capture. `units_per_usd` is the provider's raw rate and
`fx_usd_per_unit` is its reciprocal; both are written with 10 significant digits, because
rounding a rate like 154.24 JPY per USD to 4 decimals would shift the converted price
by 0.26%. The primary source is the Frankfurter v2 daily reference rate; if it fails,
a dated jsDelivr currency-api file for the capture date or the day before; if that
fails too, the newest non-carried row within 7 days from `fx.csv`, copied with
`fx_source = "carried-forward:<original source>"`. If every option fails, the USD
columns are null for that capture and the prices are still published. USD rows use
`fx_usd_per_unit = 1.0` and `fx_source = "identity"`.

**Placeholders and pre-opening prices.** Sources publish prices for warehouses that
have no gas station or have not opened. Rows are dropped when the warehouse has no gas
service, when its opening date is after the capture date, when the station has no gas
hours, when the price cannot be parsed, or when the price falls outside the per-unit
sanity bounds that reject placeholders such as the Canadian `.959`. Every drop is
counted by reason in the capture status.

**UTC days and local dates.** `capture_date` is a UTC date and is what the daily files
are keyed by. `local_date` is the same instant in the station's own timezone. For Japan,
Taiwan and Australia the last capture of a UTC day lands early the next local morning,
so the two dates differ for part of each day. Charts use `capture_date`.

## Repository layout

```
.
├── pyproject.toml            # uv project; console script `club-gas`
├── uv.lock
├── .python-version           # 3.13
├── src/club_gas/
│   ├── cli.py                # capture | publish | close-periods | alerts | rebuild | discover | site-data
│   ├── config.py             # load + validate config/*
│   ├── http.py                # shared Client: headers, deadlines, retries, pacing, block signals
│   ├── fx.py                 # exchange rates
│   ├── sources/               # base.py (the feed registry), us.py, ca.py, costco_lookup.py, occ.py, sams.py
│   ├── normalize.py          # filters, grades, units, bounds, FX columns
│   ├── schema.py              # column specs, dtypes, writers, validators
│   ├── capture.py             # capture orchestration
│   ├── checks.py              # statuses, counters, fingerprints
│   ├── store.py               # ReleaseStore; GitHubReleaseStore; LocalReleaseStore
│   ├── publish.py             # daily file, month manifest, current
│   ├── rollup.py              # daily grain; month and year close
│   ├── rebuild.py             # re-parse stored bundles
│   ├── issues.py              # open/refresh/close issue helper
│   ├── alerts.py              # capture-related issues
│   ├── discover.py            # monthly US id sweep
│   └── sitedata.py            # the dashboard's five files, uploaded to `current`
├── config/
│   ├── http.toml              # UA, deadlines, budgets, retries, pacing
│   ├── countries.toml         # URLs, params, units, region→timezone tables, bounds, floors
│   ├── feeds.toml             # per-feed overrides, including whether a feed runs at all
│   ├── grades.csv              # brand,country,grade_raw,grade,priority,label,spec,spec_source,spec_source_url
│   ├── us_extra_ids.csv        # stations that price live but are missing from ecom-api
│   ├── station_links.csv       # old_station_key,new_station_key,effective_date,note
│   └── site.toml               # basemap providers, notice text
├── status/latest.json         # committed after every non-dry capture
├── tests/                     # unit tests, fixtures/
└── .github/
    ├── dependabot.yml
    └── workflows/              # test.yml, capture.yml, discover.yml, rebuild.yml
```

## Running it

Collect without publishing anything. Reads from the GitHub release store need no token:

```bash
uv run club-gas capture --out out
```

Publish into a local directory instead of GitHub:

```bash
CLUB_GAS_STORE=local:./releases uv run club-gas publish out/capture
```

Publishing to the GitHub store from a local machine is unsupported: release writes
require both `GITHUB_TOKEN` and `CLUB_GAS_WRITER=1`, which only `capture.yml` and
`rebuild.yml` set.

Check what the dashboard would receive, without publishing:

```bash
gh release download current -R coatless-data/club-gas-prices -D state/current
uv run club-gas site-data --current state/current --out site/data
```

To render them, clone the dashboard and point it at that directory; its README has
the steps.

Run the checks:

```bash
uv sync --locked
uv run ruff check
uv run ruff format --check
uv run pytest
actionlint
```

## Sources

A capture runs one **feed** per chain and country, keyed `<COUNTRY>-<BRAND>`, so a feed
that fails or is turned off cannot affect any other. One request per feed per capture,
except the United States, which needs a metadata request and about 60 batched price
requests. At most one request per second per host.

| Feed | Endpoint |
|---|---|
| US-COSTCO | `ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json` for metadata, then `www.costco.com/AjaxGetGasPricesService?warehouseid=<10 ids>` for prices. Falls back to `www.costco.ca/AjaxWarehouseBrowseLookupView?...&countryCode=US`. |
| CA-COSTCO | `www.costco.ca/AjaxWarehouseBrowseLookupView?hasGas=true&populateWarehouseDetails=true&countryCode=CA`. Falls back to `www.costco.ca/AjaxGetGasPricesService`. |
| MX-COSTCO | `www.costco.com.mx/rest/v2/mexico/stores?fields=FULL&...` |
| GB-COSTCO | `www.costco.co.uk/rest/v2/uk/stores?fields=FULL&...` |
| AU-COSTCO | `www.costco.com.au/rest/v2/australia/stores?fields=FULL&...` |
| JP-COSTCO | `www.costco.co.jp/rest/v2/japan/stores?fields=FULL&...` |
| TW-COSTCO | `www.costco.com.tw/rest/v2/taiwan/stores?fields=FULL&...` |
| US-SAMS | `www.samsclub.com/api/node/vivaldi/browse/v2/clubfinder/list` for the roster, then the `HyperLocalPagesTempo` GraphQL query per club for prices. **Off** in `config/feeds.toml`: hosted runners are refused by the site's bot protection (HTTP 412), and the roster's own `gasPrices` field is stale by a third or more, so only the per-club query gives a current price. See `docs/samsclub-vivaldi-api.md`. |
| FX | `api.frankfurter.dev/v2/rates?base=USD&quotes=CAD,MXN,GBP,AUD,JPY,TWD`, then `cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@<date>/v1/currencies/usd.json` |

No source publishes a timestamp with its prices, prices change during the day, and the
storefront responses declare CDN caching of up to 90 minutes. Four captures a day and
`captured_at_utc` on every row are how that uncertainty is exposed rather than hidden.
These are one chain's posted prices, not national averages, and grades are not
equivalent across chains or countries.

**Basemap.** The map uses CARTO raster basemaps when a `CARTO_BASEMAP_KEY` repository
variable is set: © OpenStreetMap contributors, © CARTO
(<https://www.openstreetmap.org/copyright>, <https://carto.com/attributions>).
Without a key it falls back to OpenStreetMap tiles: © OpenStreetMap contributors
(<https://www.openstreetmap.org/copyright>), a best-effort service with no SLA.

## License

The code is MIT licensed. The data files carry no license claim and include the notice
above; they are derived from the retailers' public websites.
