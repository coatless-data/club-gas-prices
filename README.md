# costco-gas-prices

Posted fuel prices at every Costco gas station in the United States, Canada,
Mexico, the United Kingdom, Australia, Japan and Taiwan. Prices are captured
four times a day and published as GitHub Releases, with a public dashboard on
top of them.

- Dashboard: <https://dashboard.thecoatlessprofessor.com/costco-gas-prices/>
- Data releases: <https://github.com/coatless-dashboard/costco-gas-prices/releases>

> [!IMPORTANT]
> Unofficial. Not affiliated with, endorsed by, or connected to Costco
> Wholesale Corporation. Prices are collected from Costco's public websites and
> may differ from the price at the pump.

## Layout

```
.
├── config/                 # URLs, units, grade table, bounds, basemap providers
├── src/costco_gas/         # the collector, roll-ups and site-data builder
├── status/latest.json      # committed after every capture
├── site/                   # Quarto dashboard sources
├── tests/                  # pytest suite; tests/smoke is CI only
└── .github/workflows/      # Test, Capture, Render, Discover, Rebuild
```

## Configuration

| File | Holds |
|---|---|
| `config/http.toml` | User-Agent, timeout profiles, retries, pacing, budgets, block signals |
| `config/countries.toml` | Per-country URLs and parameters, units, sanity bounds, station floors, region to timezone tables |
| `config/grades.csv` | Every source grade label, the grade it maps to, and its fuel specification with attribution |
| `config/us_extra_ids.csv` | US warehouse IDs that price live but are missing from Costco's own station list |
| `config/station_links.csv` | Station relocations: old key to new key |
| `config/site.toml` | Basemap providers and the notice text shown on every page |

## Running it

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest
```

## Sources

Every price comes from Costco's own public websites: `ecom-api.costco.com` and
`www.costco.com` for the United States, `www.costco.ca` for Canada, and the SAP
storefront `stores` endpoints for Mexico, the United Kingdom, Australia, Japan
and Taiwan. Exchange rates come from Frankfurter. The dashboard's basemap is
either CARTO or OpenStreetMap; both are credited in the page footer.

## Licence

The code is MIT (see `LICENSE`). The data files carry no licence claim and
include the notice above.
