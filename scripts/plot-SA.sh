#!/usr/bin/bash

# current one, will be used for the report. consistent with the result based on 4c3f619
uv run python scripts/plot_gsa_sobol.py     --in ./data-SA/24253272-70b12f2-sobol/gsa_sobol.csv              --out ./data-SA/24253272-70b12f2-sobol.png
uv run python scripts/plot_gsa_sobol.py     --in ./data-SA/24253272-70b12f2-sobol/gsa_sobol_S2.csv           --out ./data-SA/24253272-70b12f2-sobol-S2.png
# additional run. the result is consistent
uv run python scripts/plot_gsa_sobol.py     --in ./data-SA/24255073-9b1b60f-sobol/gsa_sobol.csv              --out ./data-SA/24255073-9b1b60f-sobol.png
uv run python scripts/plot_gsa_sobol_s2.py  --in ./data-SA/24255073-9b1b60f-sobol/gsa_sobol_S2.csv           --out ./data-SA/24255073-9b1b60f-sobol-S2.png

# current one, will be used for presentation and report until "sequential data" is ready
uv run python scripts/plot_gsa_morris.py    --in ./data-SA/24216220-4c3f619-morris-parallel/gsa_morris.csv   --out ./data-SA/24216220-4c3f619-morris-parallel.png
uv run python scripts/plot_gsa_sobol.py     --in ./data-SA/24216291-4c3f619-sobol-parallel/gsa_sobol.csv     --out ./data-SA/24216291-4c3f619-sobol-parallel.png
uv run python scripts/plot_gsa_sobol_s2.py  --in ./data-SA/24216291-4c3f619-sobol-parallel/gsa_sobol_S2.csv  --out ./data-SA/24216291-4c3f619-sobol-S2-parallel.png

# the below are reference to ensure the parallel data are correct
uv run python scripts/plot_gsa_morris.py    --in ./data-SA/24217522-11cc007-morris-128sample/gsa_morris.csv  --out ./data-SA/24217522-11cc007-morris-128sample.png
uv run python scripts/plot_gsa_sobol.py     --in ./data-SA/24217056-11cc007-sobol-128sample/gsa_sobol.csv    --out ./data-SA/24217056-11cc007-sobol-128sample.png
uv run python scripts/plot_gsa_sobol_s2.py  --in ./data-SA/24217056-11cc007-sobol-128sample/gsa_sobol_S2.csv --out ./data-SA/24217056-11cc007-sobol-S2-128sample.png

uv run python scripts/plot_gsa_morris.py    --in ./data-SA/24214965-6896023-morris-parallel/gsa_morris.csv   --out ./data-SA/24214965-6896023-morris-parallel.png
uv run python scripts/plot_gsa_sobol.py     --in ./data-SA/24214967-6896023-sobol-parallel/gsa_sobol.csv     --out ./data-SA/24214967-6896023-sobol-parallel.png
#uv run python scripts/plot_gsa_sobol_s2.py  --in ./data-SA/24214967-6896023-sobol-parallel/gsa_sobol_S2.csv  --out ./data-SA/24214967-6896023-sobol-S2-parallel.png

