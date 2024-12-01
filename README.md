1. Clone the code
2. Create python venv
3. Enable venv
4. Run `pip install -r requirements.txt`
5. Run `python edgar_bulk_extractor.py -y [DESIRED_YEAR] -q [DESIRED_QUARTER]`

```
edgar_bulk_extractor.py

ARGS:

-y --year      <int>         (Required)                    Year of interest 
-q --quarter   <int>[1-4]    (Required)                    Quarter of interest 
-t --threads   <int>[1-8]    (Optional. Default: 4)        Threads for multithreaded .nc file processor 
--debug        flag          (Optional.)                   Enable debug single day run. Skips duplicate filings
```
