# 4DN data access after the portal went away

The 4DN data portal, data.4dnucleome.org, stopped answering in November 2026. The domain was
registered again through NameCheap and resolves to a parking host, 104.219.250.37 on
parity.domains name servers, which refuses TLS. The portal's API and its `@@download` links
are gone with it. `validation/studies/fetch.py` builds every 4DN link from that host, so
`python -m validation fetch` fails for every manifest with a 4DN source: the Hi-C, RNAPII and
Micro-C manifests of GM12878, H1ESC and HFFC6.

The files themselves are still served. They live in the public S3 bucket
`4dn-open-data-public`, which answers anonymous listings and range requests:

```
https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/<uuid>/<accession>.<ext>
```

The uuid is the portal's file uuid and is not derivable from the accession. A listing of the
prefix `fourfront-webprod/wfoutput/` finds it; the whole prefix was 34 pages of 1,000 keys on
2026-11-12, and a scan by accession takes about a minute.

```python
q = {"list-type": "2", "prefix": "fourfront-webprod/wfoutput/", "max-keys": "1000", "continuation-token": tok}
urllib.request.urlopen("https://4dn-open-data-public.s3.amazonaws.com/?" + urllib.parse.urlencode(q))
```

Found on 2026-11-12:

| accession | size | key |
|---|---|---|
| 4DNFI1UEG1HD.hic | 22.6 GB | fourfront-webprod/wfoutput/a98ca64a-861a-4a8c-92e9-586af457b1fb/4DNFI1UEG1HD.hic |
| 4DNFIXP4QG5B.mcool | 27.4 GB | fourfront-webprod/wfoutput/d6abea45-b0bb-4154-9854-1d3075b98097/4DNFIXP4QG5B.mcool |

Both read through the website's link route, `web/gnome3d_web/remote.py`: a 1.1 Mb window of
chr1 in 6.2 s from the .hic and 9.6 s from the .mcool.

Needed for the RNAPII and Micro-C inputs. `validation/studies/fetch.py` now resolves a 4DN
accession through `validation/manifests/4dn_keys.tsv`, and an accession missing from the table
is found by the listing and added to it, so a new RNAPII or Micro-C manifest needs nothing more
than its accessions. The portal's md5 and assay metadata are gone with it; fetch checks no md5
for 4DN files. Unit checks in `harness/test_fetch_4dn.py`.
