# Amazon S3 Glacier

Source: https://en.wikipedia.org/wiki/Amazon_S3_Glacier
Retrieved at: 2026-06-22T21:39:01Z
Revision ID: 1334963224
Raw text SHA256: a7d19bbbefc37d7ce2a7dc0312c24c066e56be67499bade6e4835788164af5b0
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
Amazon S3 Glacier is an online file storage web service that provides storage for data archiving and backup.
Glacier is part of the Amazon Web Services suite of cloud computing services, and is designed for long-term storage of data that is infrequently accessed and for which standard retrieval latency times of 12 to 48 hours are acceptable. Base monthly storage costs are $0.00099/GB or about $1/TB (East Coast USA Prices). This is substantially cheaper than the Simple Storage Service (S3) Standard tier with its nearly instant egress times.
Amazon aims to move customers from on-premises tape backup drives to cloud-based backup.

Storage
The storage technology used by Glacier, indeed for all of S3, is an Amazon trade secret. It is subject to speculation.
Amazon officially states in their S3 FAQS:

ZDNet says, that according to private e-mail, Glacier runs on "inexpensive commodity hardware components". In 2012, ZDNet quoted a former Amazon employee as saying that Glacier is based on custom low-RPM hard drives attached to custom logic boards where only a percentage of a rack's drives can be spun at full speed at any one time. Similar technology is also used by Facebook.
There is some belief among users that the underlying hardware used for Glacier storage is tape-based, because Amazon has positioned Glacier as a direct competitor to tape backup services both on-premises and cloud-based.  This is seems supported by the fact that Glacier has a standard long archive retrieval of up to 12 hours, called "thawing", and a pricing model that discourages frequent data retrieval (the effect of batching requests to reduce wear on mechanical equipment).
The Register claimed in 2012 that Glacier ran on Spectra T-Finity tape libraries with LTO-6 tapes. Others have conjectured Amazon uses off-line shingled magnetic recording hard drives, multi-layer Blu-ray optical discs, or an alternative proprietary storage technology. Data storage consultant Robin Harris speculated that the storage is based on cheap optical disks such as Blu-ray, based on hints from public sources.

Cost
Glacier has at least four costs associated. The following prices reflect East Coast USA as of 2026 in USD, and is representative of how pricing is segmented into categories and their relative differences. Other regions may have different prices.

A per-file upload fee of 5 cents per 1,000 files
A per-month storage fee of $0.00099/GB or about $1/TB
A thaw fee. Standard is 0.02 per GB (12 hour guarantee). Bulk is 0.0025/GB (48 hour guarantee).
An egress (download) fee of 0.09 per GB.
The prices for monthly storage have dropped over time: when Glacier launched in 2012, the storage charge was set to 1 cent per gigabyte per month. This was reduced to 0.7 cents in September 2015 and to 0.4 cents in December 2016.
Glacier previously charged for retrievals based on peak monthly retrieval rate, meaning that (ignoring the free tier) if you downloaded four gigabytes in four hours, it would cost the same as if you downloaded 720 gigabytes in 720 hours, in a 30-day month. This made it cheaper to spread out data retrievals over a long period of time, but failing to do so could result in a surprisingly large bill. In one case, a user stored 15 GB of data in Glacier, retrieved 693 MB for testing, and ended up being charged for 126 GB due to retrieval rate calculation. This pricing policy was widely regarded as a "time bomb" set to go off on retrieval.
In 2016, AWS revised their retrieval pricing model. The new model bases the retrieval fee on the number of gigabytes retrieved. This can amount to a 99% price cut for users who perform only one Glacier retrieval in a month. At the same time, AWS introduced new methods of retrieval that take different amounts of time. An expedited retrieval costs one cent per request and three cents per gigabyte, and can retrieve data in one to five minutes. A standard retrieval costs five cents per thousand requests and one cent per gigabyte, and takes three to five hours. A bulk retrieval costs 2.5 cents per thousand requests and 0.25 cents per gigabyte, and takes seven to twelve hours. AWS also introduced provisioned capacity for expedited retrievals, each unit of which costs $100 per month and guarantees at least three expedited retrievals every five minutes, and up to 150 MB/s of retrieval bandwidth. Without provisioned capacity, expedited retrievals are done on a capacity available basis.
Data deleted from Glacier less than 180 days after being stored incurs a charge equal to the cost of storage for the remainder of the 180 days. In effect, the user pays for 180 days minimum. This move was designed to discourage the service's use in cases where Amazon's other storage offerings (e.g. S3) are more appropriate for real-time access. After 180 days, deletion from Glacier has no penalty.
Retrieving data from Glacier is a two-step process. The first step is to retrieve the data into a staging area, where it stays for 24 hours. The second step is to download the data from the staging area, which incurs bandwidth charges.
In 2019, Glacier migrated from the old "Vault" model to a storage class in the S3 product line. This made it easier to access and manage through standard S3 APIs and tools.

References

External links
Official website
