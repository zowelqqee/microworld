# Amazon DynamoDB

Source: https://en.wikipedia.org/wiki/Amazon_DynamoDB
Retrieved at: 2026-06-22T21:38:07Z
Revision ID: 1358898013
Raw text SHA256: 837b2cd69ef3494072b21ab19b1724003cef7d7029b51003cadb8e7c5091fec1
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
Amazon DynamoDB is a managed NoSQL database service provided by Amazon Web Services (AWS). It supports key-value and document data structures and is designed to handle a wide range of applications requiring scalability and performance.

History
Werner Vogels, CTO at Amazon.com, provided a motivation for the project in his 2012 announcement. Amazon began as a decentralized network of services. Originally, services had direct access to each other's databases. When this became a bottleneck on engineering operations, services moved away from this direct access pattern in favor of public-facing APIs. Still, third-party relational database management systems struggled to handle Amazon's client base. This culminated during the 2004 holiday season, when several technologies failed under high traffic.
Traditional databases often split data into smaller pieces to save space, but combining those pieces during searches can make queries slower. Many of Amazon's services demanded mostly primary-key reads on their data, and with speed a top priority, putting these pieces together was extremely taxing.
Content with compromising storage efficiency, Amazon's response was Dynamo: a highly available key–value store built for internal use. Dynamo, it seemed, was everything their engineers needed, but adoption lagged. Amazon's developers opted for "just works" design patterns with S3 and SimpleDB. While these systems had noticeable design flaws, they did not demand the overhead of provisioning hardware and scaling and re-partitioning data. Amazon's next iteration of NoSQL technology, DynamoDB, automated these database management operations.

Overview
DynamoDB organizes data into tables, which are similar to spreadsheets. Each table contains items (rows), and each item is made up of attributes (columns). Each item has a unique identifier called a primary key, which helps locate it within the table.

DynamoDB Tables
A DynamoDB Table is a logical grouping of items, which represent the data stored in this Table. Given the NoSQL nature of DynamoDB, the Tables do not require that all items in a Table conform to some predefined schema.

DynamoDB Items
An Item in DynamoDB is a set of attributes that can be uniquely identified in a Table. An Attribute is an atomic data entity that in itself is a Key-Value pair. The Key is always of String type, while the value can be of one of multiple data types.
An Item is uniquely identified in a Table using a subset of its attributes called Keys.

Keys In DynamoDB
A Primary Key is a set of attributes that uniquely identifies items in a DynamoDB Table. Creation of a DynamoDB Table requires definition of a Primary Key. Each item in a DynamoDB Table is required to have all of the attributes that constitute the Primary Key, and no two items in a Table can have the same Primary Key. Primary Keys in Dynamo DB can consist of either one or two attributes.
When a Primary Key is made up of only one attribute, it is called a Partition Key. Partition Keys determine the physical location of the associated item. In this case, no two items in a table can have the same Partition Key.
When a Primary Key is made up of two attributes, the first one is called a "Partition Key" and the second is called a "Sort Key". As before, the Partition Key decides the physical Location of Data, but the Sort Key then decides the relative logical position of associated item's record inside that physical location. In this case, two items in a Table can have the same Partition Key, but no two items in a partition can have the same Sort Key. In other words, a given combination of Partition Key and Sort Key is guaranteed to have at most one item associated with it in a DynamoDB Table.

DynamoDB Data Types
DynamoDB supports numerical, String, Boolean, Document, and Set Data Types.

DynamoDB Indices
Primary Key of a Table is the Default or Primary Index of a DynamoDB Table.
In addition, a DynamoDB Table can have Secondary Indices. A Secondary Index is defined on an attribute that is different from Partition Key or Sort Key as the Primary Index.
When a Secondary Index has same Partition Key as Primary Index but a different Sort Key, it is called as the Local Secondary Index.
When Primary Index and Secondary Index have different Partition Key, the Secondary index is known as the Global Secondary Index.

Architectural patterns in DynamoDB data modeling
DynamoDB data modeling patterns are architectural approaches used in Amazon DynamoDB, a NoSQL database service designed for distributed systems. These patterns address various data organization challenges and include "Single Table Design", which consolidates related data while adhering to DynamoDB's 400KB item size limit; "Multiple Table Design", which separates data into distinct tables based on access patterns and data model differences; and Hybrid Design, which blends both approaches to balance flexibility and efficiency.
Additional patterns described in AWS documentation include "Event Sourcing", where data changes are stored as immutable events, enabling historical state reconstruction; "Materialized Views", which simplify analytical queries through pre-computed aggregations, often implemented via DynamoDB Streams, application-level processing, or periodic batch updates using Lambda functions. As well as "Time-Series Design", optimized for workloads like logging and metrics, typically using a partition key for entity identification and a sort key representing timestamps to efficiently query time-based datasets.
Each pattern addresses specific technical requirements. "Single Table Design" can optimize query efficiency by co-locating related data under the same partition key to reduce access latency. "Multiple Table Design" enables separation of concerns by isolating data into purpose-specific tables with distinct access patterns. "Event Sourcing" preserves a historical log of state changes, often implemented with immutable data storage. "Materialized Views" simplify complex analytical queries through pre-aggregation strategies tailored to access patterns. "Time-Series Design" uses partitioning and sorting strategies to efficiently store and query large volumes of temporal data.

Performance limitations of DynamoDB's latency claims
Amazon DynamoDB's claim of single-digit millisecond latency primarily applies to simple operations such as GetItem and PutItem, which retrieve or modify individual items using their primary keys. This reflects the average latency under ideal conditions, such as even partition distribution and sufficient throughput provisioning, and does not account for transport overhead incurred during communication with the DynamoDB endpoint. More complex operations, such as Query with filters, Scan, or those involving large datasets, may experience increased latency due to additional computation and data transfer requirements.

Locking
Although DynamoDB does not natively support locking, different mechanisms exist. Optimistic locking may use a version number to detect conflicts that occur after updates, rather than preventing them in advance. Pessimistic locking, by contrast, may involve conditional updates with attributes such as lockTime and lockedBy. When combined with Time to Live (TTL), these attributes enable the automated removal of expired locks, potentially enhancing concurrency management in event-driven architectures.

System architecture

Data structures
DynamoDB uses hashing and B-trees to manage data. Upon entry, data is first distributed into different partitions by hashing on the partition key. Each partition can store up to 10 GB of data and handle by default 1,000 write capacity units (WCU) and 3,000 read capacity units (RCU). One RCU represents one strongly consistent read per second or two eventually consistent reads per second for items up to 4 KB in size. One WCU represents one write per second for an item up to 1 KB in size.
To prevent data loss, DynamoDB features a two-tier backup system of replication and long-term storage. Each partition features three nodes, each of which contains a copy of that partition's data. Each node also contains two data structures: a B tree used to locate items, and a replication log that notes all changes made to the node. DynamoDB periodically takes snapshots of these two data structures and stores them for a month in S3 so that engineers can perform point-in-time restores of their databases.
Within each partition, one of the three nodes is designated the "leader node". All write operations travel first through the leader node before propagating, which makes writes consistent in DynamoDB. To maintain its status, the leader sends a "heartbeat" to each other node every 1.5 seconds. Should another node stop receiving heartbeats, it can initiate a new leader election. DynamoDB uses the Paxos algorithm to elect leaders.
Amazon engineers originally avoided Dynamo due to engineering overheads like provisioning and managing partitions and nodes. In response, the DynamoDB team built a service it calls AutoAdmin to manage a database. AutoAdmin replaces a node when it stops responding by copying data from another node. When a partition exceeds any of its three thresholds (RCU, WCU, or 10 GB), AutoAdmin will automatically add additional partitions to further segment the data.
Just like indexing systems in the relational model, DynamoDB demands that any updates to a table be reflected in each of the table's indices. DynamoDB handles this using a service it calls the "log propagator", which subscribes to the replication logs in each node and sends additional Put, Update, and Delete requests to indices as necessary. Because indices result in substantial performance hits for write requests, DynamoDB allows a user at most five of them on any given table.

Query execution

Suppose that a DynamoDB user issues a write operation (a Put, Update, or Delete). While a typical relational system would convert the SQL query to relational algebra and run optimization algorithms, DynamoDB skips both processes. The request arrives at the DynamoDB request router, which authenticates—"Is the request coming from where/whom it claims to be?"—and checks for authorization—"Does the user submitting the request have the requisite permissions?" Assuming these checks pass, the system hashes the request's partition key to arrive in the appropriate partition. There are three nodes within, each with a copy of the partition's data. The system first writes to the leader node, then writes to a second node, then sends a "success" message, and finally continues propagating to the third node. Writes are consistent because they always travel first through the leader node.
Finally, the log propagator propagates the change to all indices. For each index, it grabs that index's primary key value from the item, then performs the same write on that index without log propagation. If the operation is an Update to a preexisting item, the updated attribute may serve as a primary key for an index, and thus the B tree for that index must update as well. B trees only handle insert, delete, and read operations, so in practice, when the log propagator receives an Update operation, it issues both a Delete operation and a Put operation to all indices.
Now suppose that a DynamoDB user issues a Get operation. The request router proceeds as before with authentication and authorization. Next, as above, we hash our partition key to arrive in the appropriate hash. Now, we encounter a problem: with three nodes in eventual consistency with one another, how can we decide which to investigate? DynamoDB offers the user two options when issuing a read: consistent and eventually consistent. A consistent read visits the leader node. But the consistency-availability trade-off rears its head again here: in read-heavy systems, always reading from the leader can overwhelm a single node and reduce availability.
The second option, an eventually consistent read, selects a random node. In practice, this is where DynamoDB trades consistency for availability. If we take this route, what are the odds of an inconsistency? We'd need a write operation to return "success" and begin propagating to the third node, but not finish. We'd also need our Get to target this third node. This means a 1-in-3 chance of inconsistency within the write operation's propagation window. How long is this window? Any number of catastrophes could cause a node to fall behind, but in the vast majority of cases, the third node is up-to-date within milliseconds of the leader.

Controversies
A race condition in a DynamoDB component triggered a 14 hour outage for Amazon's US-EAST-1 region on 19 October 2025, resulting in widespread outages of global services.

See also
Amazon Aurora
Amazon DocumentDB
Amazon Redshift
Amazon Relational Database Service
ScyllaDB (DynamoDB compatible)
Comparison of relational database management systems

References

External links
Official website
Video: AWS re:Invent 2019: [REPEAT 1] Amazon DynamoDB deep dive: Advanced design patterns (DAT403-R1)
