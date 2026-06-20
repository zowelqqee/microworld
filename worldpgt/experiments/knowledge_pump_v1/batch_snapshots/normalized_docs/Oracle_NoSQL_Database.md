# Oracle NoSQL Database

Source: https://en.wikipedia.org/wiki/Oracle_NoSQL_Database
Retrieved at: 2026-06-20T08:05:43Z
Revision ID: 1343154156
Raw text SHA256: 9dbc491d2e1ec029ad62fdec30ed91a02d62e051112b1cffe50e99bb8ff03ea0
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
Oracle NoSQL Database is a NoSQL-type distributed key-value database from Oracle Corporation. It provides transactional semantics for data manipulation, horizontal scalability, and simple administration and monitoring.
Oracle NoSQL Database Cloud Service is a managed cloud service for applications that require low latency, flexible data models, and elastic scaling for dynamic workloads.
Developers focus on application development and data store requirements rather than managing back-end servers, storage expansion, cluster deployments, topology, software installation/patches/upgrades, backup, operating systems, and availability. NoSQL database scales to meet dynamic application workloads and throughput requirements.
Users create tables to store their application data and perform database operations. A NoSQL table is similar to a relational table with additional properties including provisioned write units, read units, and storage capacity. Users provision the throughput and storage capacity in each table based on anticipated workloads. NoSQL Database resources are allocated and scaled accordingly to meet workload requirements. Users are billed hourly based on the capacity provisioned.
NoSQL Database supports tabular model. Each row is identified by a unique key, and has a value, of arbitrary length, which is interpreted by the application. The application can manipulate (insert, delete, update, read) a single row in a transaction. The application can also perform an iterative, non-transactional scan of all the rows in the database.

Licensing
Oracle Corporation distributes the Oracle NoSQL Database in three editions:

Oracle NoSQL Database Server Community Edition under an Apache License, Version 2.0
Oracle NoSQL Enterprise Edition under the Oracle Commercial License
Oracle NoSQL Basic Edition
Oracle NoSQL Database is licensed using a freemium model: open-source versions of Oracle NoSQL Community Edition are available, but end-users can purchase additional features and support via the Oracle Store.
Oracle NoSQL Database drivers, licensed pursuant to the Apache 2.0 License, are used with both the community and enterprise editions.

Main features

Architecture
Oracle NoSQL Database is built upon the Oracle Berkeley DB Java Edition high-availability storage engine. It adds services to provide a distributed, highly available key/value store, suited for large-volume, latency-sensitive applications.

Sharding and replication
Oracle NoSQL Database is a client-server, sharded, shared-nothing system. The data in each shard are replicated on each of the nodes that comprise the shard. The major key for a record is hashed to identify the shard that the record belongs to. Oracle NoSQL Database is designed to support changing the number of shards dynamically in response to availability of additional hardware. If the number of shards changes, key-value pairs are redistributed across the new set of shards dynamically, without requiring a system shutdown and restart. A shard is made up of a single electable master node to serve read and write requests, and several replicas (usually two or more) that can serve read requests. Replicas are kept up to date using streaming replication. Each change on the master node is committed locally to disk and also propagated to the replicas.

High availability and fault-tolerance
Oracle NoSQL Database provides single-master, multi-replica database replication. Transactional data is delivered to all replica nodes with flexible durability policies per transaction. In the event the master replica node fails, a consensus-based PAXOS-based automated fail-over election process minimizes downtime. As soon as the failed node is repaired, it rejoins the shard, updated and then becomes available for processing read requests. Thus, Oracle NoSQL Database applications can tolerate failures of nodes within a shard and also multiple failures of nodes in distinct shards.
Proper placement of masters and replicas on server hardware (racks and interconnect switches) by Oracle NoSQL Database is intended to increase availability on commodity servers.

Transparent load balancing
Oracle NoSQL Database Driver partitions the data in real time and evenly distributes it across the storage nodes. It is network topology and latency-aware, routing read and write operations to the most appropriate storage node in order to optimize load distribution and performance.

Administration and system monitoring
Oracle NoSQL Database's administration service can be accessed from a web console or a command-line interface. This service supports functionality such as the ability to configure, start, stop and monitor a storage node, without requiring configuration files, shell scripts, or explicit database operations. It allows Java Management Extensions (JMX) or Simple Network Management Protocol (SNMP) agents to be available for monitoring. This allows management clients to poll information about the status, performance metrics and operational parameters of a storage node and its managed services.

Elastic configuration
"Elasticity" refers to dynamic online expansion of the deployed cluster. Adding storage nodes increases capacity, performance and reliability. Oracle NoSQL Database includes a topology planning feature, with which an administrator can modify the configuration of a NoSQL database while the database is online. The administrator can:

Increase data distribution: by increasing number of shards in the cluster, which increases write throughput.
Increase replication factor: by assigning additional replication nodes to each shard, which increases read throughput and system availability.
Rebalance data store: by modifying the capacity of storage nodes, the system can be rebalanced, re-allocating replication nodes to storage nodes as appropriate.
Administrators can move replication nodes and/or partitions from over-utilized nodes onto underutilized storage nodes or vice versa.

Multi-zone deployment
Oracle NoSQL Database supports multiple zones to intelligently allocate replication of processes and data, in order to improve reliability during hardware, network and power-related failure modes. The two types of zones are: primary zones that contain nodes that can serve as masters or replicas and are typically connected by fast interconnects. Secondary zones contain nodes that can only serve as replicas. Secondary zones can be used to provide low latency read access to data at a distant location, or to offload read-only workloads such as analytics, report generation and data exchange for improved workload management.

JSON data format
Oracle NoSQL Database supports Avro data serialization, which provides a compact, schema-based binary data format. Schemas are defined using JSON. Oracle NoSQL Database supports schema evolution. Configurable Smart Topology System administrators indicate how much capacity is available on a given storage node, allowing more capable nodes to host multiple replication nodes. Once the system knows about the capacity for the storage nodes in a configuration, it automatically allocates replication nodes intelligently. This is intended for better load balancing, better use of system resources and minimizing system impact in the event of storage node failure. Smart Topology supports data centers, ensuring that a full set of replicas is initially allocated to each data center.

Online rolling upgrade
Oracle NoSQL Database provides facilities to perform a rolling upgrade, allowing a system administrator to upgrade cluster nodes while the database remains available.

Fault tolerance
Oracle NoSQL Database is configurable to be either C/P or A/P in CAP. In particular, if writes are configured to be performed synchronously to all replicas, it is C/P in CAP i.e. a partition or node failure causes the system to be unavailable for writes. If replication is performed asynchronously, and reads are configured to be served from any replica, it is A/P in CAP i.e. the system is always available, but there is no guarantee of consistency.

Database features

Table data model
Release 3.0 introduced tabular data structure, which simplifies application data modeling by leveraging existing schema design concepts. Table model is layered on top of the distributed key-value structure, inheriting all its advantages and simplifying application design by enabling seamless integration with familiar SQL-based applications

Secondary index
Primary key only based indexing limits the number of low latency access paths. Sometimes applications need a non-primary-key based paths to support specific application requirements. OND supports secondary index on any value field.

Large object support
Oracle NoSQL Database EE Stream based APIs allow reading and writing large objects (LOBs) such as audio and video files, without having to materialize the entire file in memory. This is intended to decrease the latency of operations across mixed workloads of objects of varying sizes.

ACID compliant transaction
Oracle NoSQL Database provides ACID compliant transactions for full create, read, update and delete (CRUD) operations, with adjustable durability and consistency transaction guarantees. A sequence of operations can operate as a single atomic unit as long as all the affected records share the same major key path.

Integration
Oracle NoSQL Database includes support for Java, C, Python, C# and REST APIs. These allow the application developer to perform CRUD operations. These libraries include Avro support, so that developers can serialize key-value records and de-serialize key-value records interchangeably between C and Java applications.

Oracle RESTful Services
Oracle NoSQL Database supports Oracle REST Data Services (ORDS). This allows customers to build a REST-based application that can access data in either Oracle Database or OND.

GeoJSON
Supports spatial queries on RFC7946 compliant GeoJSON data. Spatial functions and indexing for GeoJSON data are supported.

Apache Hadoop
KVAvroInputFormat and KVInputFormat classes are available to read data from OND natively into Hadoop MapReduce jobs. One use for this class is to read NoSQL database records into Oracle Loader for Hadoop.

Oracle integration

Oracle Big Data SQL and Hive
Oracle Big Data SQL is a common SQL access layer to data stored in Hadoop, HDFS, Hive and OND. This allows customers to query Oracle NoSQL Data from Hive or Oracle Database. Users can run MapReduce jobs against data stored in OND that is configured for secure access. The latest release also supports both primitive and complex data types

Oracle Database
Oracle NoSQL Database EE supports external table allows fetching Oracle NoSQL data from Oracle database using SQL statements such as Select, Select Count(*) etc. Once NoSQL data is exposed through external tables, one can access the data via standard JDBC drivers and/or visualize it through enterprise business intelligence tools.

Other Oracle products
Oracle Event Processing (OEP) provides read access to Oracle NoSQL Database via the NoSQL Database cartridge. Once the cartridge is configured, CQL queries can be used. Oracle Semantic Graph includes a Jena Adapter for Oracle NoSQL Database to store large volumes of RDF data (as triplets/quadruplets). This adapter enables fast access to graph data stored in OND via SPARQL queries. Integration with Oracle Coherence allows OND to be used as a cache for Oracle Coherence applications, allowing applications to directly access cached data from OND.

Enterprise security
Oracle NoSQL Database EE supports OS-independent, cluster-wide password-based user authentication and Oracle Wallet integration and enables greater protection from unauthorized access to sensitive data. Additionally, session-level Secure Sockets Layer (SSL) encryption and network port restrictions improve protection from network intrusion.

Performance
The Oracle NoSQL Database team has worked with several key Oracle partners, including Intel and Cisco, performing Yahoo! Cloud Serving Benchmarks (YCSB) on various hardware configurations, and published its results. For example, in 2012 Oracle reported that Oracle NoSQL Database exceeded 1 million mixed YCSB Ops/Sec.

See also
Database scalability
List of NoSQL software and tools

References
