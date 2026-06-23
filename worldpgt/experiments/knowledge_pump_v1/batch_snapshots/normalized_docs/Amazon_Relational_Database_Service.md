# Amazon Relational Database Service

Source: https://en.wikipedia.org/wiki/Amazon_Relational_Database_Service
Retrieved at: 2026-06-22T21:40:09Z
Revision ID: 1342981399
Raw text SHA256: 581bf87163d2a25e86f19613d849340c00654260b5a5dbc064b34c65345fa26a
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
Amazon Relational Database Service (or Amazon RDS) is a distributed relational database service by Amazon Web Services (AWS). It is a web service running "in the cloud" designed to simplify the setup, operation, and scaling of a relational database for use in applications. Administration processes like patching the database software, backing up databases and enabling point-in-time recovery are managed automatically. Scaling storage and compute resources can be performed by a single API call to the AWS control plane on-demand. AWS does not offer an SSH connection to the underlying virtual machine as part of the managed service.

History
Amazon RDS was first released on 26 October 2009, supporting MySQL databases. This was followed by support for Oracle Database in June 2011, Microsoft SQL Server in May 2012, PostgreSQL in November 2013, and MariaDB (a fork of MySQL) in October 2015, and an additional 80 features during 2017.
In November 2014 AWS announced Amazon Aurora, a MySQL-compatible database offering enhanced high availability and performance, and in October 2017 a PostgreSQL-compatible database offering was launched.
In March 2019 AWS announced support of PostgreSQL 11 in RDS, five months after official release.

Features
New database instances can be launched from the AWS Management Console or using the Amazon RDS APIs. Amazon RDS offers different features to support different use cases. Some of the major features are:

Multi-Availability Zone (AZ) deployment
In May 2010 Amazon announced Multi-Availability Zone deployment support. Amazon RDS Multi-Availability Zone (AZ) allows users to automatically provision and maintain a synchronous physical or logical "standby" replica, depending on database engine, in a different Availability Zone (independent infrastructure in a physically separate location). Multi-AZ database instance can be developed at creation time or modified to run as a Multi-AZ deployment later. Multi-AZ deployments aim to provide enhanced availability and data durability for MySQL, MariaDB, Oracle, PostgreSQL and SQL Server instances and are targeted for production environments. In the event of planned database maintenance or unplanned service disruption, Amazon RDS automatically fails over to the up-to-date standby, allowing database operations to resume without administrative intervention.
Multi-AZ RDS instances are optional and have a cost associated with them. When creating a RDS instance, the user is asked if they would like to use a Multi-AZ RDS instance. In Multi-AZ RDS deployments backups are done in the standby instance so I/O activity is not suspended any time but users may experience elevated latencies for a few minutes during backups.

Read replicas
Read replicas allow different use cases such as to scale in for read-heavy database workloads. There are up to five replicas available for MySQL, MariaDB, and PostgreSQL. Instances use the native, asynchronous replication functionality of their respective database engines. They have no backups configured by default and are accessible and can be used for read scaling. MySQL and MariaDB read replicas can be made writeable again since October 2012; PostgreSQL read replicas do not support it. Replicas are done at database instance level and do not support replication at database or table level.

Performance metrics and monitoring
Performance metrics for Amazon RDS are available from the AWS Management Console or the Amazon CloudWatch API. In December 2015, Amazon announced an optional enhanced monitoring feature that provides an expanded set of metrics for the MySQL, MariaDB, and Aurora database engines.

RDS costs
Amazon RDS instances are priced very similarly to Amazon Elastic Compute Cloud (EC2). RDS is charged per hour and comes in two packages: On-Demand DB Instances and Reserved DB Instances. On-Demand Instances are at an ongoing hourly usage rate. Reserved RDS Instances are offered in 1-year and 3-year terms and include no-upfront, partial-upfront, and all-upfront payment options. Currently, AWS does not offer a 3-year reservation with an "no-upfront" payment option.
Apart from the hourly cost of running the RDS instance, users are charged for the amount of storage provisioned, data transfers and input and output operations performed. AWS have introduced Provisioned Input and Output Operations, in which the user can define how many IO per second are required by their application. IOPS can contribute significantly to the total cost of running the RDS instance.
Amazon RDS also has an Aurora Serverless option. The serverless pricing unit is dollars per ACU hour. ACU stands for 'Aurora Capacity Unit'. This option is designed for customers that need to dramatically scale workloads.
As part of the AWS Free Tier, the Amazon RDS Free Tier helps new AWS customers get started with a managed database service in the cloud for free. You can use the Amazon RDS Free Tier to develop new applications, test existing applications, or simply gain hands-on experience with Amazon RDS.

Automatic backups
Amazon RDS creates and saves automated backups of RDS DB instances. The first snapshot of a DB instance contains the data for the full DB instance and subsequent snapshots are incremental, maximum retention period is 35 days. In Multi-AZ RDS deployments backups are done in the standby instance so I/O activity is not suspended for any amount of time but you may experience elevated latencies for a few minutes during backups.

Operation
Database instances can be managed from the AWS Management Console, using the Amazon RDS APIs and using AWS CLI. Since 1 June 2017, you can stop AWS RDS instances from AWS Management Console or AWS CLI for 7 days at a time. After 7 days, it will be automatically started, and since September 2018 RDS instances can be protected from accidental deletion. Increase DB space is supported, but not decrease allocated space. Additionally there is at least a six-hour period where new allocation cannot be done.

Database instance types
As of August 2020, Amazon RDS supports 82 DB instance types - to support different types of workloads:

General Purpose: 31 instances
Memory Optimized: 33 instances
Previous Generation: 18 instances

General purpose

Memory optimized

Previous generation

References

External links
Amazon Relational Database Service - official homepage
Getting Started with Amazon Relational Database Service (Amazon RDS) on YouTube
